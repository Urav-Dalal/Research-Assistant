"""
arxiv_agent.py — Agentic ArXiv search layer for the Research Assistant.

This module sits BETWEEN the /chat endpoint and the existing RAG pipeline.
It adds two agentic capabilities:

  1. Intent Detection
     Analyses the user's message and current retrieval confidence to decide
     whether the question can be answered from the local Qdrant store, or
     whether it needs live ArXiv results.

  2. ArXiv Search + User-Confirmed Ingest
     When ArXiv search is triggered, results are returned to the user for
     review BEFORE any ingestion happens. The user explicitly calls
     POST /papers/ingest-arxiv to add a paper to their library.
     This is the "human in the loop" pattern — the system never silently
     adds documents the user didn't approve.

Flow:
  User message
       ↓
  route_query()          ← decides: RAG or ArXiv?
       ↓            ↓
  existing          search_arxiv()
  rag_query()            ↓
                    return previews to user
                         ↓
                    user calls /papers/ingest-arxiv
                         ↓
                    download + process_pdf() + store in Qdrant
                         ↓
                    future queries hit local store (cache-on-confirm)
"""

from __future__ import annotations

from http import client
import logging
import os
import re
import tempfile
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import arxiv
import httpx
from groq import Groq

from .crud import create_paper
from .retrieval import Message as RAGMessage
from .retrieval import RAGResult
from .retrieval import query as rag_query
from .upload_pipeline import process_pdf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Model used for intent classification and synthesis.
# 70b is required here — 8b is not reliable enough for tool-use decisions.
AGENT_LLM_MODEL = "llama-3.3-70b-versatile"

# Number of ArXiv results to fetch per search.
ARXIV_MAX_RESULTS = 5

# If the top RAG retrieval score is below this, we consider local context
# insufficient and trigger ArXiv search as a fallback.
WEAK_RETRIEVAL_THRESHOLD = 0.45

# Explicit phrases that signal the user wants to search for new papers,
# regardless of local retrieval confidence.
EXPLICIT_SEARCH_PHRASES = [
    "find me recent",
    "find papers",
    "find me papers",
    "search for",
    "recent work on",
    "recent papers",
    "latest research",
    "other approaches",
    "other papers",
    "related papers",
    "literature on",
    "papers on",
    "articles about",
    "what does the literature say",
    "what have others done",
    "compare with other",
    "how do others",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class QueryIntent(str, Enum):
    """
    The two possible routing decisions for a user query.
    RAG_LOCAL  → answer from uploaded papers already in Qdrant.
    ARXIV_SEARCH → fetch live results from ArXiv API.
    """
    RAG_LOCAL    = "rag_local"
    ARXIV_SEARCH = "arxiv_search"


@dataclass
class ArxivPaper:
    """
    A single ArXiv search result, ready to be shown to the user
    or passed to the ingest pipeline.
    """
    arxiv_id:  str           # e.g. "2301.07041"
    title:     str
    authors:   list[str]
    summary:   str           # abstract, truncated for display
    pdf_url:   str           # direct link to the PDF
    published: str           # ISO date string "YYYY-MM-DD"


@dataclass
class AgentResult:
    """
    Unified return type for route_query().
    Only one of `rag_result` or `arxiv_papers` will be populated,
    depending on the routing decision.
    """
    intent:       QueryIntent
    rag_result:   Optional[RAGResult]          = None
    arxiv_papers: list[ArxivPaper]             = field(default_factory=list)
    message:      str                          = ""   # human-readable status


# ---------------------------------------------------------------------------
# Intent detection
# ---------------------------------------------------------------------------

def _has_explicit_search_intent(message: str) -> bool:
    """
    Check whether the user's message contains an explicit phrase that
    signals they want to search for new papers rather than query existing ones.

    Uses simple substring matching — fast, zero LLM cost, no latency.
    This covers the majority of search-intent queries in practice.
    """
    lowered = message.lower()
    return any(phrase in lowered for phrase in EXPLICIT_SEARCH_PHRASES)


def _classify_intent_with_llm(message: str) -> QueryIntent:
    """
    Use the LLM to classify intent for ambiguous queries that don't match
    explicit phrases. Called only when rule-based detection is inconclusive.

    Returns QueryIntent.ARXIV_SEARCH or QueryIntent.RAG_LOCAL.
    Falls back to RAG_LOCAL on any LLM error to avoid breaking the
    main conversation flow.
    """
    groq_client = Groq()

    try:
        response = groq_client.chat.completions.create(
            model=AGENT_LLM_MODEL,
            max_tokens=10,
            temperature=0.0,    # deterministic classification
            messages=[{
                "role": "user",
                "content": (
                    "Classify this query. Reply with ONLY one word: "
                    "'search' if the user wants to find new papers/articles, "
                    "'local' if they are asking about documents they already have.\n\n"
                    f"Query: {message}"
                ),
            }],
        )
        label = response.choices[0].message.content.strip().lower()
        logger.debug("LLM intent classification: '%s'", label)

        if "search" in label:
            return QueryIntent.ARXIV_SEARCH
        return QueryIntent.RAG_LOCAL

    except Exception as exc:
        # Never let intent classification crash the whole request.
        logger.warning("LLM intent classification failed (%s). Defaulting to RAG.", exc)
        return QueryIntent.RAG_LOCAL


def detect_intent(message: str, top_rag_score: float | None = None) -> QueryIntent:
    """
    Master intent detection function. Uses a two-stage approach:

    Stage 1 — Rule-based (fast, free):
      If the message contains explicit search phrases → ARXIV_SEARCH immediately.

    Stage 2 — Score-based fallback:
      If RAG retrieval returned weak results (score below threshold),
      route to ArXiv even if the user didn't explicitly ask to search.
      This is the self-correcting behaviour that makes the system agentic.

    Stage 3 — LLM classification (for genuinely ambiguous cases):
      Ask the LLM to classify intent. Only reached if stages 1 and 2
      don't give a clear signal.

    Args:
        message:       The user's latest message.
        top_rag_score: Highest cosine similarity score from a prior RAG
                       attempt. Pass None to skip score-based routing.
    """
    # ── Stage 1: explicit phrase matching ─────────────────────────────────────
    if _has_explicit_search_intent(message):
        logger.info("Explicit search intent detected in query.")
        return QueryIntent.ARXIV_SEARCH

    # ── Stage 2: weak retrieval confidence ────────────────────────────────────
    if top_rag_score is not None and top_rag_score < WEAK_RETRIEVAL_THRESHOLD:
        logger.info(
            "Weak RAG score (%.4f < %.2f). Routing to ArXiv.",
            top_rag_score, WEAK_RETRIEVAL_THRESHOLD,
        )
        return QueryIntent.ARXIV_SEARCH

    # ── Stage 3: LLM classification for ambiguous queries ─────────────────────
    return _classify_intent_with_llm(message)


# ---------------------------------------------------------------------------
# ArXiv search
# ---------------------------------------------------------------------------

def search_arxiv(query: str, max_results: int = ARXIV_MAX_RESULTS) -> list[ArxivPaper]:
    """
    Search the ArXiv API and return structured paper previews.
    Uses arxiv v2+ API: arxiv.Client().results(search) instead of search.results()
    """
    try:
        client = arxiv.Client()        # v2+ requires explicit client
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        papers = []
        for result in client.results(search):      # v2+ way to iterate results
            truncated_summary = (
                result.summary[:500] + "..."
                if len(result.summary) > 500
                else result.summary
            )
            papers.append(ArxivPaper(
                arxiv_id=result.entry_id.split("/")[-1],
                title=result.title,
                authors=[a.name for a in result.authors[:5]],
                summary=truncated_summary,
                pdf_url=result.pdf_url,
                published=result.published.strftime("%Y-%m-%d"),
            ))

        logger.info("ArXiv search for '%s' returned %d results.", query, len(papers))
        return papers

    except Exception as exc:
        logger.error("ArXiv search failed for query '%s': %s", query, exc)
        return []

def _rewrite_for_arxiv(user_message: str,context_hint: str = "") -> str:
    """
    Rewrite a conversational user message into a clean ArXiv search query.

    ArXiv search works best with technical keyword phrases rather than
    natural language questions. We use the LLM to do this translation,
    falling back to the raw message if the rewrite fails.
    """
    groq_client = Groq()
    try:
        response = groq_client.chat.completions.create(
            model=AGENT_LLM_MODEL,
            max_tokens=40,
            temperature=0.0,
            messages=[{
                "role": "user",
                "content": (
                    "Rewrite this as a SHORT ArXiv search query (4-7 technical keywords). "
                    "Be specific — include the exact technical topic, not generic terms like 'deep learning' or 'neural network'. "
                    "Return ONLY the query, no punctuation, no explanation.\n\n"
                    f"User message: {user_message}"
                    + (f"\nContext: {context_hint}" if context_hint else "")
                ),
            }],
        )
        rewritten = response.choices[0].message.content.strip()
        if len(rewritten.split()) > 10 or len(rewritten) < 3:
            return user_message
        logger.info("ArXiv query rewritten: '%s' → '%s'", user_message, rewritten)
        return rewritten
    except Exception as exc:
        logger.warning("ArXiv query rewrite failed (%s). Using raw message.", exc)
        return user_message


# ---------------------------------------------------------------------------
# User-confirmed ingest
# ---------------------------------------------------------------------------

def ingest_arxiv_paper(
    arxiv_id: str,
    login_id: str,
) -> dict:
    """
    Download an ArXiv PDF by ID and run it through the existing ingestion
    pipeline (process_pdf → Qdrant + SQLite).

    This is called ONLY after the user explicitly confirms they want to
    add a paper to their library — never automatically.

    Args:
        arxiv_id:  The ArXiv paper ID, e.g. "2301.07041".
        login_id:  The user's login ID, used to scope the Qdrant vectors.

    Returns a dict with ingestion metadata on success.
    Raises RuntimeError with a descriptive message on failure.
    """

    # ── 1. Fetch paper metadata from ArXiv ────────────────────────────────────
    try:
        client = arxiv.Client()
        search = arxiv.Search(id_list=[arxiv_id], max_results=1)
        results = list(client.results(search))
        if not results:
            raise RuntimeError(f"ArXiv paper '{arxiv_id}' not found.")
        paper_meta = results[0]
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch ArXiv metadata for '{arxiv_id}': {exc}") from exc

    # ── 2. Download the PDF to a temp file ────────────────────────────────────
    # We use a temp file so we don't pollute the uploads directory with
    # papers the user hasn't yet confirmed they want (they already confirmed
    # by calling this endpoint, but temp keeps cleanup automatic).
    pdf_filename = f"{arxiv_id.replace('/', '_')}.pdf"

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = os.path.join(tmp_dir, pdf_filename)

            # Download with a timeout to avoid hanging on slow connections.
            with httpx.Client(timeout=60.0, follow_redirects=True) as http:
                response = http.get(paper_meta.pdf_url)
                response.raise_for_status()
                with open(pdf_path, "wb") as f:
                    f.write(response.content)

            logger.info("Downloaded ArXiv PDF '%s' (%d bytes).", arxiv_id, len(response.content))

            # ── 3. Run through the existing ingestion pipeline ─────────────────
            # process_pdf() handles: Docling parsing → chunking → embedding → Qdrant upsert
            # We reuse it exactly as-is — the agent layer adds no new ingestion logic.
            paper_id = str(uuid.uuid4())

            chunks_stored = process_pdf(
                pdf_path=pdf_path,
                filename=pdf_filename,
                paper_id=paper_id,
                login_id=login_id,
            )

            # ── 4. Record in SQLite (same as manual upload flow) ───────────────
            create_paper(
                paper_id=paper_id,
                user_id=login_id,
                filename=pdf_filename,
            )

            logger.info(
                "Ingested ArXiv paper '%s' → paper_id=%s, chunks=%d",
                arxiv_id, paper_id, chunks_stored,
            )

            return {
                "paper_id":     paper_id,
                "arxiv_id":     arxiv_id,
                "title":        paper_meta.title,
                "filename":     pdf_filename,
                "chunks_stored": chunks_stored,
            }

    except RuntimeError:
        raise   # re-raise our own errors unchanged
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download or ingest ArXiv paper '{arxiv_id}': {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Master router — public API called by /chat
# ---------------------------------------------------------------------------

def route_query(
    conversation: list[RAGMessage],
    login_id: str,
    paper_id: Optional[str] = None,
) -> AgentResult:
    """
    Main entry point for the agent layer.

    Strategy (order matters):
      1. Check explicit intent FIRST — if user clearly wants to search,
         skip RAG entirely. High local scores must never override an
         explicit search request.
      2. If no explicit intent, try RAG. If score is strong → return it.
      3. If RAG score is weak → fall back to ArXiv automatically.
    """
    if not conversation:
        raise ValueError("conversation must contain at least one message.")

    user_messages = [m for m in conversation if m.role == "user"]
    if not user_messages:
        raise ValueError("conversation must contain at least one user message.")

    latest_message = user_messages[-1].content

    # ── Step 1: Check explicit search intent BEFORE running RAG ───────────────
    # This must come first — we never want high RAG scores to suppress
    # an explicit user request to find new papers.
    if _has_explicit_search_intent(latest_message):
        logger.info("Explicit search intent detected — skipping RAG, going to ArXiv.")
        arxiv_query = _rewrite_for_arxiv(latest_message)
        papers      = search_arxiv(arxiv_query)

        if not papers:
            # ArXiv returned nothing — fall back to RAG as last resort.
            logger.warning("ArXiv returned no results. Falling back to RAG.")
            rag_result = rag_query(
                conversation=conversation,
                login_id=login_id,
                paper_id=paper_id,
            )
            return AgentResult(
                intent=QueryIntent.RAG_LOCAL,
                rag_result=rag_result,
                message="ArXiv returned no results. Here's what I found locally:",
            )

        return AgentResult(
            intent=QueryIntent.ARXIV_SEARCH,
            arxiv_papers=papers,
            message=(
                f"Found {len(papers)} papers on ArXiv. "
                "Add any to your library via POST /papers/ingest-arxiv."
            ),
        )

    # ── Step 2: No explicit search intent — try RAG first ─────────────────────
    rag_result = None
    top_score  = None

    try:
        rag_result = rag_query(
            conversation=conversation,
            login_id=login_id,
            paper_id=paper_id,
        )
        if rag_result.sources:
            top_score = max(s.get("score", 0.0) for s in rag_result.sources)
            logger.info("RAG top score: %.4f", top_score)

    except Exception as exc:
        logger.warning("RAG query failed (%s). Will attempt ArXiv fallback.", exc)

    # ── Step 3: Strong local retrieval → return RAG answer ────────────────────
    if rag_result is not None and (top_score is None or top_score >= WEAK_RETRIEVAL_THRESHOLD):
        logger.info("Routing decision: RAG_LOCAL (score=%.4f)", top_score or 0)
        return AgentResult(
            intent=QueryIntent.RAG_LOCAL,
            rag_result=rag_result,
            message="Answered from your uploaded papers.",
        )

    # ── Step 4: Weak retrieval — self-correct by searching ArXiv ──────────────
    logger.info("Weak RAG score (%.4f). Routing to ArXiv.", top_score or 0)
    arxiv_query = _rewrite_for_arxiv(latest_message)
    papers      = search_arxiv(arxiv_query)

    if not papers:
        # Nothing on ArXiv either — return whatever RAG had.
        return AgentResult(
            intent=QueryIntent.RAG_LOCAL,
            rag_result=rag_result,
            message="Local context was thin and ArXiv returned nothing. Best available answer:",
        )

    return AgentResult(
        intent=QueryIntent.ARXIV_SEARCH,
        arxiv_papers=papers,
        message=(
            f"Local context was insufficient. "
            f"Found {len(papers)} related papers on ArXiv."
        ),
    )