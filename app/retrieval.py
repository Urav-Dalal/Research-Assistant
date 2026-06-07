"""
RAG query pipeline.

Given a conversation (list of messages) and optional filters, this module:
  1. Embeds the latest user question
  2. Retrieves top-k child chunks from Qdrant
  3. Fetches their parent chunks for richer context
  4. Deduplicates and ranks context by relevance score
  5. Builds a prompt and calls Groq (llama-3.1-8b-instant) to generate an answer
  6. Returns the answer plus the source chunks used
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from groq import Groq
from qdrant_client.models import Filter, FieldCondition, MatchValue

from .upload_pipeline import _get_model, _batch_encode
from .qdrant import COLLECTION_NAME, client

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RETRIEVAL_TOP_K      = 8    # child chunks to retrieve from Qdrant
MAX_CONTEXT_CHUNKS   = 5    # deduplicated parent chunks passed to LLM
LLM_MODEL            = "llama-3.1-8b-instant"   # free on Groq — swap to llama-3.3-70b-versatile for higher quality
LLM_MAX_TOKENS       = 1024
MIN_RELEVANCE_SCORE  = 0.30  # discard chunks below this cosine similarity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Message:
    """A single turn in a conversation."""
    role: str       # "user" or "assistant"
    content: str


@dataclass
class RAGResult:
    """Return type of query()."""
    answer: str
    sources: list[dict] = field(default_factory=list)   # metadata of chunks used
    query: str = ""


# ---------------------------------------------------------------------------
# Qdrant helpers
# ---------------------------------------------------------------------------

def _build_filter(login_id: str | None, paper_id: str | None) -> Filter | None:
    """Build a Qdrant filter from whichever scope params are provided."""
    conditions = []
    if login_id:
        conditions.append(FieldCondition(key="login_id", match=MatchValue(value=login_id)))
    if paper_id:
        conditions.append(FieldCondition(key="paper_id", match=MatchValue(value=paper_id)))
    if not conditions:
        return None
    return Filter(must=conditions)


def _retrieve_child_chunks(
    query_vector: list[float],
    top_k: int,
    qdrant_filter: Filter | None,
) -> list[dict]:
    """Search Qdrant for the most relevant child chunks."""
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=qdrant_filter,
        limit=top_k,
        with_payload=True,
        score_threshold=MIN_RELEVANCE_SCORE,
    )
    return [
        {"score": hit.score, "payload": hit.payload}
        for hit in results.points
        if hit.payload.get("chunk_type") == "child"
    ]


def _fetch_parent_chunks(parent_ids: list[str]) -> dict[str, dict]:
    """
    Retrieve parent PointStructs by ID from Qdrant.
    Returns a mapping of parent_chunk_id → payload.
    """
    if not parent_ids:
        return {}
    results = client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=parent_ids,
        with_payload=True,
    )
    return {str(r.id): r.payload for r in results}


def _collect_context(child_hits: list[dict], max_chunks: int) -> list[dict]:
    """
    For each child hit, swap in the parent chunk text for richer context.
    Deduplicate by parent_id and keep the highest-scoring hit per parent.
    Returns up to max_chunks entries sorted by score descending.
    """
    # Best score seen per parent_id
    best: dict[str, dict] = {}
    for hit in child_hits:
        pid = hit["payload"].get("parent_chunk_id")
        if pid is None:
            continue
        if pid not in best or hit["score"] > best[pid]["score"]:
            best[pid] = hit

    # Fetch all parent payloads in one round-trip
    parent_payloads = _fetch_parent_chunks(list(best.keys()))

    context_chunks = []
    for pid, hit in best.items():
        parent_payload = parent_payloads.get(pid, {})
        context_chunks.append({
            "score":          hit["score"],
            "text":           parent_payload.get("text") or hit["payload"].get("text", ""),
            "section_title":  hit["payload"].get("section_title", ""),
            "section_path":   hit["payload"].get("section_path", ""),
            "source":         hit["payload"].get("source", ""),
            "paper_id":       hit["payload"].get("paper_id", ""),
            "chunk_id":       pid,
        })

    context_chunks.sort(key=lambda x: x["score"], reverse=True)
    return context_chunks[:max_chunks]


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _build_system_prompt(context_chunks: list[dict]) -> str:
    """Construct the system prompt with injected context passages."""
    passages = "\n\n".join(
        f"[{i+1}] (Section: {c['section_path']} | Source: {c['source']})\n{c['text']}"
        for i, c in enumerate(context_chunks)
    )
    return f"""You are a precise research assistant. Answer the user's question using ONLY the context passages below.

Rules:
- Cite passages by their number, e.g. [1] or [2][3].
- If the context does not contain enough information to answer, say so clearly — do not hallucinate.
- Be concise and factual.

CONTEXT:
{passages}"""


def _build_messages(conversation: list[Message]) -> list[dict]:
    """Convert internal Message objects to the Anthropic API message format."""
    return [{"role": m.role, "content": m.content} for m in conversation]


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _call_groq(system_prompt: str, messages: list[dict]) -> str:
    """Send a request to Groq and return the text response."""
    groq_client = Groq()   # reads GROQ_API_KEY from env automatically
    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=LLM_MAX_TOKENS,
        messages=[{"role": "system", "content": system_prompt}, *messages],
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def query(
    conversation: list[Message],
    login_id: str | None = None,
    paper_id: str | None = None,
) -> RAGResult:
    """
    Run a full RAG cycle over a conversation and return a grounded answer.

    Args:
        conversation:  Full chat history. The last Message with role="user"
                       is used as the search query.
        login_id:      Optional — scope retrieval to this user's documents.
        paper_id:      Optional — scope retrieval to a specific paper.

    Returns:
        RAGResult with the answer text and the source chunks used.
    """
    if not conversation:
        raise ValueError("conversation must contain at least one message.")

    # Extract the most recent user turn as the query.
    user_messages = [m for m in conversation if m.role == "user"]
    if not user_messages:
        raise ValueError("conversation must contain at least one user message.")
    query_text = user_messages[-1].content

    # ------------------------------------------------------------------
    # 1. Embed the query
    # ------------------------------------------------------------------
    query_vector = _batch_encode([query_text])[0]
    logger.info("Query: '%s'", query_text[:80])

    # ------------------------------------------------------------------
    # 2. Retrieve child chunks
    # ------------------------------------------------------------------
    qdrant_filter = _build_filter(login_id, paper_id)
    child_hits = _retrieve_child_chunks(query_vector, RETRIEVAL_TOP_K, qdrant_filter)
    logger.info("Retrieved %d child chunks above score threshold.", len(child_hits))

    if not child_hits:
        return RAGResult(
            answer="I could not find any relevant information in the uploaded documents for your question.",
            sources=[],
            query=query_text,
        )

    # ------------------------------------------------------------------
    # 3. Fetch parent chunks and deduplicate
    # ------------------------------------------------------------------
    context_chunks = _collect_context(child_hits, MAX_CONTEXT_CHUNKS)
    logger.info("Using %d context chunks for generation.", len(context_chunks))

    # ------------------------------------------------------------------
    # 4. Build prompt + call LLM
    # ------------------------------------------------------------------
    system_prompt = _build_system_prompt(context_chunks)
    messages      = _build_messages(conversation)
    answer        = _call_groq(system_prompt, messages)

    return RAGResult(
        answer=answer,
        sources=[
            {
                "section_path": c["section_path"],
                "source":       c["source"],
                "paper_id":     c["paper_id"],
                "score":        round(c["score"], 4),
            }
            for c in context_chunks
        ],
        query=query_text,
    )