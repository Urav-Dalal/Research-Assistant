"""
PDF ingestion pipeline for RAG.

Processing stages:
  1. Convert PDF to structured Markdown via Docling (fallback: PyMuPDF plain text)
  2. Parse Markdown headings into a section hierarchy
  3. Embed sections using batched SentenceTransformer inference
  4. Upsert parent + child chunk vectors into Qdrant in safe batches
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from typing import Generator

import fitz
from docling.document_converter import DocumentConverter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer

from .qdrant import COLLECTION_NAME, client

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBED_MODEL_NAME    = "BAAI/bge-small-en-v1.5"
CHILD_CHUNK_SIZE    = 800
CHILD_CHUNK_OVERLAP = 150
UPSERT_BATCH_SIZE   = 100   # max points per Qdrant upsert call
EMBED_BATCH_SIZE    = 64    # max texts per SentenceTransformer encode call

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy-loaded singletons  (no cost until first call)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Load the embedding model exactly once per process."""
    logger.info("Loading SentenceTransformer model: %s", EMBED_MODEL_NAME)
    return SentenceTransformer(EMBED_MODEL_NAME)


@lru_cache(maxsize=1)
def _get_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
    )


# ---------------------------------------------------------------------------
# Document parsing  (Docling primary, PyMuPDF fallback)
# ---------------------------------------------------------------------------

def _parse_with_docling(pdf_path: str) -> str:
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    return result.document.export_to_markdown()


def _parse_with_pymupdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text() for page in doc)


def _extract_markdown(pdf_path: str) -> tuple[str, str]:
    """
    Try Docling first; fall back to PyMuPDF on any error.
    Returns (markdown_text, method_used).
    """
    try:
        text = _parse_with_docling(pdf_path)
        logger.info("Parsed '%s' with Docling.", pdf_path)
        return text, "docling"
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Docling failed for '%s' (%s). Falling back to PyMuPDF.", pdf_path, exc
        )
        return _parse_with_pymupdf(pdf_path), "pymupdf"


# ---------------------------------------------------------------------------
# Section hierarchy parsing
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)")


def _parse_markdown_sections(markdown_text: str) -> list[dict]:
    """
    Walk Markdown line-by-line, building a flat list of section dicts.

    Key design decision: each section's `body` contains *only* its own
    direct text — not any descendant sections. This keeps parent-chunk
    embeddings semantically focused rather than diluted by the whole subtree.
    """
    root: dict = {
        "section_id": str(uuid.uuid4()),
        "title": "Document",
        "level": 0,
        "path": ["Document"],
        "body": "",
        "parent_section_id": None,
    }
    sections: list[dict] = [root]
    stack: list[dict] = [root]   # ancestry chain, innermost last

    for line in markdown_text.splitlines(keepends=True):
        m = _HEADING_RE.match(line)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()

            # Unwind the stack to find the correct parent.
            while stack and stack[-1]["level"] >= level:
                stack.pop()

            parent = stack[-1] if stack else root
            section: dict = {
                "section_id": str(uuid.uuid4()),
                "title": title,
                "level": level,
                "path": parent["path"] + [title],
                "body": "",
                "parent_section_id": parent["section_id"],
            }
            sections.append(section)
            stack.append(section)
        else:
            # Non-heading text belongs only to the innermost open section.
            if stack:
                stack[-1]["body"] += line

    return sections


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def _batch_encode(texts: list[str]) -> list[list[float]]:
    """Encode a list of texts in configurable batches."""
    if not texts:
        return []
    model = _get_model()
    vectors = model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=True,  # unit vectors → cosine ≡ dot product
    )
    return [v.tolist() for v in vectors]


# ---------------------------------------------------------------------------
# Point construction
# ---------------------------------------------------------------------------

def _build_points(
    sections: list[dict],
    filename: str,
    paper_id: str,
    login_id: str,
) -> list[PointStruct]:
    """
    Build all Qdrant PointStructs in two batched encode passes —
    one for parent texts, one for child chunks — instead of calling
    model.encode() once per chunk inside a loop.
    """
    splitter  = _get_splitter()
    now_iso   = datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # 1. Collect all texts that need embedding before touching the model.
    # ------------------------------------------------------------------

    # Parent: embed the section body (or just the title if body is empty)
    parent_records: list[dict] = []
    for idx, section in enumerate(sections):
        text = section["body"].strip() or section["title"]
        parent_records.append({
            "section":       section,
            "section_index": idx,
            "text":          text,
        })

    # Children: split each non-empty body into sub-chunks
    child_records: list[dict] = []
    for idx, section in enumerate(sections):
        if not section["body"].strip():
            continue
        for offset, chunk in enumerate(splitter.split_text(section["body"])):
            child_records.append({
                "section":       section,
                "section_index": idx,
                "chunk_offset":  offset,
                "text":          chunk,
            })

    # ------------------------------------------------------------------
    # 2. Two batched encode calls (not N calls).
    # ------------------------------------------------------------------
    parent_vectors = _batch_encode([r["text"] for r in parent_records])
    child_vectors  = _batch_encode([r["text"] for r in child_records])

    # ------------------------------------------------------------------
    # 3. Assemble PointStructs.
    # ------------------------------------------------------------------
    points: list[PointStruct] = []

    for vector, rec in zip(parent_vectors, parent_records):
        section   = rec["section"]
        point_id  = section["section_id"]
        points.append(PointStruct(
            id=point_id,
            vector=vector,
            payload={
                "login_id":         login_id,
                "paper_id":         paper_id,
                "source":           filename,
                "chunk_type":       "parent",
                "parent_chunk_id":  None,
                "chunk_id":         point_id,
                "section_title":    section["title"],
                "section_path":     " > ".join(section["path"]),
                "section_level":    section["level"],
                "section_parent_id": section["parent_section_id"],
                "chunk_index":      rec["section_index"],
                "text":             rec["text"],
                "upload_timestamp": now_iso,
            },
        ))

    for vector, rec in zip(child_vectors, child_records):
        section  = rec["section"]
        child_id = str(uuid.uuid4())
        points.append(PointStruct(
            id=child_id,
            vector=vector,
            payload={
                "login_id":         login_id,
                "paper_id":         paper_id,
                "source":           filename,
                "chunk_type":       "child",
                "parent_chunk_id":  section["section_id"],
                "chunk_id":         child_id,
                "section_title":    section["title"],
                "section_path":     " > ".join(section["path"]),
                "section_level":    section["level"],
                "section_parent_id": section["parent_section_id"],
                "chunk_index":      f"{rec['section_index']}.{rec['chunk_offset']}",
                "text":             rec["text"],
                "upload_timestamp": now_iso,
            },
        ))

    return points


# ---------------------------------------------------------------------------
# Batched Qdrant upsert
# ---------------------------------------------------------------------------

def _batched(items: list, size: int) -> Generator[list, None, None]:
    """Yield successive fixed-size slices of *items*."""
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _upsert_points(points: list[PointStruct]) -> None:
    """Upsert into Qdrant in safe batches to avoid payload-size timeouts."""
    for batch in _batched(points, UPSERT_BATCH_SIZE):
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
    logger.info("Upserted %d points into '%s'.", len(points), COLLECTION_NAME)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_pdf(pdf_path: str, filename: str, paper_id: str, login_id: str) -> int:
    """
    Full ingestion pipeline: PDF → Markdown → sections → vectors → Qdrant.
    Returns the total number of points upserted.
    """
    markdown_text, parse_method = _extract_markdown(pdf_path)
    logger.info("Extracted text via %s from '%s'.", parse_method, filename)

    sections = _parse_markdown_sections(markdown_text)
    logger.info("Parsed %d sections from '%s'.", len(sections), filename)

    points = _build_points(sections, filename, paper_id, login_id)
    logger.info("Built %d embedding points for '%s'.", len(points), filename)

    _upsert_points(points)
    return len(points)