import os
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from .upload_pipeline import process_pdf
from .qdrant import create_collection
from .retrieval import query as rag_query, Message as RAGMessage

from .database import engine
from .models import Base
from .crud import create_paper

from .database import SessionLocal
from .models import Paper

from .arxiv_agent import route_query, ingest_arxiv_paper, QueryIntent

# FastAPI app entrypoint for the `app` package.
# When using `uvicorn app.main:app --reload`, this module is loaded as part
# of the `app` package, so relative imports are required.
app = FastAPI()
Base.metadata.create_all(bind=engine)

# Directory where uploaded PDF files are temporarily written before processing.
UPLOAD_DIR = "temp"

os.makedirs(UPLOAD_DIR, exist_ok=True)

# Ensure the Qdrant collection exists before accepting uploads.
create_collection()

@app.get("/")
def health_check():
    """Simple health check endpoint."""
    return {"status": "running"}


class ChatRequest(BaseModel):
    login_id: str
    message: Optional[str] = None
    messages: Optional[List[dict]] = None
    paper_id: Optional[str] = None


@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Accept a user message and return a grounded answer.
 
    The agent layer (route_query) now sits in front of the RAG pipeline:
      - If local retrieval is confident → answers from uploaded papers (unchanged behaviour)
      - If query needs new papers → returns ArXiv search results for user review
      - If user explicitly searches → returns ArXiv results immediately
 
    The response shape has one new field: `intent`
      "rag_local"    → same as before, answer + sources returned
      "arxiv_search" → arxiv_papers list returned instead, no answer yet
    """
    # Build conversation list for retrieval.query
    conversation: List[RAGMessage] = []
    # Only use messages if they actually contain valid content
    valid_messages = [
    m for m in (req.messages or [])
    if m.get("role") and m.get("content", "").strip()
    ]

    if valid_messages:
        for m in valid_messages:
            conversation.append(RAGMessage(role=m["role"], content=m["content"]))

    elif req.message:
        conversation.append(RAGMessage(role="user", content=req.message))
    else:
        raise HTTPException(status_code=400, detail="Either `message` or `messages` is required")

    # Call the RAG pipeline
    try:
        result = route_query(
            conversation=conversation,
            login_id=req.login_id,
            paper_id=req.paper_id or None,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if result.intent == QueryIntent.RAG_LOCAL:
        # Identical response shape to what you had before — nothing breaks.
        return {
            "intent":  "rag_local",
            "answer":  result.rag_result.answer,
            "sources": result.rag_result.sources,
            "query":   result.rag_result.query,
            "message": result.message,
        }
 
    else:
        # ArXiv results returned — user reviews and calls /papers/ingest-arxiv
        return {
            "intent": "arxiv_search",
            "message": result.message,
            "arxiv_papers": [
                {
                    "arxiv_id":  p.arxiv_id,
                    "title":     p.title,
                    "authors":   p.authors,
                    "summary":   p.summary,
                    "pdf_url":   p.pdf_url,
                    "published": p.published,
                }
                for p in result.arxiv_papers
            ],
            "next_step": (
                "Pick a paper from arxiv_papers and call "
                "POST /papers/ingest-arxiv with its arxiv_id to add it to your library."
            ),
        }

@app.post("/upload-pdf")
async def upload_pdf(login_id: str,file: UploadFile = File(...)):
    """Accept an uploaded PDF, save it locally, process it, and return metadata."""

    # Save the uploaded file to the local temp directory.
    file_path = f"{UPLOAD_DIR}/{file.filename}"

    with open(file_path, "wb") as f:
        f.write(await file.read())

    paper_id = str(uuid.uuid4())
    
    create_paper(
    paper_id=paper_id,
    user_id=login_id,
    filename=file.filename
)

    # Process the PDF content and store embeddings in Qdrant.
    chunks_stored = process_pdf(
    pdf_path=file_path,
    filename=file.filename,
    login_id=login_id,
    paper_id=paper_id
)
    return {
        "message": "PDF processed successfully",
        "filename": file.filename,
        "chunks_stored": chunks_stored,
    }
    
@app.get("/papers")
def get_papers():
    db = SessionLocal()

    try:
        papers = db.query(Paper).all()

        return [
            {
                "paper_id": p.paper_id,
                "user_id": p.user_id,
                "filename": p.filename,
                "uploaded_at": p.uploaded_at
            }
            for p in papers
        ]

    finally:
        db.close()



class ArxivIngestRequest(BaseModel):
    """
    Called after the user reviews ArXiv results from /chat and picks a paper.
    Downloads the PDF and runs it through the same pipeline as /upload-pdf.
    """
    login_id:  str
    arxiv_id:  str    # e.g. "2301.07041" — taken from arxiv_papers in /chat response

@app.post("/papers/ingest-arxiv")
async def ingest_arxiv(req: ArxivIngestRequest):
    """
    User-confirmed ingest of an ArXiv paper into the knowledge base.
 
    Flow:
      1. User calls /chat → gets arxiv_papers list
      2. User picks a paper they want → copies its arxiv_id
      3. User calls this endpoint → paper downloaded + ingested
      4. Future /chat queries now find this paper in local Qdrant
    """
    try:
        result = ingest_arxiv_paper(
            arxiv_id=req.arxiv_id,
            login_id=req.login_id,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
 
    return {
        "message": (
            f"'{result['title']}' added to your library. "
            "You can now ask questions about it in /chat."
        ),
        **result,
    }
