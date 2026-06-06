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
    """Accept a user message or conversation and return a RAG-grounded answer."""
    # Build conversation list for retrieval.query
    conversation: List[RAGMessage] = []
    if req.messages:
        for m in req.messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            conversation.append(RAGMessage(role=role, content=content))
    elif req.message:
        conversation.append(RAGMessage(role="user", content=req.message))
    else:
        raise HTTPException(status_code=400, detail="Either `message` or `messages` is required")

    # Call the RAG pipeline
    try:
        result = rag_query(conversation=conversation, login_id=req.login_id, paper_id=req.paper_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "answer": result.answer,
        "sources": result.sources,
        "query": result.query,
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