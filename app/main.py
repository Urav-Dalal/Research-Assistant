import os

from fastapi import FastAPI, UploadFile, File
from .upload_pipeline import process_pdf
from .qdrant import create_collection

# FastAPI app entrypoint for the `app` package.
# When using `uvicorn app.main:app --reload`, this module is loaded as part
# of the `app` package, so relative imports are required.
app = FastAPI()

# Directory where uploaded PDF files are temporarily written before processing.
UPLOAD_DIR = "temp"

os.makedirs(UPLOAD_DIR, exist_ok=True)

# Ensure the Qdrant collection exists before accepting uploads.
create_collection()

@app.get("/")
def health_check():
    """Simple health check endpoint."""
    return {"status": "running"}

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Accept an uploaded PDF, save it locally, process it, and return metadata."""

    # Save the uploaded file to the local temp directory.
    file_path = f"{UPLOAD_DIR}/{file.filename}"

    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Process the PDF content and store embeddings in Qdrant.
    chunks_stored = process_pdf(file_path, file.filename)

    return {
        "message": "PDF processed successfully",
        "filename": file.filename,
        "chunks_stored": chunks_stored,
    }