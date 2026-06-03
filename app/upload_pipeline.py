import datetime

import fitz
import uuid

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

from qdrant_client.models import PointStruct

from .qdrant import client, COLLECTION_NAME
from datetime import datetime 

# PDF processing pipeline:
# 1. extract raw text from the uploaded PDF
# 2. split text into overlapping chunks
# 3. encode each chunk into an embedding vector
# 4. upsert embeddings into Qdrant
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# Text splitter settings define chunk size and overlap for retrieval quality.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
)


def extract_text_from_pdf(pdf_path):
    """Extract plain text from every page of a PDF file."""
    doc = fitz.open(pdf_path)

    text = ""
    for page in doc:
        text += page.get_text()

    return text


def process_pdf(pdf_path, filename, paper_id, login_id):
    """Transform a PDF file into stored Qdrant vectors."""

    # Read all text from the PDF.
    raw_text = extract_text_from_pdf(pdf_path)

    # Split the raw text into chunks for embedding.
    chunks = splitter.split_text(raw_text)

    points = []
    for idx, chunk in enumerate(chunks):
        # Encode each chunk into a numeric vector.
        embedding = model.encode(chunk).tolist()

        # Build the Qdrant point with metadata that includes source and chunk index.
        child_chunk_id = str(uuid.uuid4())

        point = PointStruct(
            id=child_chunk_id,
            vector=embedding,
            payload={
                "login_id": login_id,
                "paper_id": paper_id,

                "source": filename,

                "chunk_type": "child",

                "parent_chunk_id": None,
                "child_chunk_id": child_chunk_id,

                "section_title": None,

                "chunk_index": idx,

                "text": chunk,
                "upload_timestamp": datetime.utcnow().isoformat()
            },
        )
        points.append(point)

    # Upload all chunk vectors to the configured Qdrant collection.
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )

    return len(chunks)