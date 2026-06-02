import os

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

from dotenv import load_dotenv

# Load environment variables from a .env file if present.
load_dotenv()

# Create a Qdrant client using configured service URL and API key.
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

# The name of the Qdrant collection to use for storing embeddings.
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

def create_collection():
    """Create the vector collection if it does not already exist."""
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]

    if COLLECTION_NAME not in collection_names:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=384,
                distance=Distance.COSINE,
            ),
        )