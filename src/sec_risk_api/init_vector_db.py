import chromadb
from chromadb.config import Settings
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
import os
from typing import Tuple

def initialize_chroma(persist_path: str = "./chroma_db") -> Tuple[ClientAPI, Collection]:
    """
    Initializes a persistent ChromaDB client and creates the risk_factors
    collection.

    Args:
        persist_path: The filesystem path where the database will be stored.

    Returns:
        A tuple containing the Chroma Client and the specific Collection
        instance.
    """
    # Ensure the directory exists for WSL persistence
    if not os.path.exists(persist_path):
        os.makedirs(persist_path)

    # Initialize Persistent Client
    client: ClientAPI = chromadb.PersistentClient(path=persist_path)

    # Create or Get Collection
    # metadata "hnsw:space" defines the distance metric for semantic search
    collection: Collection = client.get_or_create_collection(
        name="sec_risk_factors",
        metadata={"hnsw:space": "cosine"}
    )

    return client, collection

if __name__ == "__main__":
    client, col = initialize_chroma()
    print(f"Chroma DB initialized. Heartbeat: {client.heartbeat()}")
