import unittest
import os
import shutil
from typing import Tuple
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from sec_risk_api.init_vector_db import initialize_chroma

class TestChromaInfrastructure(unittest.TestCase):
    def setUp(self) -> None:
        """Create a clean test directory before each test."""
        self.test_db_path: str = "./test_chroma_db"
        if os.path.exists(self.test_db_path):
            shutil.rmtree(self.test_db_path)

    def tearDown(self) -> None:
        """Clean up the test directory after each test."""
        if os.path.exists(self.test_db_path):
            shutil.rmtree(self.test_db_path)

    def test_persistence_directory_creation(self) -> None:
        """Confirm the DB creates a physical directory on the WSL disk."""
        initialize_chroma(self.test_db_path)
        self.assertTrue(
            os.path.isdir(self.test_db_path),
            "Chroma should create a directory at the specified persist_path."
        )

    def test_client_heartbeat(self) -> None:
        """Confirm the database can be 'pinged' via the heartbeat method."""
        client, _ = initialize_chroma(self.test_db_path)
        heartbeat: int = client.heartbeat()

        self.assertIsInstance(heartbeat, int)
        self.assertGreater(heartbeat, 0, "Heartbeat should return a positive integer timestamp.")

    def test_collection_initialization(self) -> None:
        """Verify the 'sec_risk_factors' collection is created with correct properties."""
        _, collection = initialize_chroma(self.test_db_path)

        self.assertEqual(collection.name, "sec_risk_factors")
        # Checking metadata to ensure our 'cosine' configuration stuck
        self.assertEqual(collection.metadata.get("hnsw:space"), "cosine")

    def test_type_integrity(self) -> None:
        """Ensure the objects returned match the expected API types for mypy compliance."""
        client, collection = initialize_chroma(self.test_db_path)

        self.assertIsInstance(client, ClientAPI)
        self.assertIsInstance(collection, Collection)

if __name__ == "__main__":
    unittest.main()
