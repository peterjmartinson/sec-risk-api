import unittest
import os
import shutil
import tempfile
from typing import Tuple
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from sec_risk_api.init_vector_db import initialize_chroma

class TestChromaInfrastructure(unittest.TestCase):
    """
    Test suite for Chroma DB infrastructure (Subissue 1.1).
    Each test verifies exactly one behavior of the database initialization.
    """
    
    def setUp(self) -> None:
        """Create a clean temporary directory for each test."""
        # Use tempfile to create an isolated, clean directory
        # This avoids permission issues and cleanup race conditions
        self.test_db_dir = tempfile.TemporaryDirectory()
        self.test_db_path: str = self.test_db_dir.name

    def tearDown(self) -> None:
        """Clean up the temporary directory after each test."""
        # TemporaryDirectory cleanup is automatic and thread-safe
        self.test_db_dir.cleanup()

    def test_persistence_directory_creation(self) -> None:
        """Confirm the DB creates a physical directory at the specified path."""
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