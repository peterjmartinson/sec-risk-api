# Copyright (c) 2025 Peter Martinson, Distracted Fortune. All rights reserved.
# This software is proprietary and not licensed for use, modification, or distribution.

import pytest
import os
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from sec_risk_api.init_vector_db import initialize_chroma


@pytest.mark.slow
def test_persistence_directory_creation(tmp_path) -> None:
    """Confirm the DB creates a physical directory at the specified path."""
    test_db_path = str(tmp_path / "chroma_test")
    initialize_chroma(test_db_path)
    assert os.path.isdir(test_db_path), "Chroma should create a directory at the specified persist_path."


@pytest.mark.slow
def test_client_heartbeat(tmp_path) -> None:
    """Confirm the database can be 'pinged' via the heartbeat method."""
    test_db_path = str(tmp_path / "chroma_test")
    client, _ = initialize_chroma(test_db_path)
    heartbeat: int = client.heartbeat()

    assert isinstance(heartbeat, int)
    assert heartbeat > 0, "Heartbeat should return a positive integer timestamp."


@pytest.mark.slow
def test_collection_initialization(tmp_path) -> None:
    """Verify the 'sec_risk_factors' collection is created with correct properties."""
    test_db_path = str(tmp_path / "chroma_test")
    _, collection = initialize_chroma(test_db_path)

    assert collection.name == "sec_risk_factors"
    # Checking metadata to ensure our 'cosine' configuration stuck
    assert collection.metadata.get("hnsw:space") == "cosine"


@pytest.mark.slow
def test_type_integrity(tmp_path) -> None:
    """Ensure the objects returned match the expected API types for mypy compliance."""
    test_db_path = str(tmp_path / "chroma_test")
    client, collection = initialize_chroma(test_db_path)

    assert isinstance(client, ClientAPI)
    assert isinstance(collection, Collection)
