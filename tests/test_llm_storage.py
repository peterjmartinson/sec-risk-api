# Copyright (c) 2025 Peter Martinson, Distracted Fortune. All rights reserved.
# This software is proprietary and not licensed for use, modification, or distribution.

"""
Tests for SQLite persistence layer for LLM classification results.

This module tests storing and retrieving LLM responses along with their
embeddings for future lookups.
"""

import pytest
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Generator
import tempfile

from sigmak.llm_storage import (
    LLMStorage,
    LLMStorageRecord,
    StorageError,
)
from sigmak.risk_taxonomy import RiskCategory


@pytest.fixture
def temp_db() -> Generator[Path, None, None]:
    """Create a temporary database file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = Path(tmp.name)
    
    yield db_path
    
    # Cleanup
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def storage(temp_db: Path) -> LLMStorage:
    """Create an LLMStorage instance with a temporary database."""
    return LLMStorage(db_path=str(temp_db))


class TestLLMStorageRecord:
    """Tests for LLMStorageRecord dataclass."""
    
    def test_record_creation_valid(self) -> None:
        """Test creating a valid LLM storage record."""
        record = LLMStorageRecord(
            text="Supply chain disruptions",
            embedding=[0.1, 0.2, 0.3],
            category=RiskCategory.OPERATIONAL,
            confidence=0.95,
            evidence="Text mentions supply chain",
            rationale="Operational risk",
            model_version="gemini-2.5-flash",
            timestamp=datetime.now(),
            response_time_ms=150.5,
            input_tokens=100,
            output_tokens=50
        )
        
        assert record.text == "Supply chain disruptions"
        assert len(record.embedding) == 3
        assert record.category == RiskCategory.OPERATIONAL
        assert record.confidence == 0.95
    
    def test_record_confidence_validation(self) -> None:
        """Test that confidence must be in [0.0, 1.0] range."""
        # Valid bounds
        record = LLMStorageRecord(
            text="test",
            embedding=[0.1],
            category=RiskCategory.OPERATIONAL,
            confidence=0.0,
            evidence="test",
            rationale="test",
            model_version="gemini-2.5-flash",
            timestamp=datetime.now(),
            response_time_ms=100.0,
            input_tokens=10,
            output_tokens=10
        )
        assert record.confidence == 0.0
        
        # Invalid bounds should raise ValueError in __post_init__
        with pytest.raises(ValueError):
            LLMStorageRecord(
                text="test",
                embedding=[0.1],
                category=RiskCategory.OPERATIONAL,
                confidence=1.5,
                evidence="test",
                rationale="test",
                model_version="gemini-2.5-flash",
                timestamp=datetime.now(),
                response_time_ms=100.0,
                input_tokens=10,
                output_tokens=10
            )


class TestLLMStorageInit:
    """Tests for LLMStorage initialization."""
    
    def test_storage_initialization_creates_db(self, temp_db: Path) -> None:
        """Test that initialization creates the database file."""
        storage = LLMStorage(db_path=str(temp_db))
        assert temp_db.exists()
    
    def test_storage_initialization_creates_table(self, storage: LLMStorage) -> None:
        """Test that initialization creates the llm_classifications table."""
        # Query the database to verify table exists
        with sqlite3.connect(storage.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='llm_classifications'"
            )
            result = cursor.fetchone()
        
        assert result is not None
        assert result[0] == "llm_classifications"
    
    def test_storage_initialization_creates_index(self, storage: LLMStorage) -> None:
        """Test that initialization creates an index on text_hash."""
        with sqlite3.connect(storage.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_text_hash'"
            )
            result = cursor.fetchone()
        
        assert result is not None
        assert result[0] == "idx_text_hash"


class TestLLMStorageInsert:
    """Tests for inserting records into storage."""
    
    def test_insert_record_success(self, storage: LLMStorage) -> None:
        """Test successful insertion of a record."""
        record = LLMStorageRecord(
            text="Geopolitical tensions impact operations",
            embedding=[0.1, 0.2, 0.3, 0.4],
            category=RiskCategory.GEOPOLITICAL,
            confidence=0.92,
            evidence="Text discusses geopolitical tensions",
            rationale="Clear geopolitical risk indicators",
            model_version="gemini-2.5-flash",
            timestamp=datetime.now(),
            response_time_ms=200.0,
            input_tokens=120,
            output_tokens=60
        )
        
        record_id = storage.insert(record)
        assert isinstance(record_id, int)
        assert record_id > 0
    
    def test_insert_duplicate_text(self, storage: LLMStorage) -> None:
        """Test inserting duplicate text (should succeed but be retrievable)."""
        record1 = LLMStorageRecord(
            text="Duplicate risk text",
            embedding=[0.1, 0.2],
            category=RiskCategory.OPERATIONAL,
            confidence=0.9,
            evidence="test",
            rationale="test",
            model_version="gemini-2.5-flash",
            timestamp=datetime.now(),
            response_time_ms=100.0,
            input_tokens=10,
            output_tokens=10
        )
        
        record2 = LLMStorageRecord(
            text="Duplicate risk text",
            embedding=[0.3, 0.4],
            category=RiskCategory.FINANCIAL,
            confidence=0.85,
            evidence="test2",
            rationale="test2",
            model_version="gemini-2.5-flash",
            timestamp=datetime.now(),
            response_time_ms=150.0,
            input_tokens=15,
            output_tokens=15
        )
        
        id1 = storage.insert(record1)
        id2 = storage.insert(record2)
        
        # Both should be inserted successfully
        assert id1 != id2
        assert id1 > 0
        assert id2 > 0


class TestLLMStorageQuery:
    """Tests for querying records from storage."""
    
    def test_query_by_text_exact_match(self, storage: LLMStorage) -> None:
        """Test querying by exact text match."""
        text = "Cybersecurity threats to infrastructure"
        record = LLMStorageRecord(
            text=text,
            embedding=[0.5, 0.6, 0.7],
            category=RiskCategory.TECHNOLOGICAL,
            confidence=0.88,
            evidence="Cybersecurity mentioned",
            rationale="Tech risk",
            model_version="gemini-2.5-flash",
            timestamp=datetime.now(),
            response_time_ms=180.0,
            input_tokens=110,
            output_tokens=55
        )
        
        storage.insert(record)
        
        # Query by text
        results = storage.query_by_text(text)
        
        assert len(results) == 1
        assert results[0].text == text
        assert results[0].category == RiskCategory.TECHNOLOGICAL
        assert results[0].confidence == 0.88
    
    def test_query_by_text_no_match(self, storage: LLMStorage) -> None:
        """Test querying with text that doesn't exist."""
        results = storage.query_by_text("Nonexistent text")
        assert len(results) == 0
    
    def test_query_by_category(self, storage: LLMStorage) -> None:
        """Test querying by risk category."""
        # Insert records with different categories
        for i, category in enumerate([RiskCategory.OPERATIONAL, RiskCategory.FINANCIAL, RiskCategory.OPERATIONAL]):
            record = LLMStorageRecord(
                text=f"Risk text {i}",
                embedding=[float(i)] * 3,
                category=category,
                confidence=0.9,
                evidence=f"Evidence {i}",
                rationale=f"Rationale {i}",
                model_version="gemini-2.5-flash",
                timestamp=datetime.now(),
                response_time_ms=100.0,
                input_tokens=10,
                output_tokens=10
            )
            storage.insert(record)
        
        # Query for OPERATIONAL category
        results = storage.query_by_category(RiskCategory.OPERATIONAL)
        
        assert len(results) == 2
        assert all(r.category == RiskCategory.OPERATIONAL for r in results)
    
    def test_query_by_confidence_threshold(self, storage: LLMStorage) -> None:
        """Test querying records above a confidence threshold."""
        confidences = [0.75, 0.85, 0.95]
        
        for i, conf in enumerate(confidences):
            record = LLMStorageRecord(
                text=f"Risk {i}",
                embedding=[float(i)] * 2,
                category=RiskCategory.REGULATORY,
                confidence=conf,
                evidence=f"Evidence {i}",
                rationale=f"Rationale {i}",
                model_version="gemini-2.5-flash",
                timestamp=datetime.now(),
                response_time_ms=100.0,
                input_tokens=10,
                output_tokens=10
            )
            storage.insert(record)
        
        # Query for confidence >= 0.80
        results = storage.query_by_confidence(min_confidence=0.80)
        
        assert len(results) == 2
        assert all(r.confidence >= 0.80 for r in results)
    
    def test_query_recent_records(self, storage: LLMStorage) -> None:
        """Test querying most recent records."""
        # Insert multiple records
        for i in range(5):
            record = LLMStorageRecord(
                text=f"Recent risk {i}",
                embedding=[float(i)] * 2,
                category=RiskCategory.COMPETITIVE,
                confidence=0.9,
                evidence=f"Evidence {i}",
                rationale=f"Rationale {i}",
                model_version="gemini-2.5-flash",
                timestamp=datetime.now(),
                response_time_ms=100.0,
                input_tokens=10,
                output_tokens=10
            )
            storage.insert(record)
        
        # Query for 3 most recent
        results = storage.query_recent(limit=3)
        
        assert len(results) == 3
        # Verify they're in descending timestamp order
        for i in range(len(results) - 1):
            assert results[i].timestamp >= results[i + 1].timestamp


class TestLLMStorageStats:
    """Tests for storage statistics."""
    
    def test_get_total_count(self, storage: LLMStorage) -> None:
        """Test getting total record count."""
        assert storage.get_total_count() == 0
        
        # Insert records
        for i in range(3):
            record = LLMStorageRecord(
                text=f"Risk {i}",
                embedding=[float(i)] * 2,
                category=RiskCategory.OTHER,
                confidence=0.9,
                evidence=f"Evidence {i}",
                rationale=f"Rationale {i}",
                model_version="gemini-2.5-flash",
                timestamp=datetime.now(),
                response_time_ms=100.0,
                input_tokens=10,
                output_tokens=10
            )
            storage.insert(record)
        
        assert storage.get_total_count() == 3
    
    def test_get_category_counts(self, storage: LLMStorage) -> None:
        """Test getting count by category."""
        # Insert records with different categories
        categories = [
            RiskCategory.OPERATIONAL,
            RiskCategory.OPERATIONAL,
            RiskCategory.FINANCIAL,
            RiskCategory.TECHNOLOGICAL
        ]
        
        for i, category in enumerate(categories):
            record = LLMStorageRecord(
                text=f"Risk {i}",
                embedding=[float(i)] * 2,
                category=category,
                confidence=0.9,
                evidence=f"Evidence {i}",
                rationale=f"Rationale {i}",
                model_version="gemini-2.5-flash",
                timestamp=datetime.now(),
                response_time_ms=100.0,
                input_tokens=10,
                output_tokens=10
            )
            storage.insert(record)
        
        counts = storage.get_category_counts()
        
        assert counts[RiskCategory.OPERATIONAL.value] == 2
        assert counts[RiskCategory.FINANCIAL.value] == 1
        assert counts[RiskCategory.TECHNOLOGICAL.value] == 1


class TestLLMStorageErrorHandling:
    """Tests for error handling in storage operations."""
    
    def test_invalid_db_path(self) -> None:
        """Test initialization with invalid database path."""
        with pytest.raises(StorageError):
            LLMStorage(db_path="/nonexistent/directory/db.sqlite")
    
    def test_insert_with_invalid_record(self, storage: LLMStorage) -> None:
        """Test inserting an invalid record."""
        with pytest.raises(StorageError):
            # Missing required fields
            storage.insert(None)  # type: ignore
