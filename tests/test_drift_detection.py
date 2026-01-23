# Copyright (c) 2025 Peter Martinson, Distracted Fortune. All rights reserved.
# This software is proprietary and not licensed for use, modification, or distribution.

"""
Unit tests for drift detection and periodic review system.

Tests cover:
- SQLite schema for classification examples with provenance
- Dual storage (SQLite + ChromaDB) integration
- Periodic review sampling and drift calculation
- Archive versioning for embedding model changes
"""

import json
import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import List
from unittest.mock import Mock, patch

import pytest
import numpy as np

from sigmak.drift_detection import (
    DriftDetectionSystem,
    DriftMetrics,
    DriftReviewJob,
    ClassificationSource,
)
from sigmak.llm_classifier import LLMClassificationResult
from sigmak.llm_storage import LLMStorageRecord
from sigmak.risk_taxonomy import RiskCategory


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_db() -> Path:
    """Create a temporary database file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
        db_path = Path(f.name)
    yield db_path
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def drift_system(temp_db: Path) -> DriftDetectionSystem:
    """Create a drift detection system with temp database."""
    return DriftDetectionSystem(db_path=str(temp_db))


@pytest.fixture
def sample_embedding() -> List[float]:
    """Generate a sample 384-dimensional embedding."""
    return np.random.rand(384).tolist()


@pytest.fixture
def sample_llm_result() -> LLMClassificationResult:
    """Create a sample LLM classification result."""
    return LLMClassificationResult(
        category=RiskCategory.OPERATIONAL,
        confidence=0.85,
        evidence="Supply chain disruptions may impact production.",
        rationale="Clear operational risk related to supply chain.",
        model_version="gemini-2.0-flash-lite",
        prompt_version="v1",
        timestamp=datetime.now(),
        response_time_ms=1250.0,
        input_tokens=150,
        output_tokens=50
    )


# ============================================================================
# Schema Tests
# ============================================================================


class TestDriftDetectionSchema:
    """Test SQLite schema for drift detection system."""
    
    def test_schema_has_source_field(self, drift_system: DriftDetectionSystem) -> None:
        """Schema must include 'source' field for tracking classification origin."""
        with sqlite3.connect(drift_system.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(risk_classifications)")
            columns = {row[1] for row in cursor.fetchall()}
        
        assert "source" in columns
    
    def test_schema_has_archive_version_field(self, drift_system: DriftDetectionSystem) -> None:
        """Schema must include 'archive_version' for embedding versioning."""
        with sqlite3.connect(drift_system.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(risk_classifications)")
            columns = {row[1] for row in cursor.fetchall()}
        
        assert "archive_version" in columns
    
    def test_schema_has_review_metadata(self, drift_system: DriftDetectionSystem) -> None:
        """Schema must track review history."""
        with sqlite3.connect(drift_system.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(risk_classifications)")
            columns = {row[1] for row in cursor.fetchall()}
        
        assert "last_reviewed_at" in columns
        assert "review_count" in columns
    
    def test_schema_has_chromadb_id(self, drift_system: DriftDetectionSystem) -> None:
        """Schema must store ChromaDB document ID for cross-reference."""
        with sqlite3.connect(drift_system.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(risk_classifications)")
            columns = {row[1] for row in cursor.fetchall()}
        
        assert "chroma_id" in columns


# ============================================================================
# Dual Storage Tests
# ============================================================================


class TestDualStorage:
    """Test SQLite + ChromaDB integration."""
    
    def test_insert_stores_in_both_databases(
        self,
        drift_system: DriftDetectionSystem,
        sample_embedding: List[float],
        sample_llm_result: LLMClassificationResult
    ) -> None:
        """Insert must store record in both SQLite and ChromaDB."""
        text = "Supply chain risks may affect operations."
        
        record_id, chroma_id = drift_system.insert_classification(
            text=text,
            embedding=sample_embedding,
            llm_result=sample_llm_result,
            source=ClassificationSource.LLM
        )
        
        # Verify SQLite storage
        assert record_id > 0
        assert chroma_id is not None
        
        # Verify record can be retrieved
        record = drift_system.get_record_by_id(record_id)
        assert record is not None
        assert record["text"] == text
        assert record["chroma_id"] == chroma_id
    
    def test_similarity_search_returns_sqlite_record(
        self,
        drift_system: DriftDetectionSystem,
        sample_embedding: List[float],
        sample_llm_result: LLMClassificationResult
    ) -> None:
        """Similarity search must return full SQLite record with provenance."""
        text = "Regulatory compliance risks."
        
        # Insert classification
        drift_system.insert_classification(
            text=text,
            embedding=sample_embedding,
            llm_result=sample_llm_result,
            source=ClassificationSource.LLM
        )
        
        # Search for similar text
        query_embedding = np.array(sample_embedding) + np.random.rand(384) * 0.01
        results = drift_system.similarity_search(
            query_embedding=query_embedding.tolist(),
            n_results=1
        )
        
        assert len(results) == 1
        assert results[0]["text"] == text
        assert "confidence" in results[0]
        assert "source" in results[0]


# ============================================================================
# Drift Detection Tests
# ============================================================================


class TestDriftDetection:
    """Test periodic drift detection system."""
    
    def test_sample_low_confidence_classifications(
        self,
        drift_system: DriftDetectionSystem,
        sample_embedding: List[float]
    ) -> None:
        """System must sample low-confidence classifications for review."""
        # Insert mix of confidence levels
        for confidence in [0.60, 0.70, 0.85, 0.90]:
            llm_result = LLMClassificationResult(
                category=RiskCategory.OPERATIONAL,
                confidence=confidence,
                evidence="Test evidence",
                rationale="Test rationale",
                model_version="gemini-2.0-flash-lite",
                prompt_version="v1",
                timestamp=datetime.now(),
                response_time_ms=1000.0,
                input_tokens=100,
                output_tokens=50
            )
            
            drift_system.insert_classification(
                text=f"Risk text with confidence {confidence}",
                embedding=sample_embedding,
                llm_result=llm_result,
                source=ClassificationSource.LLM
            )
        
        # Sample low confidence records
        records = drift_system.sample_for_review(
            min_confidence=0.0,
            max_confidence=0.75,
            sample_size=2
        )
        
        assert len(records) == 2
        assert all(r["confidence"] <= 0.75 for r in records)
    
    def test_sample_old_classifications(
        self,
        drift_system: DriftDetectionSystem,
        sample_embedding: List[float],
        sample_llm_result: LLMClassificationResult
    ) -> None:
        """System must sample older classifications for drift review."""
        # Insert old and new records (manipulate timestamps in DB)
        record_id, _ = drift_system.insert_classification(
            text="Old risk classification",
            embedding=sample_embedding,
            llm_result=sample_llm_result,
            source=ClassificationSource.LLM
        )
        
        # Manually update timestamp to be old
        with sqlite3.connect(drift_system.db_path) as conn:
            cursor = conn.cursor()
            old_date = (datetime.now() - timedelta(days=90)).isoformat()
            cursor.execute(
                "UPDATE risk_classifications SET timestamp = ? WHERE id = ?",
                (old_date, record_id)
            )
            conn.commit()
        
        # Sample old records
        cutoff_date = datetime.now() - timedelta(days=30)
        records = drift_system.sample_old_records(
            before_date=cutoff_date,
            sample_size=10
        )
        
        assert len(records) >= 1
        assert records[0]["id"] == record_id
    
    def test_compute_drift_metrics(
        self,
        drift_system: DriftDetectionSystem,
        sample_embedding: List[float],
        sample_llm_result: LLMClassificationResult
    ) -> None:
        """System must compute agreement rate between old and new classifications."""
        # Insert test records with LOW confidence so they get sampled
        texts = [
            "Operational risk text",
            "Financial risk text",
            "Regulatory risk text"
        ]
        
        # Create low-confidence result for sampling
        low_conf_result = LLMClassificationResult(
            category=RiskCategory.OPERATIONAL,
            confidence=0.70,  # Below 0.75 threshold
            evidence="Supply chain disruptions may impact production.",
            rationale="Clear operational risk related to supply chain.",
            model_version="gemini-2.0-flash-lite",
            prompt_version="v1",
            timestamp=datetime.now(),
            response_time_ms=1250.0,
            input_tokens=150,
            output_tokens=50
        )
        
        for text in texts:
            drift_system.insert_classification(
                text=text,
                embedding=sample_embedding,
                llm_result=low_conf_result,
                source=ClassificationSource.LLM
            )
        
        # Create mock LLM classifier that returns different category (simulate drift)
        mock_llm = Mock()
        mock_llm.classify.return_value = LLMClassificationResult(
            category=RiskCategory.FINANCIAL,  # Different from original OPERATIONAL
            confidence=0.80,
            evidence="New evidence",
            rationale="New rationale",
            model_version="gemini-2.0-flash-lite",
            prompt_version="v1",
            timestamp=datetime.now(),
            response_time_ms=1200.0,
            input_tokens=120,
            output_tokens=45
        )
        
        # Run drift detection with mocked LLM
        review_job = DriftReviewJob(
            drift_system=drift_system,
            llm_classifier=mock_llm
        )
        metrics = review_job.run_review(sample_size=3, low_conf_ratio=1.0)  # Sample all from low-conf
        
        assert isinstance(metrics, DriftMetrics)
        assert metrics.total_reviewed == 3
        assert metrics.disagreements == 3  # All should be different
        assert 0.0 <= metrics.agreement_rate <= 1.0
    
    def test_drift_threshold_triggers_alert(
        self,
        drift_system: DriftDetectionSystem
    ) -> None:
        """Agreement rate below threshold must trigger alert."""
        metrics = DriftMetrics(
            total_reviewed=100,
            agreements=70,  # 70% agreement
            disagreements=30,
            agreement_rate=0.70,
            avg_confidence_change=-0.15,
            timestamp=datetime.now()
        )
        
        # 70% is below 75% CRITICAL threshold
        assert metrics.requires_manual_review(critical_threshold=0.75)
    
    def test_drift_metrics_logged_to_database(
        self,
        drift_system: DriftDetectionSystem
    ) -> None:
        """Drift metrics must be persisted for audit trail."""
        metrics = DriftMetrics(
            total_reviewed=50,
            agreements=45,
            disagreements=5,
            agreement_rate=0.90,
            avg_confidence_change=0.02,
            timestamp=datetime.now()
        )
        
        drift_system.log_drift_metrics(metrics)
        
        # Verify metrics are stored
        retrieved = drift_system.get_recent_drift_metrics(limit=1)
        assert len(retrieved) == 1
        assert retrieved[0]["agreement_rate"] == 0.90


# ============================================================================
# Archive and Versioning Tests
# ============================================================================


class TestArchiveVersioning:
    """Test embedding archive and versioning system."""
    
    def test_archive_old_embeddings_before_reembedding(
        self,
        drift_system: DriftDetectionSystem,
        sample_embedding: List[float],
        sample_llm_result: LLMClassificationResult
    ) -> None:
        """System must archive old embeddings when model changes."""
        # Insert original classification
        record_id, _ = drift_system.insert_classification(
            text="Test risk text",
            embedding=sample_embedding,
            llm_result=sample_llm_result,
            source=ClassificationSource.LLM
        )
        
        # Archive before re-embedding
        new_embedding = (np.array(sample_embedding) * 1.1).tolist()
        drift_system.archive_and_update_embedding(
            record_id=record_id,
            new_embedding=new_embedding,
            new_model_version="all-MiniLM-L6-v2-v2"
        )
        
        # Verify old embedding is archived
        archives = drift_system.get_embedding_archives(record_id)
        assert len(archives) >= 1
        assert archives[0]["embedding"] == sample_embedding
    
    def test_duplicate_detection(
        self,
        drift_system: DriftDetectionSystem,
        sample_embedding: List[float],
        sample_llm_result: LLMClassificationResult
    ) -> None:
        """System must detect duplicate classifications by text hash."""
        text = "Duplicate risk text"
        
        # Insert first time
        id1, _ = drift_system.insert_classification(
            text=text,
            embedding=sample_embedding,
            llm_result=sample_llm_result,
            source=ClassificationSource.LLM
        )
        
        # Try to insert duplicate
        id2, _ = drift_system.insert_classification(
            text=text,
            embedding=sample_embedding,
            llm_result=sample_llm_result,
            source=ClassificationSource.LLM,
            allow_duplicates=False
        )
        
        # Should return existing record ID
        assert id1 == id2
    
    def test_get_model_change_statistics(
        self,
        drift_system: DriftDetectionSystem
    ) -> None:
        """System must report statistics on embedding model changes."""
        stats = drift_system.get_model_statistics()
        
        assert "total_records" in stats
        assert "current_model_version" in stats
        assert "archived_versions" in stats
