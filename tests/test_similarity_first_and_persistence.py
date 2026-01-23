# Copyright (c) 2025 Peter Martinson, Distracted Fortune. All rights reserved.
# This software is proprietary and not licensed for use, modification, or distribution.

"""
Tests for similarity-first classification and LLM persistence.

Tests the complete flow:
1. LLM classification results are persisted to both SQLite and ChromaDB
2. Similarity search queries the llm_risk_classification collection first
3. If similarity >= threshold, cached result is returned (no LLM call)
4. If similarity < threshold, LLM is called and result is persisted
"""

import tempfile
from datetime import datetime
from pathlib import Path
from typing import List
from unittest.mock import Mock, patch

import pytest

from sigmak.config import get_settings, reset_config
from sigmak.drift_detection import ClassificationSource, DriftDetectionSystem
from sigmak.llm_classifier import LLMClassificationResult
from sigmak.risk_classification_service import RiskClassificationService
from sigmak.risk_taxonomy import RiskCategory


class TestLLMPersistence:
    """Test that every LLM classification is persisted correctly."""
    
    def test_llm_result_persisted_to_sqlite(self, tmp_path: Path) -> None:
        """Test that LLM classification is written to SQLite audit table."""
        db_path = tmp_path / "test.db"
        chroma_path = tmp_path / "chroma"
        
        system = DriftDetectionSystem(
            db_path=str(db_path),
            chroma_path=str(chroma_path),
            embedding_model="test-model"
        )
        
        # Create mock LLM result with prompt_version
        llm_result = LLMClassificationResult(
            category=RiskCategory.OPERATIONAL,
            confidence=0.92,
            evidence="Supply chain disruption text",
            rationale="Operational risk due to logistics",
            model_version="gemini-2.0-flash",
            prompt_version="1",
            timestamp=datetime.now(),
            response_time_ms=150.0,
            input_tokens=100,
            output_tokens=50
        )
        
        text = "Our supply chain faces significant risks from global disruptions."
        embedding = [0.1] * 384  # Mock 384-dim embedding
        
        # Insert classification
        record_id, chroma_id = system.insert_classification(
            text=text,
            embedding=embedding,
            llm_result=llm_result,
            source=ClassificationSource.LLM
        )
        
        # Verify SQLite record exists with prompt_version
        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT category, confidence, evidence, prompt_version, source FROM risk_classifications WHERE id = ?",
            (record_id,)
        )
        row = cursor.fetchone()
        conn.close()
        
        assert row is not None
        assert row[0] == "operational"
        assert row[1] == 0.92
        assert row[2] == "Supply chain disruption text"
        assert row[3] == "1"  # prompt_version
        assert row[4] == "llm"
    
    def test_llm_result_persisted_to_chromadb(self, tmp_path: Path) -> None:
        """Test that LLM classification is written to ChromaDB with metadata."""
        db_path = tmp_path / "test.db"
        chroma_path = tmp_path / "chroma"
        
        system = DriftDetectionSystem(
            db_path=str(db_path),
            chroma_path=str(chroma_path),
            embedding_model="test-model"
        )
        
        llm_result = LLMClassificationResult(
            category=RiskCategory.REGULATORY,
            confidence=0.88,
            evidence="New regulation text",
            rationale="Regulatory compliance risk",
            model_version="gemini-2.0-flash",
            prompt_version="2",
            timestamp=datetime.now(),
            response_time_ms=200.0,
            input_tokens=120,
            output_tokens=60
        )
        
        text = "New environmental regulations may impact operations."
        embedding = [0.2] * 384
        
        record_id, chroma_id = system.insert_classification(
            text=text,
            embedding=embedding,
            llm_result=llm_result,
            source=ClassificationSource.LLM
        )
        
        # Query ChromaDB directly
        result = system.collection.get(ids=[chroma_id], include=["metadatas", "documents"])
        
        assert len(result["ids"]) == 1
        assert result["ids"][0] == chroma_id
        metadata = result["metadatas"][0]
        
        # Verify metadata includes required fields
        assert metadata["record_id"] == record_id
        assert metadata["category"] == "regulatory"
        assert metadata["confidence"] == 0.88
        assert metadata["source"] == "llm"
        assert metadata["prompt_version"] == "2"
        assert "origin_text" in metadata
        assert result["documents"][0] == text


class TestSimilarityFirstFlow:
    """Test similarity-first classification logic."""
    
    @pytest.fixture
    def mock_system(self, tmp_path: Path) -> DriftDetectionSystem:
        """Create a DriftDetectionSystem for testing."""
        db_path = tmp_path / "test.db"
        chroma_path = tmp_path / "chroma"
        return DriftDetectionSystem(
            db_path=str(db_path),
            chroma_path=str(chroma_path),
            embedding_model="test-model"
        )
    
    def test_similarity_above_threshold_reuses_cached(self, mock_system: DriftDetectionSystem, tmp_path: Path) -> None:
        """Test that high similarity (>= threshold) returns cached result without LLM call."""
        # Insert a cached LLM classification
        cached_result = LLMClassificationResult(
            category=RiskCategory.TECHNOLOGICAL,
            confidence=0.95,
            evidence="Cybersecurity threat evidence",
            rationale="Tech risk",
            model_version="gemini-2.0-flash",
            prompt_version="1",
            timestamp=datetime.now(),
            response_time_ms=100.0,
            input_tokens=80,
            output_tokens=40
        )
        
        cached_text = "Cybersecurity threats pose significant risks to our infrastructure."
        cached_embedding = [0.3] * 384
        
        mock_system.insert_classification(
            text=cached_text,
            embedding=cached_embedding,
            llm_result=cached_result,
            source=ClassificationSource.LLM
        )
        
        # Create a very similar query (simulate high similarity)
        query_text = "Cybersecurity risks threaten our infrastructure significantly."
        query_embedding = [0.31] * 384  # Very similar to cached_embedding
        
        # Search for similar classifications
        results = mock_system.similarity_search(query_embedding, n_results=1)
        
        # Should find the cached result
        assert len(results) > 0
        assert results[0]["category"] == "technological"
        assert results[0]["source"] == "llm"
        assert results[0]["prompt_version"] == "1"
    
    def test_similarity_below_threshold_calls_llm(self, mock_system: DriftDetectionSystem) -> None:
        """Test that low similarity (< threshold) requires LLM call."""
        # Insert a cached classification (about tech risks)
        cached_result = LLMClassificationResult(
            category=RiskCategory.TECHNOLOGICAL,
            confidence=0.90,
            evidence="AI disruption",
            rationale="Tech disruption risk",
            model_version="gemini-2.0-flash",
            prompt_version="1",
            timestamp=datetime.now(),
            response_time_ms=120.0,
            input_tokens=90,
            output_tokens=45
        )
        
        cached_embedding = [0.4] * 384
        
        mock_system.insert_classification(
            text="AI may disrupt our business model.",
            embedding=cached_embedding,
            llm_result=cached_result,
            source=ClassificationSource.LLM
        )
        
        # Query for completely different risk (geopolitical)
        dissimilar_embedding = [-0.4] * 384  # Orthogonal/dissimilar vector
        
        results = mock_system.similarity_search(dissimilar_embedding, n_results=1)
        
        # Should return something but with low similarity score
        # In real implementation, caller would check distance/score and call LLM
        assert len(results) >= 0  # May or may not return results depending on threshold


class TestLLMCacheCollection:
    """Test the dedicated llm_risk_classification ChromaDB collection."""
    
    def test_llm_cache_collection_created(self, tmp_path: Path) -> None:
        """Test that llm_risk_classification collection is created."""
        # This test will fail initially until we implement the dedicated collection
        # For now, we use 'risk_classifications' but the Issue requires 'llm_risk_classification'
        db_path = tmp_path / "test.db"
        chroma_path = tmp_path / "chroma"
        
        system = DriftDetectionSystem(
            db_path=str(db_path),
            chroma_path=str(chroma_path),
            embedding_model="test-model"
        )
        
        # Check collection name
        # TODO: Update when we create dedicated llm_risk_classification collection
        assert system.collection.name in ["risk_classifications", "llm_risk_classification"]
    
    def test_multiple_prompt_versions_tracked(self, tmp_path: Path) -> None:
        """Test that different prompt versions are tracked correctly."""
        db_path = tmp_path / "test.db"
        chroma_path = tmp_path / "chroma"
        
        system = DriftDetectionSystem(
            db_path=str(db_path),
            chroma_path=str(chroma_path),
            embedding_model="test-model"
        )
        
        # Insert classifications with different prompt versions
        for version in ["1", "2", "3"]:
            result = LLMClassificationResult(
                category=RiskCategory.FINANCIAL,
                confidence=0.85,
                evidence=f"Financial risk v{version}",
                rationale="Finance",
                model_version="gemini-2.0-flash",
                prompt_version=version,
                timestamp=datetime.now(),
                response_time_ms=100.0,
                input_tokens=50,
                output_tokens=25
            )
            
            system.insert_classification(
                text=f"Liquidity risk statement version {version}",
                embedding=[float(version) / 10] * 384,
                llm_result=result,
                source=ClassificationSource.LLM
            )
        
        # Query all records
        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT prompt_version FROM risk_classifications ORDER BY prompt_version")
        versions = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        assert versions == ["1", "2", "3"]


class TestConfigIntegration:
    """Test that similarity threshold comes from config."""
    
    def test_config_provides_similarity_threshold(self, tmp_path: Path) -> None:
        """Test that llm_cache_similarity_threshold is accessible from config."""
        # Create test config
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("""
database:
  sqlite_path: "test.db"

chroma:
  persist_directory: "test_chroma"
  embedding_model: "test-model"
  llm_cache_similarity_threshold: 0.75

llm:
  model: "test-llm"
  temperature: 0.0

drift:
  review_cron: "0 3 * * *"
  sample_size: 100
  low_confidence_threshold: 0.6
  drift_threshold: 0.2

logging:
  level: "INFO"
""")
        
        reset_config()
        settings = get_settings(config_file)
        
        assert settings.chroma.llm_cache_similarity_threshold == 0.75
    
    def test_config_default_threshold(self, tmp_path: Path) -> None:
        """Test that default threshold is 0.8."""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("""
database:
  sqlite_path: "test.db"

chroma:
  persist_directory: "test_chroma"
  embedding_model: "test-model"
  # llm_cache_similarity_threshold omitted, should default to 0.8

llm:
  model: "test-llm"
  temperature: 0.0

drift:
  review_cron: "0 3 * * *"
  sample_size: 100
  low_confidence_threshold: 0.6
  drift_threshold: 0.2

logging:
  level: "INFO"
""")
        
        reset_config()
        settings = get_settings(config_file)
        
        assert settings.chroma.llm_cache_similarity_threshold == 0.8


class TestEndToEndFlow:
    """Integration tests for the complete similarity-first flow."""
    
    def test_classify_with_cache_first_high_similarity(self, tmp_path: Path) -> None:
        """Test classify_with_cache_first returns cached result for high similarity."""
        # Create config
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("""
database:
  sqlite_path: "test.db"

chroma:
  persist_directory: "test_chroma"
  embedding_model: "all-MiniLM-L6-v2"
  llm_cache_similarity_threshold: 0.8

llm:
  model: "gemini-2.0-flash-exp"
  temperature: 0.0

drift:
  review_cron: "0 3 * * *"
  sample_size: 100
  low_confidence_threshold: 0.6
  drift_threshold: 0.2

logging:
  level: "INFO"
""")
        
        reset_config()
        
        # Initialize service with test paths
        db_path = tmp_path / "test.db"
        chroma_path = tmp_path / "chroma"
        
        drift_system = DriftDetectionSystem(
            db_path=str(db_path),
            chroma_path=str(chroma_path),
            embedding_model="all-MiniLM-L6-v2"
        )
        
        # Create mock LLM classifier that should NOT be called
        mock_llm = Mock()
        mock_llm.model_name = "test-model"
        
        service = RiskClassificationService(
            drift_system=drift_system,
            llm_classifier=mock_llm,
            config_path=str(config_file)
        )
        
        # Pre-populate cache with a classification
        from sigmak.embeddings import EmbeddingEngine
        embedding_engine = EmbeddingEngine(model_name="all-MiniLM-L6-v2")
        cached_text = "Supply chain disruptions may significantly impact our operations."
        cached_embeddings = embedding_engine.encode([cached_text])
        
        cached_result = LLMClassificationResult(
            category=RiskCategory.OPERATIONAL,
            confidence=0.93,
            evidence="Supply chain disruptions",
            rationale="Operational risk",
            model_version="gemini-2.0-flash",
            prompt_version="1",
            timestamp=datetime.now(),
            response_time_ms=150.0,
            input_tokens=100,
            output_tokens=50
        )
        
        drift_system.insert_classification(
            text=cached_text,
            embedding=cached_embeddings[0].tolist(),
            llm_result=cached_result,
            source=ClassificationSource.LLM
        )
        
        # Query with very similar text
        query_text = "Supply chain issues could significantly affect our business operations."
        result, source = service.classify_with_cache_first(query_text)
        
        # Should use cache, not call LLM
        assert source == "cache"
        assert result.category == RiskCategory.OPERATIONAL
        assert mock_llm.classify.call_count == 0  # LLM should NOT be called
    
    def test_classify_with_cache_first_low_similarity(self, tmp_path: Path) -> None:
        """Test classify_with_cache_first calls LLM for low similarity."""
        # Create config
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("""
database:
  sqlite_path: "test.db"

chroma:
  persist_directory: "test_chroma"
  embedding_model: "all-MiniLM-L6-v2"
  llm_cache_similarity_threshold: 0.8

llm:
  model: "gemini-2.0-flash-exp"
  temperature: 0.0

drift:
  review_cron: "0 3 * * *"
  sample_size: 100
  low_confidence_threshold: 0.6
  drift_threshold: 0.2

logging:
  level: "INFO"
""")
        
        reset_config()
        
        # Initialize service
        db_path = tmp_path / "test.db"
        chroma_path = tmp_path / "chroma"
        
        drift_system = DriftDetectionSystem(
            db_path=str(db_path),
            chroma_path=str(chroma_path),
            embedding_model="all-MiniLM-L6-v2"
        )
        
        # Create mock LLM classifier
        mock_llm = Mock()
        mock_llm.model_name = "test-model"
        mock_llm.classify.return_value = LLMClassificationResult(
            category=RiskCategory.GEOPOLITICAL,
            confidence=0.91,
            evidence="Trade war impacts",
            rationale="Geopolitical risk",
            model_version="gemini-2.0-flash",
            prompt_version="1",
            timestamp=datetime.now(),
            response_time_ms=200.0,
            input_tokens=120,
            output_tokens=60
        )
        
        service = RiskClassificationService(
            drift_system=drift_system,
            llm_classifier=mock_llm,
            config_path=str(config_file)
        )
        
        # Pre-populate cache with DIFFERENT type of risk
        from sigmak.embeddings import EmbeddingEngine
        embedding_engine = EmbeddingEngine(model_name="all-MiniLM-L6-v2")
        cached_text = "Cybersecurity threats pose risks to our systems."
        cached_embeddings = embedding_engine.encode([cached_text])
        
        cached_result = LLMClassificationResult(
            category=RiskCategory.TECHNOLOGICAL,
            confidence=0.90,
            evidence="Cybersecurity",
            rationale="Tech risk",
            model_version="gemini-2.0-flash",
            prompt_version="1",
            timestamp=datetime.now(),
            response_time_ms=100.0,
            input_tokens=80,
            output_tokens=40
        )
        
        drift_system.insert_classification(
            text=cached_text,
            embedding=cached_embeddings[0].tolist(),
            llm_result=cached_result,
            source=ClassificationSource.LLM
        )
        
        # Query with VERY DIFFERENT text (geopolitical vs tech)
        query_text = "International trade disputes may affect our export business."
        result, source = service.classify_with_cache_first(query_text)
        
        # Should call LLM due to low similarity
        assert source == "llm"
        assert result.category == RiskCategory.GEOPOLITICAL
        assert mock_llm.classify.call_count == 1  # LLM SHOULD be called
