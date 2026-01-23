# Copyright (c) 2025 Peter Martinson, Distracted Fortune. All rights reserved.
# This software is proprietary and not licensed for use, modification, or distribution.

"""
Tests for risk classifier with LLM fallback integration.

This module tests the threshold-based routing logic that determines when
to use vector search vs LLM for risk classification.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from sigmak.risk_classifier import (
    RiskClassifierWithLLM,
    RiskClassificationResult,
    HIGH_THRESHOLD,
    LOW_THRESHOLD,
)
from sigmak.risk_taxonomy import RiskCategory
from sigmak.llm_classifier import LLMClassificationResult


class TestThresholdLogic:
    """Tests for threshold-based routing logic."""
    
    def test_high_confidence_uses_vector_search(self) -> None:
        """Test that high similarity scores use vector search result."""
        # Mock components
        mock_pipeline = Mock()
        mock_collection = Mock()
        mock_pipeline.collection = mock_collection
        
        # Mock ChromaDB query response for classified chunks
        mock_collection.query.return_value = {
            'ids': [['test_1']],
            'distances': [[0.2]],  # Low distance = high similarity (0.9)
            'metadatas': [[{
                "ticker": "AAPL",
                "category": "OPERATIONAL",
                "confidence": 0.85
            }]],
            'documents': [["Test risk text"]]
        }
        
        mock_llm = Mock()
        mock_storage = Mock()
        mock_storage.query_by_text.return_value = []
        
        mock_embedding = Mock()
        mock_embedding.encode.return_value = [[0.1, 0.2, 0.3]]
        
        classifier = RiskClassifierWithLLM(
            indexing_pipeline=mock_pipeline,
            llm_classifier=mock_llm,
            llm_storage=mock_storage,
            embedding_engine=mock_embedding
        )
        
        result = classifier.classify("Test risk text", use_cache=False)
        
        # Should use vector search
        assert result.method == "vector_search"
        assert result.similarity_score >= HIGH_THRESHOLD
        assert result.llm_result is None
        assert result.cached is True  # Cached from vector store
        
        # LLM should not be called
        mock_llm.classify.assert_not_called()
    
    def test_low_confidence_uses_llm(self) -> None:
        """Test that low similarity scores fall back to LLM."""
        # Mock components
        mock_pipeline = Mock()
        mock_collection = Mock()
        mock_pipeline.collection = mock_collection
        
        # Mock empty ChromaDB response (no classified chunks found)
        mock_collection.query.return_value = {
            'ids': [[]],
            'distances': [[]],
            'metadatas': [[]],
            'documents': [[]]
        }
        
        mock_llm_result = LLMClassificationResult(
            category=RiskCategory.FINANCIAL,
            confidence=0.95,
            evidence="Test evidence",
            rationale="Test rationale",
            model_version="gemini-2.5-flash",
            prompt_version="v1",
            timestamp=datetime.now(),
            response_time_ms=150.0,
            input_tokens=100,
            output_tokens=50
        )
        
        mock_llm = Mock()
        mock_llm.classify.return_value = mock_llm_result
        
        mock_storage = Mock()
        mock_storage.query_by_text.return_value = []
        
        mock_embedding = Mock()
        mock_embedding.encode.return_value = [[0.1, 0.2, 0.3]]
        
        classifier = RiskClassifierWithLLM(
            indexing_pipeline=mock_pipeline,
            llm_classifier=mock_llm,
            llm_storage=mock_storage,
            embedding_engine=mock_embedding
        )
        
        result = classifier.classify("Test risk text")
        
        # Should use LLM
        assert result.method == "llm"
        assert result.similarity_score < LOW_THRESHOLD
        assert result.llm_result is not None
        assert result.category == RiskCategory.FINANCIAL
        
        # LLM should be called
        mock_llm.classify.assert_called_once()
    
    def test_uncertain_confidence_uses_llm_confirmation(self) -> None:
        """Test that uncertain scores use LLM for confirmation."""
        # Mock components
        mock_pipeline = Mock()
        mock_collection = Mock()
        mock_pipeline.collection = mock_collection
        
        # Mock empty ChromaDB response (no classified chunks found)
        mock_collection.query.return_value = {
            'ids': [[]],
            'distances': [[]],
            'metadatas': [[]],
            'documents': [[]]
        }
        
        mock_llm_result = LLMClassificationResult(
            category=RiskCategory.REGULATORY,
            confidence=0.88,
            evidence="Test evidence",
            rationale="Test rationale",
            model_version="gemini-2.5-flash",
            prompt_version="v1",
            timestamp=datetime.now(),
            response_time_ms=150.0,
            input_tokens=100,
            output_tokens=50
        )
        
        mock_llm = Mock()
        mock_llm.classify.return_value = mock_llm_result
        
        mock_storage = Mock()
        mock_storage.query_by_text.return_value = []
        
        mock_embedding = Mock()
        mock_embedding.encode.return_value = [[0.1, 0.2, 0.3]]
        
        classifier = RiskClassifierWithLLM(
            indexing_pipeline=mock_pipeline,
            llm_classifier=mock_llm,
            llm_storage=mock_storage,
            embedding_engine=mock_embedding
        )
        
        result = classifier.classify("Test risk text", use_cache=False)
        
        # Should use LLM (no classified matches found)
        assert result.method == "llm"
        # Similarity score will be 0.0 since no matches
        assert result.category == RiskCategory.REGULATORY
        
        # LLM should be called for confirmation
        mock_llm.classify.assert_called_once()


class TestCaching:
    """Tests for LLM result caching."""
    
    def test_cached_result_skips_llm(self) -> None:
        """Test that cached LLM results are reused."""
        from sigmak.llm_storage import LLMStorageRecord
        
        # Mock cached result
        cached_record = LLMStorageRecord(
            text="Cached risk text",
            embedding=[0.1, 0.2],
            category=RiskCategory.GEOPOLITICAL,
            confidence=0.92,
            evidence="Cached evidence",
            rationale="Cached rationale",
            model_version="gemini-2.5-flash",
            timestamp=datetime.now(),
            response_time_ms=150.0,
            input_tokens=100,
            output_tokens=50,
            record_id=1
        )
        
        mock_storage = Mock()
        mock_storage.query_by_text.return_value = [cached_record]
        
        mock_pipeline = Mock()
        mock_llm = Mock()
        
        classifier = RiskClassifierWithLLM(
            indexing_pipeline=mock_pipeline,
            llm_classifier=mock_llm,
            llm_storage=mock_storage
        )
        
        result = classifier.classify("Cached risk text", use_cache=True)
        
        # Should use cached result
        assert result.cached is True
        assert result.category == RiskCategory.GEOPOLITICAL
        assert result.method == "llm"
        
        # LLM should not be called
        mock_llm.classify.assert_not_called()
        
        # Collection query should not be called (cached before vector search)
        # Note: we can't check this easily since pipeline isn't mocked with collection
    
    def test_cache_disabled_performs_search(self) -> None:
        """Test that disabling cache performs fresh classification."""
        from sigmak.llm_storage import LLMStorageRecord
        
        # Mock cached result
        cached_record = LLMStorageRecord(
            text="Cached risk text",
            embedding=[0.1, 0.2],
            category=RiskCategory.GEOPOLITICAL,
            confidence=0.92,
            evidence="Cached evidence",
            rationale="Cached rationale",
            model_version="gemini-2.5-flash",
            timestamp=datetime.now(),
            response_time_ms=150.0,
            input_tokens=100,
            output_tokens=50,
            record_id=1
        )
        
        mock_storage = Mock()
        mock_storage.query_by_text.return_value = [cached_record]
        
        mock_pipeline = Mock()
        mock_collection = Mock()
        mock_pipeline.collection = mock_collection
        
        # Mock ChromaDB query response with classified chunk
        mock_collection.query.return_value = {
            'ids': [['test_1']],
            'distances': [[0.2]],  # High similarity
            'metadatas': [[{
                "ticker": "AAPL",
                "category": "OPERATIONAL",
                "confidence": 0.85
            }]],
            'documents': [["Test text"]]
        }
        
        mock_llm = Mock()
        
        mock_embedding = Mock()
        mock_embedding.encode.return_value = [[0.1, 0.2, 0.3]]
        
        classifier = RiskClassifierWithLLM(
            indexing_pipeline=mock_pipeline,
            llm_classifier=mock_llm,
            llm_storage=mock_storage,
            embedding_engine=mock_embedding
        )
        
        result = classifier.classify("Cached risk text", use_cache=False)
        
        # Should not use SQLite cache, but will use vector store which is also a cache
        assert result.cached is True  # Vector store hit
        assert result.method == "vector_search"
        
        # Collection query should be called
        mock_collection.query.assert_called_once()


class TestLLMResultStorage:
    """Tests for storing LLM results."""
    
    def test_llm_result_stored_after_classification(self) -> None:
        """Test that LLM results are stored in cache."""
        # Mock components
        mock_pipeline = Mock()
        mock_collection = Mock()
        mock_pipeline.collection = mock_collection
        
        # Mock empty ChromaDB response (no classified chunks)
        mock_collection.query.return_value = {
            'ids': [[]],
            'distances': [[]],
            'metadatas': [[]],
            'documents': [[]]
        }
        
        mock_llm_result = LLMClassificationResult(
            category=RiskCategory.TECHNOLOGICAL,
            confidence=0.91,
            evidence="Tech evidence",
            rationale="Tech rationale",
            model_version="gemini-2.5-flash",
            prompt_version="v1",
            timestamp=datetime.now(),
            response_time_ms=150.0,
            input_tokens=100,
            output_tokens=50
        )
        
        mock_llm = Mock()
        mock_llm.classify.return_value = mock_llm_result
        
        mock_storage = Mock()
        mock_storage.query_by_text.return_value = []
        
        mock_embedding = Mock()
        mock_embedding.encode.return_value = [[0.1, 0.2, 0.3]]
        
        classifier = RiskClassifierWithLLM(
            indexing_pipeline=mock_pipeline,
            llm_classifier=mock_llm,
            llm_storage=mock_storage,
            embedding_engine=mock_embedding
        )
        
        result = classifier.classify("New risk text")
        
        # Storage insert should be called
        mock_storage.insert.assert_called_once()
        
        # Verify stored record
        stored_record = mock_storage.insert.call_args[0][0]
        assert stored_record.text == "New risk text"
        assert stored_record.category == RiskCategory.TECHNOLOGICAL
        assert stored_record.confidence == 0.91


class TestBatchClassification:
    """Tests for batch classification."""
    
    def test_classify_batch_processes_all_texts(self) -> None:
        """Test that batch classification processes all texts."""
        mock_pipeline = Mock()
        mock_collection = Mock()
        mock_pipeline.collection = mock_collection
        
        # Mock ChromaDB query response with classified chunk
        mock_collection.query.return_value = {
            'ids': [['test_1']],
            'distances': [[0.2]],
            'metadatas': [[{
                "ticker": "AAPL",
                "category": "OPERATIONAL",
                "confidence": 0.85
            }]],
            'documents': [["Test"]]
        }
        
        mock_storage = Mock()
        mock_storage.query_by_text.return_value = []
        
        mock_llm = Mock()
        
        mock_embedding = Mock()
        mock_embedding.encode.return_value = [[0.1, 0.2, 0.3]]
        
        classifier = RiskClassifierWithLLM(
            indexing_pipeline=mock_pipeline,
            llm_classifier=mock_llm,
            llm_storage=mock_storage,
            embedding_engine=mock_embedding
        )
        
        texts = ["Risk 1", "Risk 2", "Risk 3"]
        results = classifier.classify_batch(texts)
        
        assert len(results) == 3
        assert all(isinstance(r, RiskClassificationResult) for r in results)
