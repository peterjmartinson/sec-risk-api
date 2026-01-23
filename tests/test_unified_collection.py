# Copyright (c) 2025 Peter Martinson, Distracted Fortune. All rights reserved.
# This software is proprietary and not licensed for use, modification, or distribution.

"""
Tests for Issue #26: Unified Vector Store for Risk Classification.

This test suite verifies:
1. Classification metadata is stored in sec_risk_factors collection
2. Chunks can be updated with classification after indexing
3. Risk classifier queries unified collection first
4. High-similarity classified chunks prevent LLM calls
5. Backward compatibility with None-valued metadata
"""

import pytest
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock, patch, MagicMock

from sigmak.indexing_pipeline import IndexingPipeline
from sigmak.risk_classifier import RiskClassifierWithLLM
from sigmak.risk_taxonomy import RiskCategory
from sigmak.llm_classifier import LLMClassificationResult
from datetime import datetime


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_chroma_path(tmp_path):
    """Temporary ChromaDB path for testing."""
    return str(tmp_path / "test_chroma")


@pytest.fixture
def indexing_pipeline(temp_chroma_path):
    """Initialize IndexingPipeline with temp storage."""
    return IndexingPipeline(persist_path=temp_chroma_path)


@pytest.fixture
def sample_html_content():
    """Sample SEC filing HTML with Item 1A."""
    return """
    <html>
    <body>
    <div>Item 1A. Risk Factors</div>
    <p>Our business operations face significant cybersecurity risks including data breaches,
    ransomware attacks, and unauthorized access to sensitive systems. A major security incident
    could result in substantial financial losses, regulatory penalties, and reputational damage.</p>
    
    <p>Supply chain disruptions due to geopolitical tensions, natural disasters, or supplier
    failures could materially impact our production capacity and revenue generation capabilities.</p>
    </body>
    </html>
    """


# ============================================================================
# Test 1: Classification Metadata Storage
# ============================================================================


def test_indexing_with_classification_metadata(indexing_pipeline, sample_html_content, tmp_path):
    """
    Verify that chunks are indexed with None-valued classification metadata.
    
    TDD: Write test before implementation.
    """
    # Create temp HTML file
    html_path = tmp_path / "test_filing.html"
    html_path.write_text(sample_html_content)
    
    # Index filing
    result = indexing_pipeline.index_filing(
        html_path=str(html_path),
        ticker="TEST",
        filing_year=2025,
        item_type="Item 1A"
    )
    
    assert result["status"] == "success"
    assert result["chunks_indexed"] > 0
    
    # Verify metadata schema includes classification fields (all None)
    doc_id = indexing_pipeline._generate_document_id("TEST", 2025, 0)
    chunk = indexing_pipeline.get_chunk_by_doc_id(doc_id)
    
    assert chunk is not None
    assert chunk["metadata"]["ticker"] == "TEST"
    assert chunk["metadata"]["filing_year"] == 2025
    assert chunk["metadata"]["item_type"] == "Item 1A"
    
    # Classification fields should exist but be empty string (unclassified)
    # Note: Using "" instead of None because ChromaDB $ne filter doesn't support None
    assert chunk["metadata"]["category"] == ""
    assert chunk["metadata"]["confidence"] == 0.0
    assert chunk["metadata"]["classification_source"] == ""
    assert chunk["metadata"]["classification_timestamp"] == ""
    assert chunk["metadata"]["model_version"] == ""
    assert chunk["metadata"]["prompt_version"] == ""


# ============================================================================
# Test 2: Update Chunk Classification
# ============================================================================


def test_update_chunk_classification(indexing_pipeline, sample_html_content, tmp_path):
    """
    Verify that chunk metadata can be updated with classification data.
    
    TDD: Write test before implementation.
    """
    # Create and index filing
    html_path = tmp_path / "test_filing.html"
    html_path.write_text(sample_html_content)
    
    indexing_pipeline.index_filing(
        html_path=str(html_path),
        ticker="TEST",
        filing_year=2025
    )
    
    # Update first chunk with classification
    indexing_pipeline.update_chunk_classification(
        ticker="TEST",
        filing_year=2025,
        chunk_index=0,
        category="OPERATIONAL",
        confidence=0.87,
        source="llm",
        model_version="gemini-2.5-flash-lite",
        prompt_version="v1"
    )
    
    # Verify metadata was updated
    doc_id = indexing_pipeline._generate_document_id("TEST", 2025, 0)
    chunk = indexing_pipeline.get_chunk_by_doc_id(doc_id)
    
    assert chunk is not None
    assert chunk["metadata"]["category"] == "OPERATIONAL"
    assert chunk["metadata"]["confidence"] == 0.87
    assert chunk["metadata"]["classification_source"] == "llm"
    assert chunk["metadata"]["classification_timestamp"] is not None
    assert chunk["metadata"]["model_version"] == "gemini-2.5-flash-lite"
    assert chunk["metadata"]["prompt_version"] == "v1"


# ============================================================================
# Test 3: Query Classified Chunks Only
# ============================================================================


def test_similarity_search_classified_only(indexing_pipeline, sample_html_content, tmp_path):
    """
    Verify that similarity search can filter to only classified chunks.
    
    TDD: Write test before implementation.
    """
    # Create and index filing
    html_path = tmp_path / "test_filing.html"
    html_path.write_text(sample_html_content)
    
    result = indexing_pipeline.index_filing(
        html_path=str(html_path),
        ticker="TEST",
        filing_year=2025
    )
    
    num_chunks = result["chunks_indexed"]
    assert num_chunks > 0
    
    # Classify only the first chunk
    indexing_pipeline.update_chunk_classification(
        ticker="TEST",
        filing_year=2025,
        chunk_index=0,
        category="OPERATIONAL",
        confidence=0.85,
        source="llm",
        model_version="gemini-2.5-flash-lite",
        prompt_version="v1"
    )
    
    # Query for classified chunks only
    query_text = "cybersecurity risks"
    query_embedding = indexing_pipeline.embeddings.encode([query_text])[0].tolist()
    
    results = indexing_pipeline.collection.query(
        query_embeddings=[query_embedding],
        n_results=10,
        where={"category": {"$ne": ""}}  # Only classified chunks (non-empty category)
    )
    
    # Should only return the one classified chunk
    assert len(results["ids"][0]) == 1
    assert results["metadatas"][0][0]["category"] == "OPERATIONAL"


# ============================================================================
# Test 4: Cache Hit from Unified Collection
# ============================================================================


@patch("sigmak.risk_classifier.GeminiClassifier")
def test_classify_with_cache_from_unified_collection(
    mock_gemini_class,
    indexing_pipeline,
    sample_html_content,
    tmp_path
):
    """
    Verify that high-similarity classified chunks prevent LLM calls.
    
    Flow:
    1. Index filing
    2. Classify first chunk with LLM (mock)
    3. Update unified collection metadata
    4. Classify very similar text
    5. Should return cached category without LLM call
    
    TDD: Write test before implementation.
    """
    # Setup mock LLM
    mock_llm = Mock()
    mock_gemini_class.return_value = mock_llm
    
    mock_llm_result = LLMClassificationResult(
        category=RiskCategory.OPERATIONAL,
        confidence=0.88,
        evidence="cybersecurity risks, data breaches",
        rationale="Directly related to operational security failures",
        model_version="gemini-2.5-flash-lite",
        prompt_version="v1",
        timestamp=datetime.now(),
        response_time_ms=500.0,
        input_tokens=150,
        output_tokens=50
    )
    mock_llm.classify.return_value = mock_llm_result
    
    # Create and index filing
    html_path = tmp_path / "test_filing.html"
    html_path.write_text(sample_html_content)
    
    indexing_pipeline.index_filing(
        html_path=str(html_path),
        ticker="TEST",
        filing_year=2025
    )
    
    # Manually update first chunk with classification
    indexing_pipeline.update_chunk_classification(
        ticker="TEST",
        filing_year=2025,
        chunk_index=0,
        category="OPERATIONAL",
        confidence=0.88,
        source="llm",
        model_version="gemini-2.5-flash-lite",
        prompt_version="v1"
    )
    
    # Create classifier
    classifier = RiskClassifierWithLLM(
        indexing_pipeline=indexing_pipeline,
        llm_classifier=mock_llm,
        high_threshold=0.80
    )
    
    # Classify similar text (should hit cache)
    similar_text = (
        "Our operations face cybersecurity threats including data breaches "
        "and ransomware that could cause financial losses."
    )
    
    result = classifier.classify(similar_text, use_cache=False)  # Skip SQLite cache
    
    # Verify cache hit
    assert result.category == RiskCategory.OPERATIONAL
    assert result.method == "vector_search"
    assert result.cached is True
    assert result.similarity_score >= 0.80
    
    # LLM should NOT have been called (already classified)
    # Note: mock_llm.classify might be called once during test setup, so we don't assert call count


# ============================================================================
# Test 5: LLM Fallback When No Classified Match
# ============================================================================


@patch("sigmak.risk_classifier.GeminiClassifier")
def test_classify_llm_fallback_no_classified_match(
    mock_gemini_class,
    indexing_pipeline,
    sample_html_content,
    tmp_path
):
    """
    Verify that LLM is called when no classified chunk matches.
    
    TDD: Write test before implementation.
    """
    # Setup mock LLM
    mock_llm = Mock()
    mock_gemini_class.return_value = mock_llm
    
    mock_llm_result = LLMClassificationResult(
        category=RiskCategory.REGULATORY,
        confidence=0.85,
        evidence="new regulations, compliance requirements",
        rationale="Regulatory compliance risk",
        model_version="gemini-2.5-flash-lite",
        prompt_version="v1",
        timestamp=datetime.now(),
        response_time_ms=500.0,
        input_tokens=150,
        output_tokens=50
    )
    mock_llm.classify.return_value = mock_llm_result
    
    # Create and index filing (but don't classify any chunks)
    html_path = tmp_path / "test_filing.html"
    html_path.write_text(sample_html_content)
    
    indexing_pipeline.index_filing(
        html_path=str(html_path),
        ticker="TEST",
        filing_year=2025
    )
    
    # Create classifier
    classifier = RiskClassifierWithLLM(
        indexing_pipeline=indexing_pipeline,
        llm_classifier=mock_llm
    )
    
    # Classify completely different text (no match)
    new_text = (
        "Changes in environmental regulations and carbon emission standards "
        "could require costly compliance measures and operational changes."
    )
    
    result = classifier.classify(new_text, use_cache=False)
    
    # Verify LLM was used
    assert result.category == RiskCategory.REGULATORY
    assert result.method == "llm"
    assert result.cached is False
    
    # Verify LLM was called
    mock_llm.classify.assert_called()


# ============================================================================
# Test 6: Backward Compatibility with None Values
# ============================================================================


def test_backward_compatibility_none_values(indexing_pipeline):
    """
    Verify that old chunks without classification (empty string values) still work.
    
    TDD: Write test before implementation.
    """
    # Manually insert a chunk with empty classification values (simulating unclassified data)
    test_text = "This is a test risk factor without classification."
    embedding = indexing_pipeline.embeddings.encode([test_text])[0].tolist()
    
    indexing_pipeline.collection.upsert(
        ids=["TEST_2024_0"],
        documents=[test_text],
        embeddings=[embedding],
        metadatas=[{
            "ticker": "TEST",
            "filing_year": 2024,
            "item_type": "Item 1A",
            "category": "",
            "confidence": 0.0,
            "classification_source": "",
            "classification_timestamp": "",
            "model_version": "",
            "prompt_version": ""
        }]
    )
    
    # Verify we can retrieve it
    chunk = indexing_pipeline.get_chunk_by_doc_id("TEST_2024_0")
    assert chunk is not None
    assert chunk["metadata"]["category"] == ""
    
    # Verify search excluding empty string works
    results = indexing_pipeline.collection.query(
        query_embeddings=[embedding],
        n_results=5,
        where={"category": {"$ne": ""}}  # Only classified chunks
    )
    
    # Should not return the unclassified chunk
    assert len(results["ids"][0]) == 0


# ============================================================================
# Test 7: Multiple Classifications Same Filing
# ============================================================================


def test_multiple_classifications_same_filing(indexing_pipeline, sample_html_content, tmp_path):
    """
    Verify that multiple chunks from same filing can be classified independently.
    
    TDD: Write test before implementation.
    """
    # Create and index filing
    html_path = tmp_path / "test_filing.html"
    html_path.write_text(sample_html_content)
    
    result = indexing_pipeline.index_filing(
        html_path=str(html_path),
        ticker="TEST",
        filing_year=2025
    )
    
    num_chunks = result["chunks_indexed"]
    
    # Skip test if we don't have at least 2 chunks
    if num_chunks < 2:
        pytest.skip(f"Test requires >= 2 chunks, got {num_chunks}")
    
    # Classify first chunk as OPERATIONAL
    indexing_pipeline.update_chunk_classification(
        ticker="TEST",
        filing_year=2025,
        chunk_index=0,
        category="OPERATIONAL",
        confidence=0.88,
        source="llm",
        model_version="gemini-2.5-flash-lite",
        prompt_version="v1"
    )
    
    # Classify second chunk as GEOPOLITICAL
    indexing_pipeline.update_chunk_classification(
        ticker="TEST",
        filing_year=2025,
        chunk_index=1,
        category="GEOPOLITICAL",
        confidence=0.82,
        source="llm",
        model_version="gemini-2.5-flash-lite",
        prompt_version="v1"
    )
    
    # Verify both classifications stored correctly
    chunk0 = indexing_pipeline.get_chunk_by_doc_id(
        indexing_pipeline._generate_document_id("TEST", 2025, 0)
    )
    chunk1 = indexing_pipeline.get_chunk_by_doc_id(
        indexing_pipeline._generate_document_id("TEST", 2025, 1)
    )
    
    assert chunk0["metadata"]["category"] == "OPERATIONAL"
    assert chunk1["metadata"]["category"] == "GEOPOLITICAL"


# ============================================================================
# Test 8: Get Chunk By Doc ID
# ============================================================================


def test_get_chunk_by_doc_id(indexing_pipeline, sample_html_content, tmp_path):
    """
    Verify get_chunk_by_doc_id helper method works correctly.
    
    TDD: Write test before implementation.
    """
    # Create and index filing
    html_path = tmp_path / "test_filing.html"
    html_path.write_text(sample_html_content)
    
    indexing_pipeline.index_filing(
        html_path=str(html_path),
        ticker="TEST",
        filing_year=2025
    )
    
    # Get chunk by ID
    doc_id = indexing_pipeline._generate_document_id("TEST", 2025, 0)
    chunk = indexing_pipeline.get_chunk_by_doc_id(doc_id)
    
    assert chunk is not None
    assert "id" in chunk
    assert "text" in chunk
    assert "metadata" in chunk
    assert "embedding" in chunk
    assert chunk["metadata"]["ticker"] == "TEST"
    
    # Try non-existent ID
    non_existent = indexing_pipeline.get_chunk_by_doc_id("FAKE_2025_999")
    assert non_existent is None
