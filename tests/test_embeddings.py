# Copyright (c) 2025 Peter Martinson, Distracted Fortune. All rights reserved.
# This software is proprietary and not licensed for use, modification, or distribution.

import pytest
from sec_risk_api.embeddings import EmbeddingEngine


@pytest.fixture(scope="session")
def embedding_engine() -> EmbeddingEngine:
    """
    Session-scoped fixture that initializes EmbeddingEngine once per test session.
    This avoids redundant model downloads and initialization.
    """
    return EmbeddingEngine()


@pytest.mark.slow
def test_encode_returns_float32_array(embedding_engine: EmbeddingEngine) -> None:
    """
    Verify that encode() returns a numpy array of float32 type.
    """
    texts = ["Hello world"]
    result = embedding_engine.encode(texts)
    
    assert result.dtype == "float32", f"Expected float32, got {result.dtype}"
    assert result.shape == (1, 384), f"Expected shape (1, 384), got {result.shape}"


@pytest.mark.slow
def test_encode_handles_multiple_texts(embedding_engine: EmbeddingEngine) -> None:
    """
    Verify that encode() correctly processes batches of texts.
    """
    texts = ["First text", "Second text", "Third text"]
    result = embedding_engine.encode(texts)
    
    assert result.shape[0] == 3, f"Expected 3 embeddings, got {result.shape[0]}"
    assert result.shape[1] == 384, f"Expected 384 dimensions, got {result.shape[1]}"


@pytest.mark.slow
def test_similarity_financial_vs_unrelated(embedding_engine: EmbeddingEngine) -> None:
    """
    Verify that semantically related financial texts have higher similarity
    than unrelated texts.
    """
    financial_1 = "Market volatility is high"
    financial_2 = "The markets are swinging significantly"
    unrelated = "The seasonal migration of birds is beginning"

    sim_financial = embedding_engine.get_similarity(financial_1, financial_2)
    sim_unrelated = embedding_engine.get_similarity(financial_1, unrelated)

    print(f"\n[Similarity Test] Related texts: {sim_financial:.4f}")
    print(f"[Similarity Test] Unrelated texts: {sim_unrelated:.4f}")

    # The all-MiniLM-L6-v2 model typically returns scores in [0, 1]
    # Related financial texts should score higher than unrelated texts
    assert sim_financial > sim_unrelated, (
        f"Related texts should have higher similarity ({sim_financial:.4f}) "
        f"than unrelated texts ({sim_unrelated:.4f})"
    )


@pytest.mark.slow
def test_similarity_identical_texts(embedding_engine: EmbeddingEngine) -> None:
    """
    Verify that identical texts produce maximum similarity (near 1.0).
    """
    text = "Geopolitical instability increases market risk"
    sim = embedding_engine.get_similarity(text, text)

    assert sim >= 0.99, f"Identical texts should have similarity >= 0.99, got {sim:.4f}"
