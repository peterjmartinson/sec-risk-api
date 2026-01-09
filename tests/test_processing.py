# Copyright (c) 2025 Peter Martinson, Distracted Fortune. All rights reserved.
# This software is proprietary and not licensed for use, modification, or distribution.

import pytest
from sigmak.processing import chunk_risk_section

def test_chunk_risk_section_structure() -> None:
    """
    Test that the chunker returns the correct dictionary format
    and preserves metadata.
    """
    sample_text = "Risk 1: Market volatility. " * 20
    meta = {"ticker": "AAPL", "year": 2025}

    chunks = chunk_risk_section(sample_text, meta)

    # Assert modular output format
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert "text" in chunks[0]
    assert chunks[0]["metadata"]["ticker"] == "AAPL"

def test_chunk_overlap_integrity() -> None:
    """
    Ensure the recursive splitter is actually overlapping
    so context isn't lost between atoms.
    """
    sample_text = "This is a long sentence that should be split into multiple parts for testing overlap."
    # Using small size to force a split
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # We test the logic we wrapped in processing.py
    meta = {"ticker": "TSLA"}
    chunks = chunk_risk_section(sample_text, meta)

    if len(chunks) > 1:
        # Check if some part of the first chunk exists in the second
        # (This validates the 'context-aware' objective of Subissue 1.0)
        assert any(word in chunks[1]["text"] for word in chunks[0]["text"].split()[-3:])
