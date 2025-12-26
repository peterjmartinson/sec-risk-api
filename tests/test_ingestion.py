import os
import pytest
from sec_risk_api.ingest import extract_text_from_pdf

def test_extract_text_is_non_empty():
    """
    Win Condition: Successfully extract string content from an SEC PDF.
    """
    sample_path = "data/sample_10k.pdf"
    
    # Check if the user (you) has provided the loot (data)
    if not os.path.exists(sample_path):
        pytest.skip(f"Level blocked: Please place a 10-K PDF at {sample_path}")

    text = extract_text_from_pdf(sample_path)
    
    assert isinstance(text, str)
    assert len(text) > 500  # A real 10-K is never just a few words
    assert "Item 1A" in text or "ITEM 1A" in text
