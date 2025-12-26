import os
import pytest
from sec_risk_api.ingest import extract_text_from_html


def test_extract_text_is_non_empty():
    """
    Win Condition: Successfully extract string content from an SEC PDF.
    """
    sample_path = "data/sample_10k.html"

    # Check if the user (you) has provided the loot (data)
    if not os.path.exists(sample_path):
        pytest.skip(f"Level blocked: Please place a 10-K PDF at {sample_path}")

    text = extract_text_from_html(sample_path)

    assert isinstance(text, str)
    assert len(text) > 500  # A real 10-K is never just a few words
    assert "Item 1A" in text or "ITEM 1A" in text


def test_extract_text_is_functional():
    """
    Win Condition: Extraction returns a string with expected 10-K markers from HTM.
    """
    sample_path = "data/sample_10k.html"

    # Requirement: You must have an HTM in data/ for this to pass
    if not os.path.exists(sample_path):
        pytest.fail("Combat aborted: Place an SEC HTM at data/sample_10k.htm first.")

    text = extract_text_from_html(sample_path)

    assert isinstance(text, str)
    assert len(text) > 1000  # Ensuring it's not a skeleton

    # Validation of SEC content markers
    upper_text = text.upper()
    assert any(term in upper_text for term in ["FORM 10-K", "ANNUAL REPORT"])
    assert "ITEM 1A" in upper_text
