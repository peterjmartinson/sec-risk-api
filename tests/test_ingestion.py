import pytest
from sec_risk_api.ingest import parse_sec_html, extract_text_from_file, slice_risk_factors

# 1. Pure Atomic Logic Tests (In-Memory)
def test_parse_sec_html_removes_scripts() -> None:
    html = "<html><body><script>alert('bad');</script>Target Text</body></html>"
    result = parse_sec_html(html)
    assert "alert" not in result
    assert "Target Text" in result

def test_parse_sec_html_separates_tags() -> None:
    # Tests that <div>s don't result in mashedwords
    html = "<div>Word1</div><div>Word2</div>"
    result = parse_sec_html(html)
    assert result == "Word1 Word2"

# 2. IO / Integration Test (Using tmp_path)
def test_extract_text_from_file_handles_encoding(tmp_path) -> None:
    # Create a file with a non-UTF-8 character
    p = tmp_path / "legacy.html"
    # Writing a character that might trigger encoding issues
    p.write_bytes("<html><body>Item 1A ©</body></html>".encode('cp1252'))

    result = extract_text_from_file(p)
    assert "Item 1A" in result

def test_slice_risk_factors_isolates_content() -> None:
    sample = "ITEM 1. BUSINESS... ITEM 1A. RISK FACTORS. This is the risk. ITEM 2. PROPERTIES..."
    sliced = slice_risk_factors(sample)

    assert "BUSINESS" not in sliced.upper()
    assert "RISK FACTORS" in sliced.upper()
    assert "PROPERTIES" not in sliced.upper()

def test_slice_risk_factors_fallback_on_no_match() -> None:
    """
    Onion Check: Ensure function returns full text if 'Item 1A' marker is missing.
    """
    garbage_text = "This is a random document with no specific SEC markers."
    
    # We expect the function to return the original text untouched
    result = slice_risk_factors(garbage_text)
    
    assert result == garbage_text
    # Optional: If you use pytest's caplog, you can even verify a warning was logged
