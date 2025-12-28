import pytest
from sec_risk_api.ingest import parse_sec_html, extract_text_from_file

# 1. Pure Atomic Logic Tests (In-Memory)
def test_parse_sec_html_removes_scripts():
    html = "<html><body><script>alert('bad');</script>Target Text</body></html>"
    result = parse_sec_html(html)
    assert "alert" not in result
    assert "Target Text" in result

def test_parse_sec_html_separates_tags():
    # Tests that <div>s don't result in mashedwords
    html = "<div>Word1</div><div>Word2</div>"
    result = parse_sec_html(html)
    assert result == "Word1 Word2"

# 2. IO / Integration Test (Using tmp_path)
def test_extract_text_from_file_handles_encoding(tmp_path):
    # Create a file with a non-UTF-8 character
    p = tmp_path / "legacy.html"
    # Writing a character that might trigger encoding issues
    p.write_bytes("<html><body>Item 1A ©</body></html>".encode('cp1252'))
    
    result = extract_text_from_file(p)
    assert "Item 1A" in result
