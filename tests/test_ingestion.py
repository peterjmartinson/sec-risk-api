# Copyright (c) 2025 Peter Martinson, Distracted Fortune. All rights reserved.
# This software is proprietary and not licensed for use, modification, or distribution.

import pytest

from sigmak.ingest import (extract_text_from_file, parse_sec_html,
                                 slice_risk_factors)


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

def test_slice_risk_factors_skips_toc_entry() -> None:
    """
    Critical: Verify extraction skips TOC and captures actual content section.

    Reproduces the bug where TOC entries like "Item 1A. Risk Factors 14" were
    being extracted instead of the actual multi-thousand word risk disclosure.
    """
    # Simulate a filing with TOC entry followed by actual section
    sample_text = """
    TABLE OF CONTENTS
    ITEM 1. BUSINESS 10
    ITEM 1A. RISK FACTORS 14
    ITEM 1B. UNRESOLVED STAFF COMMENTS 45

    [... lots of other content ...]

    ITEM 1. BUSINESS
    We design and manufacture electric vehicles.

    ITEM 1A. RISK FACTORS
    Investing in our common stock involves risks. Our business operations face
    significant challenges related to supply chain disruptions, manufacturing
    delays, regulatory compliance, and competitive pressures. The COVID-19
    pandemic has adversely impacted our operations and financial results.
    We rely heavily on international suppliers for critical battery components.
    Any disruption to these supply chains could materially harm our business.
    [... many more paragraphs of actual risk content ...]

    ITEM 1B. UNRESOLVED STAFF COMMENTS
    None.
    """

    sliced = slice_risk_factors(sample_text)

    # Should NOT be just a TOC reference
    assert "ITEM 1A. RISK FACTORS 14" != sliced.strip()

    # Should have substantial content (>500 chars minimum)
    assert len(sliced) > 500, f"Expected substantial content, got {len(sliced)} chars"

    # Should contain actual risk prose, not just headings
    assert "supply chain" in sliced.lower()
    assert "COVID-19" in sliced or "pandemic" in sliced.lower()
    assert "common stock involves risks" in sliced.lower()

    # Should NOT include the TOC section
    assert "TABLE OF CONTENTS" not in sliced

    # Should NOT include Item 1B content
    assert "UNRESOLVED STAFF COMMENTS" not in sliced or sliced.index("UNRESOLVED") > sliced.index("RISK FACTORS")
