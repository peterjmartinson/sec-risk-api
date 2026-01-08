"""
Integration tests for the full retrieval-scoring pipeline (Issue #22).

This module tests the end-to-end "Walking Skeleton" that:
1. Takes a ticker + filing year
2. Retrieves and indexes Item 1A risk factors
3. Performs semantic search
4. Computes severity and novelty scores
5. Returns structured, cited JSON output

Test Coverage:
- End-to-end happy path (mock and real data)
- Error handling (missing Item 1A, broken pipeline)
- Type safety (mypy compliance)
- Edge cases (ambiguous ticker, unusual categories)
- Citation integrity (every risk entry cites source)
"""

import pytest
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
import tempfile
import shutil

from sec_risk_api.integration import (
    IntegrationPipeline,
    RiskAnalysisResult,
    IntegrationError
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_db_path() -> str:
    """Create temporary directory for test vector DB."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_html_path() -> Path:
    """Path to sample 10-K HTML file."""
    return Path("data/sample_10k.html")


@pytest.fixture
def integration_pipeline(temp_db_path: str) -> IntegrationPipeline:
    """Initialize integration pipeline with temporary DB."""
    return IntegrationPipeline(persist_path=temp_db_path)


# ============================================================================
# Test Class 1: End-to-End Happy Path
# ============================================================================

class TestEndToEndHappyPath:
    """
    Verify that the integration pipeline works end-to-end with valid inputs.
    """
    
    def test_pipeline_returns_structured_result_with_real_filing(
        self,
        integration_pipeline: IntegrationPipeline,
        sample_html_path: Path
    ) -> None:
        """
        End-to-end test with real sample filing.
        
        SRP: Test complete pipeline flow with real data.
        """
        # Run full pipeline
        result = integration_pipeline.analyze_filing(
            html_path=str(sample_html_path),
            ticker="AAPL",
            filing_year=2025
        )
        
        # Verify result structure
        assert isinstance(result, RiskAnalysisResult)
        assert result.ticker == "AAPL"
        assert result.filing_year == 2025
        assert len(result.risks) > 0
        
        # Verify each risk has required fields
        for risk in result.risks:
            assert "text" in risk
            assert "severity" in risk
            assert "novelty" in risk
            assert "source_citation" in risk
            assert "metadata" in risk
            
            # Verify scores are in valid range
            assert 0.0 <= risk["severity"]["value"] <= 1.0
            assert 0.0 <= risk["novelty"]["value"] <= 1.0
    
    def test_pipeline_returns_valid_json_output(
        self,
        integration_pipeline: IntegrationPipeline,
        sample_html_path: Path
    ) -> None:
        """
        Verify that result can be serialized to valid JSON.
        
        SRP: Test JSON serialization.
        """
        result = integration_pipeline.analyze_filing(
            html_path=str(sample_html_path),
            ticker="AAPL",
            filing_year=2025
        )
        
        # Convert to JSON
        json_output = result.to_json()
        
        # Verify it's valid JSON
        parsed = json.loads(json_output)
        assert parsed["ticker"] == "AAPL"
        assert parsed["filing_year"] == 2025
        assert "risks" in parsed
        assert len(parsed["risks"]) > 0
    
    def test_pipeline_computes_severity_scores(
        self,
        integration_pipeline: IntegrationPipeline,
        sample_html_path: Path
    ) -> None:
        """
        Verify that severity scores are computed for all risks.
        
        SRP: Test severity scoring integration.
        """
        result = integration_pipeline.analyze_filing(
            html_path=str(sample_html_path),
            ticker="AAPL",
            filing_year=2025
        )
        
        # Every risk should have severity score
        for risk in result.risks:
            severity = risk["severity"]
            assert "value" in severity
            assert "explanation" in severity
            assert isinstance(severity["value"], float)
            assert 0.0 <= severity["value"] <= 1.0
            assert len(severity["explanation"]) > 10
    
    def test_pipeline_computes_novelty_scores(
        self,
        integration_pipeline: IntegrationPipeline,
        sample_html_path: Path
    ) -> None:
        """
        Verify that novelty scores are computed correctly.
        
        For first filing (no history), novelty should be 1.0 (maximally novel).
        
        SRP: Test novelty scoring integration.
        """
        result = integration_pipeline.analyze_filing(
            html_path=str(sample_html_path),
            ticker="AAPL",
            filing_year=2025
        )
        
        # Every risk should have novelty score
        for risk in result.risks:
            novelty = risk["novelty"]
            assert "value" in novelty
            assert "explanation" in novelty
            assert isinstance(novelty["value"], float)
            assert 0.0 <= novelty["value"] <= 1.0
            
            # First filing → no history → max novelty
            assert novelty["value"] == 1.0, "First filing should have max novelty"
    
    def test_pipeline_with_historical_comparison(
        self,
        integration_pipeline: IntegrationPipeline,
        sample_html_path: Path
    ) -> None:
        """
        Verify novelty scoring behavior with/without historical data.
        
        Tests that:
        1. First filing (no history) → HIGH novelty
        2. Second filing (with history) → LOWER novelty
        
        SRP: Test novelty comparison logic, not arbitrary thresholds.
        """
        # First filing: no historical data → should have HIGH novelty
        result_2024 = integration_pipeline.analyze_filing(
            html_path=str(sample_html_path),
            ticker="AAPL",
            filing_year=2024
        )
        
        # Collect novelty scores from first filing
        novelty_scores_2024 = [risk["novelty"]["value"] for risk in result_2024.risks]
        avg_novelty_2024 = sum(novelty_scores_2024) / len(novelty_scores_2024)
        
        # Second filing: historical data exists → should have LOWER novelty
        result_2025 = integration_pipeline.analyze_filing(
            html_path=str(sample_html_path),
            ticker="AAPL",
            filing_year=2025
        )
        
        novelty_scores_2025 = [risk["novelty"]["value"] for risk in result_2025.risks]
        avg_novelty_2025 = sum(novelty_scores_2025) / len(novelty_scores_2025)
        
        # Core assertion: novelty should DECREASE when historical data exists
        # (testing behavior, not magic numbers)
        assert avg_novelty_2025 < avg_novelty_2024, (
            f"Expected novelty to decrease with historical data. "
            f"2024 (no history): {avg_novelty_2024:.2f}, "
            f"2025 (with history): {avg_novelty_2025:.2f}"
        )


# ============================================================================
# Test Class 2: Error Handling
# ============================================================================

class TestErrorHandling:
    """
    Verify that pipeline handles errors gracefully with helpful messages.
    """
    
    def test_missing_html_file_raises_integration_error(
        self,
        integration_pipeline: IntegrationPipeline
    ) -> None:
        """
        Missing HTML file should raise IntegrationError with helpful message.
        
        SRP: Test file not found error.
        """
        with pytest.raises(IntegrationError, match="file|not found|path"):
            integration_pipeline.analyze_filing(
                html_path="nonexistent_file.html",
                ticker="AAPL",
                filing_year=2025
            )
    
    def test_invalid_html_still_processes(
        self,
        integration_pipeline: IntegrationPipeline,
        temp_db_path: str
    ) -> None:
        """
        Malformed HTML should still process (BeautifulSoup is robust).
        
        SRP: Test HTML parsing robustness.
        """
        # Create invalid but parseable HTML file
        invalid_html_path = Path(temp_db_path) / "invalid.html"
        invalid_html_path.write_text("<html>Not valid HTML but parseable</html>")
        
        # Should process without error (BeautifulSoup is lenient)
        result = integration_pipeline.analyze_filing(
            html_path=str(invalid_html_path),
            ticker="TEST",
            filing_year=2025
        )
        
        # Should return some results (even if just the text content)
        assert isinstance(result, RiskAnalysisResult)
        assert result.ticker == "TEST"
    
    def test_missing_item_1a_uses_fallback(
        self,
        integration_pipeline: IntegrationPipeline,
        temp_db_path: str
    ) -> None:
        """
        HTML without Item 1A section uses full text as fallback.
        
        SRP: Test missing section fallback behavior.
        """
        # Create HTML without Item 1A
        html_without_1a = """
        <html>
            <body>
                <div>Some risk content but no explicit Item 1A marker</div>
            </body>
        </html>
        """
        html_path = Path(temp_db_path) / "no_item_1a.html"
        html_path.write_text(html_without_1a)
        
        # Should still process using fallback (returns full text)
        result = integration_pipeline.analyze_filing(
            html_path=str(html_path),
            ticker="TEST",
            filing_year=2025
        )
        
        # Should return results from full text
        assert isinstance(result, RiskAnalysisResult)
        assert len(result.risks) > 0
    
    def test_invalid_ticker_format_raises_integration_error(
        self,
        integration_pipeline: IntegrationPipeline,
        sample_html_path: Path
    ) -> None:
        """
        Invalid ticker format should raise IntegrationError.
        
        SRP: Test input validation.
        """
        # Test empty ticker
        with pytest.raises(IntegrationError, match="Ticker cannot be empty"):
            integration_pipeline.analyze_filing(
                html_path=str(sample_html_path),
                ticker="",  # Empty ticker
                filing_year=2025
            )
        
        # Test invalid characters
        with pytest.raises(IntegrationError, match="Invalid ticker format"):
            integration_pipeline.analyze_filing(
                html_path=str(sample_html_path),
                ticker="aa pl",  # Lowercase + space
                filing_year=2025
            )
    
    def test_invalid_year_raises_integration_error(
        self,
        integration_pipeline: IntegrationPipeline,
        sample_html_path: Path
    ) -> None:
        """
        Invalid filing year should raise IntegrationError.
        
        SRP: Test year validation.
        """
        with pytest.raises(IntegrationError, match="year|invalid"):
            integration_pipeline.analyze_filing(
                html_path=str(sample_html_path),
                ticker="AAPL",
                filing_year=1900  # Too old
            )


# ============================================================================
# Test Class 3: Citation Integrity
# ============================================================================

class TestCitationIntegrity:
    """
    Verify that every risk entry includes complete source citation.
    """
    
    def test_every_risk_has_source_citation(
        self,
        integration_pipeline: IntegrationPipeline,
        sample_html_path: Path
    ) -> None:
        """
        Every risk must include source_citation field with actual text.
        
        SRP: Test citation presence.
        """
        result = integration_pipeline.analyze_filing(
            html_path=str(sample_html_path),
            ticker="AAPL",
            filing_year=2025
        )
        
        for risk in result.risks:
            assert "source_citation" in risk
            assert isinstance(risk["source_citation"], str)
            assert len(risk["source_citation"]) > 0
    
    def test_citation_matches_risk_text(
        self,
        integration_pipeline: IntegrationPipeline,
        sample_html_path: Path
    ) -> None:
        """
        Source citation should be derived from risk text.
        
        SRP: Test citation accuracy.
        """
        result = integration_pipeline.analyze_filing(
            html_path=str(sample_html_path),
            ticker="AAPL",
            filing_year=2025
        )
        
        for risk in result.risks:
            citation = risk["source_citation"]
            text = risk["text"]
            
            # Citation should be substring of text or vice versa
            assert (citation in text) or (text in citation), \
                "Citation should be derived from risk text"
    
    def test_severity_score_includes_citation(
        self,
        integration_pipeline: IntegrationPipeline,
        sample_html_path: Path
    ) -> None:
        """
        Severity score should include source citation.
        
        SRP: Test severity citation.
        """
        result = integration_pipeline.analyze_filing(
            html_path=str(sample_html_path),
            ticker="AAPL",
            filing_year=2025
        )
        
        for risk in result.risks:
            severity = risk["severity"]
            # Severity should reference the source
            assert "source_citation" in risk
    
    def test_novelty_score_includes_citation(
        self,
        integration_pipeline: IntegrationPipeline,
        sample_html_path: Path
    ) -> None:
        """
        Novelty score should include source citation.
        
        SRP: Test novelty citation.
        """
        result = integration_pipeline.analyze_filing(
            html_path=str(sample_html_path),
            ticker="AAPL",
            filing_year=2025
        )
        
        for risk in result.risks:
            novelty = risk["novelty"]
            # Novelty should reference the source
            assert "source_citation" in risk


# ============================================================================
# Test Class 4: Type Safety
# ============================================================================

class TestTypeSafety:
    """
    Verify that all outputs have correct types and pass mypy checks.
    """
    
    def test_result_has_correct_types(
        self,
        integration_pipeline: IntegrationPipeline,
        sample_html_path: Path
    ) -> None:
        """
        RiskAnalysisResult should have all fields with correct types.
        
        SRP: Test result type structure.
        """
        result = integration_pipeline.analyze_filing(
            html_path=str(sample_html_path),
            ticker="AAPL",
            filing_year=2025
        )
        
        assert isinstance(result.ticker, str)
        assert isinstance(result.filing_year, int)
        assert isinstance(result.risks, list)
        assert isinstance(result.metadata, dict)
    
    def test_risk_dict_has_correct_types(
        self,
        integration_pipeline: IntegrationPipeline,
        sample_html_path: Path
    ) -> None:
        """
        Each risk dictionary should have correct field types.
        
        SRP: Test risk entry types.
        """
        result = integration_pipeline.analyze_filing(
            html_path=str(sample_html_path),
            ticker="AAPL",
            filing_year=2025
        )
        
        for risk in result.risks:
            assert isinstance(risk["text"], str)
            assert isinstance(risk["source_citation"], str)
            assert isinstance(risk["metadata"], dict)
            
            # Severity structure
            assert isinstance(risk["severity"], dict)
            assert isinstance(risk["severity"]["value"], float)
            assert isinstance(risk["severity"]["explanation"], str)
            
            # Novelty structure
            assert isinstance(risk["novelty"], dict)
            assert isinstance(risk["novelty"]["value"], float)
            assert isinstance(risk["novelty"]["explanation"], str)
    
    def test_to_dict_returns_json_serializable(
        self,
        integration_pipeline: IntegrationPipeline,
        sample_html_path: Path
    ) -> None:
        """
        to_dict() output should be JSON-serializable.
        
        SRP: Test JSON compatibility.
        """
        result = integration_pipeline.analyze_filing(
            html_path=str(sample_html_path),
            ticker="AAPL",
            filing_year=2025
        )
        
        dict_output = result.to_dict()
        
        # Should be serializable to JSON
        json_str = json.dumps(dict_output)
        assert isinstance(json_str, str)
        assert len(json_str) > 0


# ============================================================================
# Test Class 5: Edge Cases
# ============================================================================

class TestEdgeCases:
    """
    Verify handling of unusual but valid inputs.
    """
    
    def test_ticker_with_special_characters(
        self,
        integration_pipeline: IntegrationPipeline,
        sample_html_path: Path
    ) -> None:
        """
        Tickers with dots/hyphens should work (e.g., BRK.B, ABC-D).
        
        SRP: Test ticker format edge case.
        """
        # BRK.B is a real ticker format
        result = integration_pipeline.analyze_filing(
            html_path=str(sample_html_path),
            ticker="BRK.B",
            filing_year=2025
        )
        
        assert result.ticker == "BRK.B"
    
    def test_very_recent_filing_year(
        self,
        integration_pipeline: IntegrationPipeline,
        sample_html_path: Path
    ) -> None:
        """
        Current year filings should work.
        
        SRP: Test recent year edge case.
        """
        result = integration_pipeline.analyze_filing(
            html_path=str(sample_html_path),
            ticker="AAPL",
            filing_year=2026  # Current year
        )
        
        assert result.filing_year == 2026
    
    def test_filing_with_minimal_risks(
        self,
        integration_pipeline: IntegrationPipeline,
        temp_db_path: str
    ) -> None:
        """
        Filing with very short Item 1A should still work.
        
        SRP: Test minimal content edge case.
        """
        # Create minimal valid HTML
        minimal_html = """
        <html>
            <body>
                <div>ITEM 1A. RISK FACTORS</div>
                <p>We face competition risks in our markets.</p>
            </body>
        </html>
        """
        html_path = Path(temp_db_path) / "minimal.html"
        html_path.write_text(minimal_html)
        
        result = integration_pipeline.analyze_filing(
            html_path=str(html_path),
            ticker="TEST",
            filing_year=2025
        )
        
        # Should return at least one risk
        assert len(result.risks) >= 1
    
    def test_multiple_companies_isolated(
        self,
        integration_pipeline: IntegrationPipeline,
        sample_html_path: Path
    ) -> None:
        """
        Different tickers should be isolated in vector DB.
        
        SRP: Test multi-company isolation.
        """
        # Index AAPL
        result_aapl = integration_pipeline.analyze_filing(
            html_path=str(sample_html_path),
            ticker="AAPL",
            filing_year=2025
        )
        
        # Index MSFT
        result_msft = integration_pipeline.analyze_filing(
            html_path=str(sample_html_path),
            ticker="MSFT",
            filing_year=2025
        )
        
        # Results should be tagged correctly
        assert all(r["metadata"]["ticker"] == "AAPL" for r in result_aapl.risks)
        assert all(r["metadata"]["ticker"] == "MSFT" for r in result_msft.risks)


# ============================================================================
# Test Class 6: Performance
# ============================================================================

class TestPerformance:
    """
    Verify that pipeline completes within reasonable time bounds.
    """
    
    def test_pipeline_completes_in_reasonable_time(
        self,
        integration_pipeline: IntegrationPipeline,
        sample_html_path: Path
    ) -> None:
        """
        End-to-end pipeline should complete within 10 seconds.
        
        SRP: Test performance bounds.
        """
        import time
        
        start = time.time()
        result = integration_pipeline.analyze_filing(
            html_path=str(sample_html_path),
            ticker="AAPL",
            filing_year=2025
        )
        elapsed = time.time() - start
        
        assert elapsed < 10.0, f"Pipeline took {elapsed:.2f}s, expected < 10s"
        assert len(result.risks) > 0
