"""
Unit tests for retrieval-augmented risk scoring logic (Issue #21).

This module tests Severity and Novelty scoring for SEC risk disclosures.
All scores must be traceable to cited source chunks.

Test Coverage:
- Score calculation correctness (synthetic data)
- Edge cases (ambiguous risks, missing historical data)
- Type safety (mypy compliance)
- Source citation integrity (every score has provenance)
- Failure handling (graceful degradation)
"""

import pytest
from typing import Dict, Any, List, Optional
import numpy as np
from numpy.typing import NDArray

from sec_risk_api.scoring import (
    RiskScore,
    RiskScorer,
    ScoringError
)


# ============================================================================
# Test Class 1: Score Calculation Correctness
# ============================================================================

class TestScoreCalculationCorrectness:
    """
    Verify that severity and novelty scores are computed correctly
    on synthetic/controlled data.
    """
    
    def test_severity_score_is_between_0_and_1(self) -> None:
        """
        Severity scores must be normalized to [0.0, 1.0] range.
        
        SRP: Test only score range validation.
        """
        scorer = RiskScorer()
        
        # Synthetic chunk with high-severity keywords
        chunk = {
            "text": "Catastrophic supply chain disruptions and unprecedented geopolitical conflicts",
            "metadata": {"ticker": "AAPL", "filing_year": 2025}
        }
        
        score = scorer.calculate_severity(chunk)
        
        assert 0.0 <= score.value <= 1.0, f"Severity score {score.value} out of bounds"
        assert score.source_citation == chunk["text"]
    
    def test_novelty_score_is_between_0_and_1(self) -> None:
        """
        Novelty scores must be normalized to [0.0, 1.0] range.
        
        SRP: Test only score range validation.
        """
        scorer = RiskScorer()
        
        current_chunk = {
            "text": "New AI regulation compliance requirements emerging",
            "metadata": {"ticker": "AAPL", "filing_year": 2025}
        }
        
        historical_chunks = [
            {
                "text": "Standard operational risks and market competition",
                "metadata": {"ticker": "AAPL", "filing_year": 2024}
            }
        ]
        
        score = scorer.calculate_novelty(current_chunk, historical_chunks)
        
        assert 0.0 <= score.value <= 1.0, f"Novelty score {score.value} out of bounds"
        assert score.source_citation == current_chunk["text"]
    
    def test_severity_increases_with_severe_keywords(self) -> None:
        """
        Chunks with severe language should score higher than neutral chunks.
        
        SRP: Test severity keyword sensitivity.
        """
        scorer = RiskScorer()
        
        severe_chunk = {
            "text": "Catastrophic failure, existential threat, unprecedented crisis",
            "metadata": {"ticker": "AAPL", "filing_year": 2025}
        }
        
        neutral_chunk = {
            "text": "Standard business operations may be affected by market conditions",
            "metadata": {"ticker": "AAPL", "filing_year": 2025}
        }
        
        severe_score = scorer.calculate_severity(severe_chunk)
        neutral_score = scorer.calculate_severity(neutral_chunk)
        
        assert severe_score.value > neutral_score.value, \
            f"Severe chunk ({severe_score.value}) should score higher than neutral ({neutral_score.value})"
    
    def test_novelty_increases_with_semantic_distance(self) -> None:
        """
        Chunks semantically distant from historical filings should have higher novelty.
        
        SRP: Test novelty distance sensitivity.
        """
        scorer = RiskScorer()
        
        # Novel risk: quantum computing threats (new topic)
        novel_chunk = {
            "text": "Quantum computing advances threaten current encryption standards",
            "metadata": {"ticker": "AAPL", "filing_year": 2025}
        }
        
        # Repetitive risk: standard competition language
        repetitive_chunk = {
            "text": "Competition in technology markets remains intense",
            "metadata": {"ticker": "AAPL", "filing_year": 2025}
        }
        
        # Historical context: generic competition risks
        historical_chunks = [
            {
                "text": "We face significant competition in our markets",
                "metadata": {"ticker": "AAPL", "filing_year": 2024}
            },
            {
                "text": "Competitive pressures may impact our market share",
                "metadata": {"ticker": "AAPL", "filing_year": 2023}
            }
        ]
        
        novel_score = scorer.calculate_novelty(novel_chunk, historical_chunks)
        repetitive_score = scorer.calculate_novelty(repetitive_chunk, historical_chunks)
        
        assert novel_score.value > repetitive_score.value, \
            f"Novel chunk ({novel_score.value}) should score higher than repetitive ({repetitive_score.value})"


# ============================================================================
# Test Class 2: Edge Case Handling
# ============================================================================

class TestEdgeCaseHandling:
    """
    Verify graceful handling of ambiguous, missing, or degenerate inputs.
    """
    
    def test_empty_historical_data_returns_max_novelty(self) -> None:
        """
        When no historical data exists, novelty should default to 1.0 (maximally novel).
        
        SRP: Test missing historical data edge case.
        """
        scorer = RiskScorer()
        
        chunk = {
            "text": "New risk disclosure",
            "metadata": {"ticker": "NEWCO", "filing_year": 2025}
        }
        
        score = scorer.calculate_novelty(chunk, historical_chunks=[])
        
        assert score.value == 1.0, "Empty history should yield max novelty"
        assert "no historical data" in score.explanation.lower()
    
    def test_single_word_chunk_handled_gracefully(self) -> None:
        """
        Degenerate chunks (single word) should not crash scoring.
        
        SRP: Test minimal input edge case.
        """
        scorer = RiskScorer()
        
        chunk = {
            "text": "Risk",
            "metadata": {"ticker": "AAPL", "filing_year": 2025}
        }
        
        score = scorer.calculate_severity(chunk)
        
        # Should return a valid score, even if low
        assert 0.0 <= score.value <= 1.0
        assert score.source_citation == "Risk"
    
    def test_extremely_long_chunk_handled_gracefully(self) -> None:
        """
        Very long chunks should be processed without errors.
        
        SRP: Test large input edge case.
        """
        scorer = RiskScorer()
        
        # 1000-word chunk
        long_text = " ".join(["risk disclosure"] * 1000)
        chunk = {
            "text": long_text,
            "metadata": {"ticker": "AAPL", "filing_year": 2025}
        }
        
        score = scorer.calculate_severity(chunk)
        
        assert 0.0 <= score.value <= 1.0
        # Citation should be truncated for readability
        assert len(score.source_citation) <= 500, "Citation should be truncated for long chunks"
    
    def test_identical_current_and_historical_chunks_yield_zero_novelty(self) -> None:
        """
        If current chunk is identical to historical, novelty should be ~0.0.
        
        SRP: Test zero-novelty edge case.
        """
        scorer = RiskScorer()
        
        text = "Standard competition risks in technology markets"
        current_chunk = {
            "text": text,
            "metadata": {"ticker": "AAPL", "filing_year": 2025}
        }
        
        historical_chunks = [
            {
                "text": text,  # Exact duplicate
                "metadata": {"ticker": "AAPL", "filing_year": 2024}
            }
        ]
        
        score = scorer.calculate_novelty(current_chunk, historical_chunks)
        
        assert score.value < 0.1, f"Identical chunks should have near-zero novelty, got {score.value}"
    
    def test_missing_metadata_raises_scoring_error(self) -> None:
        """
        Chunks without required metadata should raise ScoringError.
        
        SRP: Test input validation.
        """
        scorer = RiskScorer()
        
        invalid_chunk = {
            "text": "Some risk disclosure",
            # Missing metadata
        }
        
        with pytest.raises(ScoringError, match="metadata"):
            scorer.calculate_severity(invalid_chunk)


# ============================================================================
# Test Class 3: Source Citation Integrity
# ============================================================================

class TestSourceCitationIntegrity:
    """
    Verify that every score includes complete source provenance.
    """
    
    def test_severity_score_includes_source_citation(self) -> None:
        """
        Every severity score must cite the source chunk.
        
        SRP: Test citation presence.
        """
        scorer = RiskScorer()
        
        chunk = {
            "text": "Operational disruptions due to supply chain issues",
            "metadata": {"ticker": "AAPL", "filing_year": 2025}
        }
        
        score = scorer.calculate_severity(chunk)
        
        assert score.source_citation is not None
        assert len(score.source_citation) > 0
        assert "supply chain" in score.source_citation.lower()
    
    def test_novelty_score_includes_source_citation(self) -> None:
        """
        Every novelty score must cite the source chunk.
        
        SRP: Test citation presence.
        """
        scorer = RiskScorer()
        
        current_chunk = {
            "text": "Emerging quantum threats to encryption",
            "metadata": {"ticker": "AAPL", "filing_year": 2025}
        }
        
        historical_chunks = [
            {
                "text": "Standard cybersecurity risks",
                "metadata": {"ticker": "AAPL", "filing_year": 2024}
            }
        ]
        
        score = scorer.calculate_novelty(current_chunk, historical_chunks)
        
        assert score.source_citation is not None
        assert "quantum" in score.source_citation.lower()
    
    def test_score_includes_explanation(self) -> None:
        """
        Scores must include human-readable explanation of calculation.
        
        SRP: Test explanation presence.
        """
        scorer = RiskScorer()
        
        chunk = {
            "text": "Catastrophic supply chain failure",
            "metadata": {"ticker": "AAPL", "filing_year": 2025}
        }
        
        score = scorer.calculate_severity(chunk)
        
        assert score.explanation is not None
        assert len(score.explanation) > 10, "Explanation should be substantive"
    
    def test_score_includes_metadata(self) -> None:
        """
        Scores must preserve metadata from source chunk.
        
        SRP: Test metadata preservation.
        """
        scorer = RiskScorer()
        
        chunk = {
            "text": "Risk disclosure",
            "metadata": {"ticker": "AAPL", "filing_year": 2025, "item_type": "1A"}
        }
        
        score = scorer.calculate_severity(chunk)
        
        assert score.metadata == chunk["metadata"]


# ============================================================================
# Test Class 4: Failure Handling
# ============================================================================

class TestFailureHandling:
    """
    Verify that scoring failures are handled gracefully with clear error messages.
    """
    
    def test_invalid_chunk_format_raises_scoring_error(self) -> None:
        """
        Malformed chunk dictionaries should raise ScoringError.
        
        SRP: Test input validation failure.
        """
        scorer = RiskScorer()
        
        # Not a dictionary
        with pytest.raises(ScoringError, match="dictionary"):
            scorer.calculate_severity("not a dict")  # type: ignore
    
    def test_missing_text_field_raises_scoring_error(self) -> None:
        """
        Chunks without 'text' field should raise ScoringError.
        
        SRP: Test required field validation.
        """
        scorer = RiskScorer()
        
        invalid_chunk = {
            # Missing 'text'
            "metadata": {"ticker": "AAPL", "filing_year": 2025}
        }
        
        with pytest.raises(ScoringError, match="text"):
            scorer.calculate_severity(invalid_chunk)
    
    def test_embedding_failure_documented_in_error(self) -> None:
        """
        If embedding generation fails, error should explain the failure.
        
        SRP: Test embedding failure handling.
        """
        scorer = RiskScorer()
        
        # Chunk with text that might cause embedding issues (empty after processing)
        chunk = {
            "text": "",  # Empty text
            "metadata": {"ticker": "AAPL", "filing_year": 2025}
        }
        
        with pytest.raises(ScoringError, match="empty|text"):
            scorer.calculate_severity(chunk)
    
    def test_scoring_error_includes_helpful_context(self) -> None:
        """
        ScoringError should include context about what went wrong.
        
        SRP: Test error message quality.
        """
        scorer = RiskScorer()
        
        invalid_chunk = {"wrong_key": "value"}
        
        with pytest.raises(ScoringError) as exc_info:
            scorer.calculate_severity(invalid_chunk)
        
        error_msg = str(exc_info.value).lower()
        # Error should mention what was expected
        assert any(keyword in error_msg for keyword in ["text", "metadata", "required"])


# ============================================================================
# Test Class 5: Type Safety
# ============================================================================

class TestTypeSafety:
    """
    Verify that all scoring functions have proper type annotations.
    """
    
    def test_risk_score_dataclass_has_all_required_fields(self) -> None:
        """
        RiskScore dataclass must include all required fields with correct types.
        
        SRP: Test dataclass structure.
        """
        score = RiskScore(
            value=0.75,
            source_citation="Risk disclosure text",
            explanation="Calculated based on severity keywords",
            metadata={"ticker": "AAPL", "filing_year": 2025}
        )
        
        assert isinstance(score.value, float)
        assert isinstance(score.source_citation, str)
        assert isinstance(score.explanation, str)
        assert isinstance(score.metadata, dict)
    
    def test_scorer_methods_return_risk_score_type(self) -> None:
        """
        All scoring methods must return RiskScore instances.
        
        SRP: Test return type consistency.
        """
        scorer = RiskScorer()
        
        chunk = {
            "text": "Risk disclosure",
            "metadata": {"ticker": "AAPL", "filing_year": 2025}
        }
        
        severity_score = scorer.calculate_severity(chunk)
        novelty_score = scorer.calculate_novelty(chunk, [])
        
        assert isinstance(severity_score, RiskScore)
        assert isinstance(novelty_score, RiskScore)
    
    def test_type_annotations_pass_mypy(self) -> None:
        """
        The scoring module must pass mypy --strict with no errors.
        
        SRP: Test static type checking.
        
        Note: This test documents the requirement. Actual mypy checking
        happens in CI/pre-commit hooks.
        """
        # This is a documentation test - mypy is run separately
        # But we verify runtime types are consistent
        scorer = RiskScorer()
        
        chunk = {
            "text": "Risk disclosure",
            "metadata": {"ticker": "AAPL", "filing_year": 2025}
        }
        
        score = scorer.calculate_severity(chunk)
        
        # Runtime type verification
        assert hasattr(score, 'value')
        assert hasattr(score, 'source_citation')
        assert hasattr(score, 'explanation')
        assert hasattr(score, 'metadata')


# ============================================================================
# Test Class 6: Integration with Existing Pipeline
# ============================================================================

class TestPipelineIntegration:
    """
    Verify that scoring integrates cleanly with existing IndexingPipeline.
    """
    
    def test_scorer_accepts_pipeline_query_results(self) -> None:
        """
        RiskScorer should accept output from IndexingPipeline.semantic_search().
        
        SRP: Test interface compatibility.
        """
        scorer = RiskScorer()
        
        # Simulate output from IndexingPipeline.semantic_search()
        search_result = {
            "id": "AAPL_2025_0",
            "text": "Supply chain vulnerabilities due to geopolitical tensions",
            "metadata": {
                "ticker": "AAPL",
                "filing_year": 2025,
                "item_type": "1A"
            },
            "distance": 0.234
        }
        
        # Should accept this format without errors
        score = scorer.calculate_severity(search_result)
        
        assert 0.0 <= score.value <= 1.0
        assert score.metadata["ticker"] == "AAPL"
    
    def test_batch_scoring_for_multiple_chunks(self) -> None:
        """
        Scorer should efficiently handle batch scoring of multiple chunks.
        
        SRP: Test batch processing capability.
        """
        scorer = RiskScorer()
        
        chunks = [
            {
                "text": f"Risk disclosure {i}",
                "metadata": {"ticker": "AAPL", "filing_year": 2025}
            }
            for i in range(5)
        ]
        
        scores = scorer.calculate_severity_batch(chunks)
        
        assert len(scores) == 5
        assert all(isinstance(s, RiskScore) for s in scores)
        assert all(0.0 <= s.value <= 1.0 for s in scores)
