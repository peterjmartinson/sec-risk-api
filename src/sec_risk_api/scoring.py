"""
Retrieval-augmented risk scoring logic for SEC filings (Issue #21).

This module computes Severity and Novelty scores for risk disclosures,
with full source citation and traceability.

Core Concepts:
- Severity: How severe is this risk? (0.0 = minor, 1.0 = catastrophic)
- Novelty: How new is this risk vs. historical filings? (0.0 = repetitive, 1.0 = novel)

Every score includes:
- Normalized value [0.0, 1.0]
- Source citation (exact text)
- Human-readable explanation
- Original metadata (ticker, year, etc.)
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import logging
import numpy as np
from numpy.typing import NDArray

from sec_risk_api.embeddings import EmbeddingEngine

logger = logging.getLogger(__name__)


# ============================================================================
# Custom Exceptions
# ============================================================================

class ScoringError(Exception):
    """
    Raised when risk scoring fails due to invalid input or processing errors.
    
    Examples:
    - Missing required fields (text, metadata)
    - Malformed chunk structure
    - Embedding generation failure
    """
    pass


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class RiskScore:
    """
    Container for a computed risk score with full provenance.
    
    Attributes:
        value: Normalized score in [0.0, 1.0]
        source_citation: Exact text from source chunk (truncated if >500 chars)
        explanation: Human-readable description of how score was calculated
        metadata: Original chunk metadata (ticker, filing_year, item_type, etc.)
    
    Example:
        >>> score = RiskScore(
        ...     value=0.85,
        ...     source_citation="Catastrophic supply chain disruptions...",
        ...     explanation="High severity due to catastrophic language",
        ...     metadata={"ticker": "AAPL", "filing_year": 2025}
        ... )
    """
    value: float
    source_citation: str
    explanation: str
    metadata: Dict[str, Any]
    
    def __post_init__(self) -> None:
        """Validate score is in [0.0, 1.0] range."""
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"Score value must be in [0.0, 1.0], got {self.value}")


# ============================================================================
# Severity Keywords
# ============================================================================

# Keywords indicating high severity risks (catastrophic/existential language)
SEVERE_KEYWORDS = [
    "catastrophic", "existential", "unprecedented", "severe", "critical",
    "devastating", "collapse", "failure", "crisis", "threat", "significant",
    "material", "substantial", "major", "adversely", "harm", "damage",
    "disrupt", "impair", "unable", "bankruptcy", "insolvency", "default"
]

# Keywords indicating moderate severity
MODERATE_KEYWORDS = [
    "challenge", "difficulty", "risk", "uncertain", "volatility",
    "fluctuation", "pressure", "competition", "impact", "affect",
    "change", "regulatory", "compliance", "litigation"
]


# ============================================================================
# Risk Scorer
# ============================================================================

class RiskScorer:
    """
    Computes Severity and Novelty scores for risk disclosures.
    
    Severity Scoring:
    - Analyzes keyword presence and semantic intensity
    - Higher scores for catastrophic/existential language
    - Range: [0.0, 1.0]
    
    Novelty Scoring:
    - Compares current chunk with historical embeddings
    - Measures semantic distance from past disclosures
    - Range: [0.0, 1.0], where 1.0 = maximally novel
    
    Usage:
        >>> scorer = RiskScorer()
        >>> chunk = {"text": "...", "metadata": {...}}
        >>> severity = scorer.calculate_severity(chunk)
        >>> novelty = scorer.calculate_novelty(chunk, historical_chunks)
    """
    
    def __init__(self, embeddings: Optional[EmbeddingEngine] = None) -> None:
        """
        Initialize risk scorer.
        
        Args:
            embeddings: Optional pre-initialized embedding engine.
                       If None, will create a new instance (lazy loading).
        """
        self._embeddings = embeddings
    
    @property
    def embeddings(self) -> EmbeddingEngine:
        """Lazy-load embedding engine."""
        if self._embeddings is None:
            self._embeddings = EmbeddingEngine()
        return self._embeddings
    
    # ========================================================================
    # Severity Scoring
    # ========================================================================
    
    def calculate_severity(self, chunk: Any) -> RiskScore:
        """
        Calculate severity score for a risk disclosure chunk.
        
        Algorithm:
        1. Validate input structure
        2. Extract text and convert to lowercase
        3. Count severe/moderate keyword matches
        4. Normalize to [0.0, 1.0] using weighted formula
        5. Generate explanation and citation
        
        Args:
            chunk: Dictionary with 'text' and 'metadata' fields
        
        Returns:
            RiskScore with severity value, citation, explanation, metadata
        
        Raises:
            ScoringError: If chunk is malformed or missing required fields
        
        Example:
            >>> chunk = {
            ...     "text": "Catastrophic supply chain failure",
            ...     "metadata": {"ticker": "AAPL", "filing_year": 2025}
            ... }
            >>> score = scorer.calculate_severity(chunk)
            >>> assert 0.7 <= score.value <= 1.0  # High severity
        """
        # Validate input
        self._validate_chunk(chunk)
        
        text = chunk["text"]
        metadata = chunk["metadata"]
        
        # Handle empty text edge case
        if not text or not text.strip():
            raise ScoringError("Cannot score empty text")
        
        # Convert to lowercase for keyword matching
        text_lower = text.lower()
        
        # Count keyword matches
        severe_matches = sum(1 for keyword in SEVERE_KEYWORDS if keyword in text_lower)
        moderate_matches = sum(1 for keyword in MODERATE_KEYWORDS if keyword in text_lower)
        
        # Calculate severity score using weighted formula
        # Severe keywords weighted 2x moderate keywords
        # Normalize by expected maximum (assume max 5 severe + 5 moderate keywords)
        raw_score = (severe_matches * 2.0 + moderate_matches * 1.0) / 15.0
        
        # Clamp to [0.0, 1.0]
        severity_value = min(1.0, max(0.0, raw_score))
        
        # Boost score if multiple severe keywords (indicates compound risk)
        if severe_matches >= 3:
            severity_value = min(1.0, severity_value * 1.2)
        
        # Generate explanation
        explanation = self._generate_severity_explanation(
            severity_value, severe_matches, moderate_matches
        )
        
        # Truncate citation for readability (max 500 chars)
        citation = text if len(text) <= 500 else text[:497] + "..."
        
        return RiskScore(
            value=severity_value,
            source_citation=citation,
            explanation=explanation,
            metadata=metadata
        )
    
    def calculate_severity_batch(self, chunks: List[Dict[str, Any]]) -> List[RiskScore]:
        """
        Calculate severity scores for multiple chunks efficiently.
        
        Args:
            chunks: List of chunk dictionaries
        
        Returns:
            List of RiskScore objects (same order as input)
        
        Example:
            >>> chunks = [{"text": "...", "metadata": {...}}, ...]
            >>> scores = scorer.calculate_severity_batch(chunks)
            >>> assert len(scores) == len(chunks)
        """
        return [self.calculate_severity(chunk) for chunk in chunks]
    
    def _generate_severity_explanation(
        self,
        severity: float,
        severe_count: int,
        moderate_count: int
    ) -> str:
        """Generate human-readable explanation for severity score."""
        if severity >= 0.8:
            return (
                f"High severity (score: {severity:.2f}). "
                f"Contains {severe_count} severe keywords and {moderate_count} moderate keywords. "
                f"Language indicates catastrophic or existential risk."
            )
        elif severity >= 0.5:
            return (
                f"Moderate severity (score: {severity:.2f}). "
                f"Contains {severe_count} severe keywords and {moderate_count} moderate keywords. "
                f"Language indicates significant business risk."
            )
        elif severity >= 0.2:
            return (
                f"Low-moderate severity (score: {severity:.2f}). "
                f"Contains {severe_count} severe keywords and {moderate_count} moderate keywords. "
                f"Language indicates manageable business risk."
            )
        else:
            return (
                f"Low severity (score: {severity:.2f}). "
                f"Contains {severe_count} severe keywords and {moderate_count} moderate keywords. "
                f"Language indicates routine business considerations."
            )
    
    # ========================================================================
    # Novelty Scoring
    # ========================================================================
    
    def calculate_novelty(
        self,
        chunk: Dict[str, Any],
        historical_chunks: List[Dict[str, Any]]
    ) -> RiskScore:
        """
        Calculate novelty score by comparing chunk with historical filings.
        
        Algorithm:
        1. Validate inputs
        2. Handle edge case: no historical data → max novelty (1.0)
        3. Generate embeddings for current and historical chunks
        4. Compute cosine similarities between current and each historical
        5. Novelty = 1 - max(similarities)  [most similar → least novel]
        6. Generate explanation and citation
        
        Args:
            chunk: Current risk disclosure chunk
            historical_chunks: List of historical chunks for comparison
        
        Returns:
            RiskScore with novelty value, citation, explanation, metadata
        
        Raises:
            ScoringError: If chunks are malformed
        
        Example:
            >>> current = {"text": "Quantum computing threats", "metadata": {...}}
            >>> historical = [{"text": "Standard competition risks", "metadata": {...}}]
            >>> score = scorer.calculate_novelty(current, historical)
            >>> assert score.value > 0.7  # Novel topic
        """
        # Validate inputs
        self._validate_chunk(chunk)
        for hist_chunk in historical_chunks:
            self._validate_chunk(hist_chunk)
        
        text = chunk["text"]
        metadata = chunk["metadata"]
        
        # Handle empty text
        if not text or not text.strip():
            raise ScoringError("Cannot score empty text")
        
        # Edge case: no historical data → maximally novel
        if not historical_chunks:
            explanation = (
                "Maximum novelty (score: 1.00). "
                "No historical data available for comparison. "
                "This risk disclosure has no precedent in prior filings."
            )
            citation = text if len(text) <= 500 else text[:497] + "..."
            return RiskScore(
                value=1.0,
                source_citation=citation,
                explanation=explanation,
                metadata=metadata
            )
        
        # Generate embeddings
        try:
            current_embedding = self.embeddings.encode([text])[0]
            historical_texts = [h["text"] for h in historical_chunks]
            historical_embeddings = self.embeddings.encode(historical_texts)
        except Exception as e:
            raise ScoringError(f"Embedding generation failed: {e}")
        
        # Compute cosine similarities with all historical chunks
        similarities = self._compute_cosine_similarities(
            current_embedding,
            historical_embeddings
        )
        
        # Novelty = 1 - max_similarity (most similar = least novel)
        max_similarity = float(np.max(similarities))
        novelty_value = 1.0 - max_similarity
        
        # Clamp to [0.0, 1.0]
        novelty_value = max(0.0, min(1.0, novelty_value))
        
        # Generate explanation
        explanation = self._generate_novelty_explanation(
            novelty_value,
            max_similarity,
            len(historical_chunks)
        )
        
        # Truncate citation
        citation = text if len(text) <= 500 else text[:497] + "..."
        
        return RiskScore(
            value=novelty_value,
            source_citation=citation,
            explanation=explanation,
            metadata=metadata
        )
    
    def _compute_cosine_similarities(
        self,
        current_embedding: NDArray[np.float32],
        historical_embeddings: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """
        Compute cosine similarities between current and historical embeddings.
        
        Args:
            current_embedding: 1D array (384 dimensions)
            historical_embeddings: 2D array (N x 384 dimensions)
        
        Returns:
            1D array of cosine similarities (N values)
        """
        # Normalize embeddings
        current_norm = current_embedding / np.linalg.norm(current_embedding)
        historical_norms = historical_embeddings / np.linalg.norm(
            historical_embeddings, axis=1, keepdims=True
        )
        
        # Compute dot products (cosine similarity for normalized vectors)
        similarities: NDArray[np.float32] = np.dot(historical_norms, current_norm)
        
        return similarities
    
    def _generate_novelty_explanation(
        self,
        novelty: float,
        max_similarity: float,
        historical_count: int
    ) -> str:
        """Generate human-readable explanation for novelty score."""
        if novelty >= 0.8:
            return (
                f"High novelty (score: {novelty:.2f}). "
                f"Semantically distant from {historical_count} historical chunks "
                f"(max similarity: {max_similarity:.2f}). "
                f"This risk represents a significant departure from prior disclosures."
            )
        elif novelty >= 0.5:
            return (
                f"Moderate novelty (score: {novelty:.2f}). "
                f"Partially distinct from {historical_count} historical chunks "
                f"(max similarity: {max_similarity:.2f}). "
                f"This risk has some novel aspects compared to prior filings."
            )
        elif novelty >= 0.2:
            return (
                f"Low novelty (score: {novelty:.2f}). "
                f"Similar to {historical_count} historical chunks "
                f"(max similarity: {max_similarity:.2f}). "
                f"This risk closely resembles prior year disclosures."
            )
        else:
            return (
                f"Minimal novelty (score: {novelty:.2f}). "
                f"Nearly identical to {historical_count} historical chunks "
                f"(max similarity: {max_similarity:.2f}). "
                f"This risk is repetitive boilerplate language."
            )
    
    # ========================================================================
    # Validation Helpers
    # ========================================================================
    
    def _validate_chunk(self, chunk: Any) -> None:
        """
        Validate chunk structure.
        
        Raises:
            ScoringError: If chunk is malformed
        """
        if not isinstance(chunk, dict):
            raise ScoringError(
                f"Chunk must be a dictionary, got {type(chunk).__name__}"
            )
        
        if "text" not in chunk:
            raise ScoringError(
                "Chunk missing required 'text' field. "
                f"Available keys: {list(chunk.keys())}"
            )
        
        if "metadata" not in chunk:
            raise ScoringError(
                "Chunk missing required 'metadata' field. "
                f"Available keys: {list(chunk.keys())}"
            )
        
        if not isinstance(chunk["metadata"], dict):
            raise ScoringError(
                f"Metadata must be a dictionary, got {type(chunk['metadata']).__name__}"
            )
