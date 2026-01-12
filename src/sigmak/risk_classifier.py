# Copyright (c) 2025 Peter Martinson, Distracted Fortune. All rights reserved.
# This software is proprietary and not licensed for use, modification, or distribution.

"""
Risk classification with LLM fallback integration.

This module provides threshold-based risk categorization that routes to Gemini LLM
when vector search cannot confidently classify a risk paragraph.

Design Principles:
- Use vector search (ChromaDB) as primary classification method
- Fall back to LLM when similarity scores are below high threshold
- Use LLM for confirmation when scores are uncertain
- Cache all LLM responses for future lookups
- Track full provenance for audit trail
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional, List

from sigmak.embeddings import EmbeddingEngine
from sigmak.indexing_pipeline import IndexingPipeline
from sigmak.llm_classifier import GeminiClassifier, LLMClassificationResult
from sigmak.llm_storage import LLMStorage, LLMStorageRecord
from sigmak.risk_taxonomy import RiskCategory, validate_category

logger = logging.getLogger(__name__)


# ============================================================================
# Thresholds
# ============================================================================

# High confidence threshold - use vector search result directly
HIGH_THRESHOLD = 0.80

# Low confidence threshold - fall back to LLM
LOW_THRESHOLD = 0.64


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class RiskClassificationResult:
    """
    Result of risk classification with provenance.
    
    Attributes:
        text: The risk paragraph text
        category: Classified risk category
        confidence: Confidence score [0.0, 1.0]
        method: Classification method used ("vector_search" or "llm")
        similarity_score: Vector similarity score (1 - distance)
        llm_result: LLM classification result (if LLM was used)
        cached: Whether result was retrieved from cache
        timestamp: When classification was performed
    """
    text: str
    category: RiskCategory
    confidence: float
    method: str  # "vector_search" or "llm"
    similarity_score: float
    llm_result: Optional[LLMClassificationResult]
    cached: bool
    timestamp: datetime


# ============================================================================
# Risk Classifier with LLM Fallback
# ============================================================================


class RiskClassifierWithLLM:
    """
    Risk classifier that combines vector search with LLM fallback.
    
    Classification Strategy:
    1. Check cache for previous LLM classification
    2. Perform vector search in ChromaDB
    3. If similarity >= HIGH_THRESHOLD: use vector result
    4. If similarity < LOW_THRESHOLD: fall back to LLM
    5. If LOW_THRESHOLD <= similarity < HIGH_THRESHOLD: use LLM for confirmation
    6. Store LLM results in cache for future lookups
    
    Usage:
        >>> classifier = RiskClassifierWithLLM()
        >>> result = classifier.classify("Supply chain disruptions...")
        >>> print(f"Category: {result.category}, Method: {result.method}")
    """
    
    def __init__(
        self,
        indexing_pipeline: Optional[IndexingPipeline] = None,
        llm_classifier: Optional[GeminiClassifier] = None,
        llm_storage: Optional[LLMStorage] = None,
        embedding_engine: Optional[EmbeddingEngine] = None,
        high_threshold: float = HIGH_THRESHOLD,
        low_threshold: float = LOW_THRESHOLD
    ) -> None:
        """
        Initialize risk classifier with LLM fallback.
        
        Args:
            indexing_pipeline: Pipeline for vector search (will create if None)
            llm_classifier: Gemini classifier (will create if None)
            llm_storage: SQLite storage for LLM results (will create if None)
            embedding_engine: Embedding engine (will create if None)
            high_threshold: Similarity threshold for direct use (default: 0.80)
            low_threshold: Similarity threshold for LLM fallback (default: 0.64)
        """
        self.pipeline = indexing_pipeline or IndexingPipeline()
        self.llm_classifier = llm_classifier or GeminiClassifier()
        self.llm_storage = llm_storage or LLMStorage()
        self.embedding_engine = embedding_engine or EmbeddingEngine()
        
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        
        logger.info(
            f"RiskClassifierWithLLM initialized: "
            f"high_threshold={high_threshold}, low_threshold={low_threshold}"
        )
    
    def classify(
        self,
        text: str,
        use_cache: bool = True
    ) -> RiskClassificationResult:
        """
        Classify a risk paragraph with LLM fallback.
        
        Args:
            text: Risk paragraph text to classify
            use_cache: Whether to check cache for previous LLM classifications
        
        Returns:
            RiskClassificationResult with category and provenance
        """
        timestamp = datetime.now()
        
        # Step 1: Check cache for previous LLM classification
        if use_cache:
            cached_results = self.llm_storage.query_by_text(text)
            if cached_results:
                # Use most recent cached result
                cached = cached_results[0]
                logger.info(f"Found cached LLM result for text (hash: {text[:50]}...)")
                
                # Convert similarity from distance (lower = better) to score (higher = better)
                # For cached results, we use confidence as a proxy for similarity
                similarity_score = cached.confidence
                
                return RiskClassificationResult(
                    text=text,
                    category=cached.category,
                    confidence=cached.confidence,
                    method="llm",
                    similarity_score=similarity_score,
                    llm_result=None,  # Already cached
                    cached=True,
                    timestamp=timestamp
                )
        
        # Step 2: Perform vector search
        search_results = self.pipeline.semantic_search(
            query=text,
            n_results=1  # Only need top result
        )
        
        if not search_results:
            # No vector results - must use LLM
            logger.info("No vector search results - using LLM")
            return self._classify_with_llm(text, 0.0, timestamp)
        
        top_result = search_results[0]
        distance = top_result["distance"]
        
        # Convert distance to similarity score
        # ChromaDB cosine distance range: [0, 2]
        # Lower distance = more similar
        # Convert to similarity: [1.0 (identical), 0.0 (opposite)]
        similarity_score = 1.0 - (distance / 2.0)
        
        logger.info(
            f"Vector search result: distance={distance:.4f}, "
            f"similarity={similarity_score:.4f}"
        )
        
        # Step 3: Apply threshold logic
        if similarity_score >= self.high_threshold:
            # High confidence - use vector search result
            logger.info(f"High confidence ({similarity_score:.2f} >= {self.high_threshold})")
            category = self._extract_category_from_metadata(top_result["metadata"])
            
            return RiskClassificationResult(
                text=text,
                category=category,
                confidence=similarity_score,
                method="vector_search",
                similarity_score=similarity_score,
                llm_result=None,
                cached=False,
                timestamp=timestamp
            )
        
        elif similarity_score < self.low_threshold:
            # Low confidence - fall back to LLM
            logger.info(
                f"Low confidence ({similarity_score:.2f} < {self.low_threshold}) - "
                f"falling back to LLM"
            )
            return self._classify_with_llm(text, similarity_score, timestamp)
        
        else:
            # Uncertain - use LLM for confirmation
            logger.info(
                f"Uncertain confidence ({self.low_threshold} <= {similarity_score:.2f} < "
                f"{self.high_threshold}) - using LLM for confirmation"
            )
            return self._classify_with_llm(text, similarity_score, timestamp)
    
    def _classify_with_llm(
        self,
        text: str,
        vector_similarity: float,
        timestamp: datetime
    ) -> RiskClassificationResult:
        """
        Classify using Gemini LLM and store result.
        
        Args:
            text: Risk paragraph text
            vector_similarity: Similarity score from vector search
            timestamp: Classification timestamp
        
        Returns:
            RiskClassificationResult
        """
        # Call LLM
        llm_result = self.llm_classifier.classify(text)
        
        # Generate embedding for storage
        embedding_result = self.embedding_engine.encode([text])[0]
        # Handle both numpy arrays and lists (for testing)
        embedding = embedding_result.tolist() if hasattr(embedding_result, 'tolist') else embedding_result
        
        # Store in cache
        storage_record = LLMStorageRecord(
            text=text,
            embedding=embedding,
            category=llm_result.category,
            confidence=llm_result.confidence,
            evidence=llm_result.evidence,
            rationale=llm_result.rationale,
            model_version=llm_result.model_version,
            timestamp=llm_result.timestamp,
            response_time_ms=llm_result.response_time_ms,
            input_tokens=llm_result.input_tokens,
            output_tokens=llm_result.output_tokens
        )
        
        self.llm_storage.insert(storage_record)
        
        logger.info(
            f"LLM classification stored: category={llm_result.category.value}, "
            f"confidence={llm_result.confidence:.2f}"
        )
        
        return RiskClassificationResult(
            text=text,
            category=llm_result.category,
            confidence=llm_result.confidence,
            method="llm",
            similarity_score=vector_similarity,
            llm_result=llm_result,
            cached=False,
            timestamp=timestamp
        )
    
    def _extract_category_from_metadata(self, metadata: Dict[str, Any]) -> RiskCategory:
        """
        Extract risk category from search result metadata.
        
        This is a placeholder - in the current implementation, metadata contains
        ticker/filing_year/item_type but not category. For now, we default to OTHER.
        
        Args:
            metadata: Search result metadata
        
        Returns:
            RiskCategory enum
        """
        # TODO: If metadata contains category, extract it
        # For now, default to OTHER as vector search doesn't store categories
        if "category" in metadata:
            return validate_category(metadata["category"])
        
        logger.warning(
            "Vector search result does not contain category - defaulting to OTHER"
        )
        return RiskCategory.OTHER
    
    def classify_batch(
        self,
        texts: List[str],
        use_cache: bool = True
    ) -> List[RiskClassificationResult]:
        """
        Classify multiple risk paragraphs.
        
        Args:
            texts: List of risk paragraph texts
            use_cache: Whether to use cache for lookups
        
        Returns:
            List of RiskClassificationResult objects
        """
        return [self.classify(text, use_cache=use_cache) for text in texts]
