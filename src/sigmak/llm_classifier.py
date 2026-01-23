# Copyright (c) 2025 Peter Martinson, Distracted Fortune. All rights reserved.
# This software is proprietary and not licensed for use, modification, or distribution.

"""
LLM-based risk classification using Gemini 2.5 Flash Lite.

This module integrates Google's Gemini LLM for risk categorization when
vector search (ChromaDB) returns low confidence matches. The LLM provides
structured classification with evidence and rationale.

Design Principles:
- Fallback only when vector search is uncertain
- Log all LLM calls with full provenance
- Retry logic for rate limits and transient errors
- Strict type safety with full type annotations
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Any

from google import genai
from google.genai import types

from sigmak.prompt_manager import PromptManager
from sigmak.risk_taxonomy import RiskCategory, validate_category

logger = logging.getLogger(__name__)


# ============================================================================
# Custom Exceptions
# ============================================================================


class LLMClassificationError(Exception):
    """Base exception for LLM classification errors."""
    pass


class GeminiAPIError(LLMClassificationError):
    """Raised when Gemini API returns an error."""
    pass


class GeminiRateLimitError(GeminiAPIError):
    """Raised when rate limit is exceeded."""
    pass


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class LLMClassificationResult:
    """
    Result from LLM-based risk classification.
    
    Attributes:
        category: The classified risk category
        confidence: LLM's confidence score [0.0, 1.0]
        evidence: Quoted text from source that justifies classification
        rationale: LLM's explanation for the classification
        model_version: Gemini model version used
        prompt_version: Version/identifier of the prompt template used
        timestamp: When classification was performed
        response_time_ms: API response time in milliseconds
        input_tokens: Number of input tokens consumed
        output_tokens: Number of output tokens generated
    """
    category: RiskCategory
    confidence: float
    evidence: str
    rationale: str
    model_version: str
    prompt_version: str
    timestamp: datetime
    response_time_ms: float
    input_tokens: int
    output_tokens: int
    
    def __post_init__(self) -> None:
        """Validate confidence is in valid range."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"Confidence must be in [0.0, 1.0], got {self.confidence}"
            )


# ============================================================================
# Gemini Classifier
# ============================================================================


class GeminiClassifier:
    """
    Classifies risk paragraphs using Google's Gemini 2.5 Flash Lite LLM.
    
    This classifier is used as a fallback when vector search cannot
    confidently classify a risk paragraph (similarity score < threshold).
    
    Usage:
        >>> classifier = GeminiClassifier()
        >>> result = classifier.classify(
        ...     text="Supply chain disruptions due to geopolitical tensions..."
        ... )
        >>> print(f"Category: {result.category}, Confidence: {result.confidence}")
    """
    
    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        api_key: Optional[str] = None
    ) -> None:
        """
        Initialize Gemini classifier.
        
        Args:
            model_name: Gemini model to use (default: gemini-2.5-flash)
            max_retries: Maximum retry attempts for rate limits
            retry_delay: Initial delay between retries (exponential backoff)
            api_key: Google API key (defaults to GOOGLE_API_KEY env var)
        
        Raises:
            ValueError: If no API key is provided
        """
        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError(
                "Google API key required. Set GOOGLE_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Initialize Gemini client
        self.client = genai.Client(api_key=self.api_key)
        
        # Load prompt template
        self.prompt_manager = PromptManager()
        
        logger.info(f"GeminiClassifier initialized: model={model_name}")
    
    def classify(self, text: str) -> LLMClassificationResult:
        """
        Classify a risk paragraph into a risk category.
        
        Args:
            text: Risk paragraph text to classify
        
        Returns:
            LLMClassificationResult with category, confidence, and provenance
        
        Raises:
            ValueError: If text is empty
            LLMClassificationError: If classification fails
            GeminiRateLimitError: If rate limit is exceeded after retries
        """
        # Validate input
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Load prompt template and get version
        prompt_version = str(self.prompt_manager.get_latest_version("risk_classification"))
        prompt_template = self.prompt_manager.load_latest("risk_classification")
        
        # Construct full prompt
        full_prompt = f"{prompt_template}\n\n## Risk Text to Classify\n\n{text}"
        
        # Call Gemini API with retry logic
        start_time = time.time()
        response = self._call_gemini_with_retry(full_prompt)
        response_time_ms = (time.time() - start_time) * 1000
        
        # Parse response
        result = self._parse_response(response, response_time_ms, prompt_version)
        
        logger.info(
            f"LLM classification complete: category={result.category.value}, "
            f"confidence={result.confidence:.2f}, "
            f"response_time={response_time_ms:.1f}ms, "
            f"tokens={result.input_tokens}+{result.output_tokens}"
        )
        
        return result
    
    def _call_gemini_with_retry(self, prompt: str) -> Any:
        """
        Call Gemini API with exponential backoff retry logic.
        
        Args:
            prompt: Full prompt to send to LLM
        
        Returns:
            Gemini API response
        
        Raises:
            GeminiRateLimitError: If rate limit exceeded after max retries
            GeminiAPIError: If API returns non-retryable error
        """
        for attempt in range(self.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt
                )
                return response
            
            except Exception as e:
                error_msg = str(e)
                
                # Check if rate limit error
                if "429" in error_msg or "Resource exhausted" in error_msg:
                    if attempt < self.max_retries - 1:
                        delay = self.retry_delay * (2 ** attempt)
                        logger.warning(
                            f"Rate limit hit (attempt {attempt + 1}/{self.max_retries}). "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                        continue
                    else:
                        raise GeminiRateLimitError(
                            f"Rate limit exceeded after {self.max_retries} attempts"
                        )
                
                # Non-retryable error
                raise GeminiAPIError(f"Gemini API error: {error_msg}")
        
        # Should never reach here
        raise GeminiAPIError("Failed to call Gemini API")
    
    def _parse_response(
        self,
        response: Any,
        response_time_ms: float,
        prompt_version: str
    ) -> LLMClassificationResult:
        """
        Parse Gemini API response into structured result.
        
        Args:
            response: Raw Gemini API response
            response_time_ms: API response time in milliseconds
            prompt_version: Version of prompt template used
        
        Returns:
            Structured LLMClassificationResult
        
        Raises:
            LLMClassificationError: If response is invalid or missing fields
        """
        # Extract JSON from response
        try:
            response_text = response.text.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            response_text = response_text.strip()
            
            data = json.loads(response_text)
        
        except json.JSONDecodeError as e:
            raise LLMClassificationError(
                f"Failed to parse LLM response as JSON: {e}\n"
                f"Response text: {response.text[:200]}..."
            )
        
        # Validate required fields
        required_fields = ["category", "confidence", "evidence", "rationale"]
        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            raise LLMClassificationError(
                f"Missing required fields in LLM response: {missing_fields}"
            )
        
        # Validate category
        try:
            category = validate_category(data["category"])
        except ValueError as e:
            raise LLMClassificationError(f"Invalid category in LLM response: {e}")
        
        # Validate confidence
        confidence = float(data["confidence"])
        if not 0.0 <= confidence <= 1.0:
            raise LLMClassificationError(
                f"Confidence must be in [0.0, 1.0], got {confidence}"
            )
        
        # Extract token usage
        input_tokens = 0
        output_tokens = 0
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0)
            output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0)
        
        # Create result
        return LLMClassificationResult(
            category=category,
            confidence=confidence,
            evidence=data["evidence"],
            rationale=data["rationale"],
            model_version=self.model_name,
            prompt_version=prompt_version,
            timestamp=datetime.now(),
            response_time_ms=response_time_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )
