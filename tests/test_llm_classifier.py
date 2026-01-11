# Copyright (c) 2025 Peter Martinson, Distracted Fortune. All rights reserved.
# This software is proprietary and not licensed for use, modification, or distribution.

"""
Tests for LLM-based risk classification using Gemini 2.5 Flash Lite.

This module tests the integration of Gemini LLM for risk categorization
when vector search returns low confidence matches.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from sigmak.llm_classifier import (
    GeminiClassifier,
    LLMClassificationResult,
    LLMClassificationError,
    GeminiAPIError,
    GeminiRateLimitError,
)
from sigmak.risk_taxonomy import RiskCategory


class TestLLMClassificationResult:
    """Tests for LLMClassificationResult dataclass."""
    
    def test_result_creation_valid(self) -> None:
        """Test creating a valid LLM classification result."""
        result = LLMClassificationResult(
            category=RiskCategory.OPERATIONAL,
            confidence=0.95,
            evidence="Supply chain disruptions",
            rationale="Text discusses operational risks",
            model_version="gemini-2.5-flash",
            timestamp=datetime.now(),
            response_time_ms=150.5,
            input_tokens=100,
            output_tokens=50
        )
        
        assert result.category == RiskCategory.OPERATIONAL
        assert result.confidence == 0.95
        assert result.evidence == "Supply chain disruptions"
        assert result.model_version == "gemini-2.5-flash"
        assert result.input_tokens == 100
        assert result.output_tokens == 50
    
    def test_result_confidence_bounds(self) -> None:
        """Test that confidence must be in [0.0, 1.0] range."""
        # Valid bounds
        result = LLMClassificationResult(
            category=RiskCategory.OPERATIONAL,
            confidence=0.0,
            evidence="test",
            rationale="test",
            model_version="gemini-2.5-flash",
            timestamp=datetime.now(),
            response_time_ms=100.0,
            input_tokens=10,
            output_tokens=10
        )
        assert result.confidence == 0.0
        
        result = LLMClassificationResult(
            category=RiskCategory.OPERATIONAL,
            confidence=1.0,
            evidence="test",
            rationale="test",
            model_version="gemini-2.5-flash",
            timestamp=datetime.now(),
            response_time_ms=100.0,
            input_tokens=10,
            output_tokens=10
        )
        assert result.confidence == 1.0
        
        # Invalid bounds should raise ValueError in __post_init__
        with pytest.raises(ValueError):
            LLMClassificationResult(
                category=RiskCategory.OPERATIONAL,
                confidence=1.5,
                evidence="test",
                rationale="test",
                model_version="gemini-2.5-flash",
                timestamp=datetime.now(),
                response_time_ms=100.0,
                input_tokens=10,
                output_tokens=10
            )


class TestGeminiClassifier:
    """Tests for GeminiClassifier integration."""
    
    def test_classifier_initialization_without_api_key(self) -> None:
        """Test that classifier raises error if no API key provided."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
                GeminiClassifier()
    
    def test_classifier_initialization_with_api_key(self) -> None:
        """Test classifier initialization with API key."""
        with patch.dict('os.environ', {'GOOGLE_API_KEY': 'test-key'}):
            classifier = GeminiClassifier()
            assert classifier.model_name == "gemini-2.5-flash"
            assert classifier.max_retries == 3
    
    def test_classifier_custom_model(self) -> None:
        """Test classifier with custom model name."""
        with patch.dict('os.environ', {'GOOGLE_API_KEY': 'test-key'}):
            classifier = GeminiClassifier(model_name="gemini-pro")
            assert classifier.model_name == "gemini-pro"
    
    @patch('sigmak.llm_classifier.genai')
    def test_classify_success(self, mock_genai: Mock) -> None:
        """Test successful classification with valid LLM response."""
        # Mock the Gemini API response
        mock_response = Mock()
        mock_response.text = """
        {
            "category": "OPERATIONAL",
            "confidence": 0.95,
            "evidence": "Supply chain disruptions could severely impact operations",
            "rationale": "The text discusses operational risks related to supply chain management"
        }
        """
        mock_response.usage_metadata = Mock(
            prompt_token_count=100,
            candidates_token_count=50
        )
        
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        with patch.dict('os.environ', {'GOOGLE_API_KEY': 'test-key'}):
            classifier = GeminiClassifier()
            
            result = classifier.classify(
                text="Supply chain disruptions could severely impact our operations."
            )
            
            assert result.category == RiskCategory.OPERATIONAL
            assert result.confidence == 0.95
            assert "Supply chain" in result.evidence
            assert result.model_version == "gemini-2.5-flash"
            assert result.input_tokens == 100
            assert result.output_tokens == 50
    
    @patch('sigmak.llm_classifier.genai')
    def test_classify_invalid_json(self, mock_genai: Mock) -> None:
        """Test handling of invalid JSON in LLM response."""
        mock_response = Mock()
        mock_response.text = "This is not valid JSON"
        
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        with patch.dict('os.environ', {'GOOGLE_API_KEY': 'test-key'}):
            classifier = GeminiClassifier()
            
            with pytest.raises(LLMClassificationError, match="Failed to parse"):
                classifier.classify(text="Some risk text")
    
    @patch('sigmak.llm_classifier.genai')
    def test_classify_invalid_category(self, mock_genai: Mock) -> None:
        """Test handling of invalid category in LLM response."""
        mock_response = Mock()
        mock_response.text = """
        {
            "category": "INVALID_CATEGORY",
            "confidence": 0.95,
            "evidence": "test",
            "rationale": "test"
        }
        """
        
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        with patch.dict('os.environ', {'GOOGLE_API_KEY': 'test-key'}):
            classifier = GeminiClassifier()
            
            with pytest.raises(LLMClassificationError, match="Invalid category"):
                classifier.classify(text="Some risk text")
    
    @patch('sigmak.llm_classifier.genai')
    def test_classify_missing_required_field(self, mock_genai: Mock) -> None:
        """Test handling of missing required fields in LLM response."""
        mock_response = Mock()
        mock_response.text = """
        {
            "category": "OPERATIONAL",
            "confidence": 0.95
        }
        """
        
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        with patch.dict('os.environ', {'GOOGLE_API_KEY': 'test-key'}):
            classifier = GeminiClassifier()
            
            with pytest.raises(LLMClassificationError, match="Missing required field"):
                classifier.classify(text="Some risk text")
    
    @patch('sigmak.llm_classifier.genai')
    def test_classify_retry_on_rate_limit(self, mock_genai: Mock) -> None:
        """Test retry logic when rate limit is hit."""
        # First call fails with rate limit, second succeeds
        mock_response_fail = Mock()
        mock_response_fail.prompt_feedback.block_reason = "RATE_LIMIT"
        
        mock_response_success = Mock()
        mock_response_success.text = """
        {
            "category": "OPERATIONAL",
            "confidence": 0.95,
            "evidence": "test",
            "rationale": "test"
        }
        """
        mock_response_success.usage_metadata = Mock(
            prompt_token_count=100,
            candidates_token_count=50
        )
        
        mock_model = Mock()
        mock_model.generate_content.side_effect = [
            Exception("429 Resource exhausted"),
            mock_response_success
        ]
        mock_genai.GenerativeModel.return_value = mock_model
        
        with patch.dict('os.environ', {'GOOGLE_API_KEY': 'test-key'}):
            classifier = GeminiClassifier(max_retries=3, retry_delay=0.1)
            
            result = classifier.classify(text="Some risk text")
            assert result.category == RiskCategory.OPERATIONAL
            # Verify it was called twice (1 failure + 1 success)
            assert mock_model.generate_content.call_count == 2
    
    @patch('sigmak.llm_classifier.genai')
    def test_classify_max_retries_exceeded(self, mock_genai: Mock) -> None:
        """Test that classification fails after max retries."""
        mock_model = Mock()
        mock_model.generate_content.side_effect = Exception("429 Resource exhausted")
        mock_genai.GenerativeModel.return_value = mock_model
        
        with patch.dict('os.environ', {'GOOGLE_API_KEY': 'test-key'}):
            classifier = GeminiClassifier(max_retries=2, retry_delay=0.1)
            
            with pytest.raises(GeminiRateLimitError):
                classifier.classify(text="Some risk text")
    
    def test_classify_empty_text(self) -> None:
        """Test that classification fails on empty text."""
        with patch.dict('os.environ', {'GOOGLE_API_KEY': 'test-key'}):
            classifier = GeminiClassifier()
            
            with pytest.raises(ValueError, match="empty"):
                classifier.classify(text="")
            
            with pytest.raises(ValueError, match="empty"):
                classifier.classify(text="   ")
    
    @patch('sigmak.llm_classifier.genai')
    def test_prompt_includes_taxonomy(self, mock_genai: Mock) -> None:
        """Test that the prompt includes risk taxonomy information."""
        mock_response = Mock()
        mock_response.text = """
        {
            "category": "OPERATIONAL",
            "confidence": 0.95,
            "evidence": "test",
            "rationale": "test"
        }
        """
        mock_response.usage_metadata = Mock(
            prompt_token_count=100,
            candidates_token_count=50
        )
        
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        with patch.dict('os.environ', {'GOOGLE_API_KEY': 'test-key'}):
            classifier = GeminiClassifier()
            classifier.classify(text="Some risk text")
            
            # Check the prompt passed to generate_content
            call_args = mock_model.generate_content.call_args
            prompt = call_args[0][0]
            
            # Verify prompt includes key taxonomy categories
            assert "OPERATIONAL" in prompt
            assert "SYSTEMATIC" in prompt
            assert "GEOPOLITICAL" in prompt
            assert "json" in prompt.lower()


class TestLLMExceptions:
    """Tests for custom LLM exceptions."""
    
    def test_llm_classification_error(self) -> None:
        """Test LLMClassificationError can be raised."""
        with pytest.raises(LLMClassificationError, match="test error"):
            raise LLMClassificationError("test error")
    
    def test_gemini_api_error(self) -> None:
        """Test GeminiAPIError can be raised."""
        with pytest.raises(GeminiAPIError, match="API error"):
            raise GeminiAPIError("API error")
    
    def test_gemini_rate_limit_error(self) -> None:
        """Test GeminiRateLimitError can be raised."""
        with pytest.raises(GeminiRateLimitError, match="Rate limit"):
            raise GeminiRateLimitError("Rate limit")
