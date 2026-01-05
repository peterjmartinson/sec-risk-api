"""
Unit tests for FastAPI REST API wrapper (Issue #24).

This module tests the RESTful API endpoints for risk scoring.
All endpoints must be type-safe with Pydantic validation.

Test Coverage:
- OpenAPI schema validation
- Request validation (bad inputs, missing fields)
- Response structure (mock and real data)
- Type safety (mypy compliance)
- Error handling (404, 422, 500)
"""

import pytest
from fastapi.testclient import TestClient
from typing import Dict, Any, List
from pathlib import Path
import json

from sec_risk_api.api import app


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def client() -> TestClient:
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def sample_html_path() -> Path:
    """Path to sample 10-K HTML file."""
    return Path("data/sample_10k.html")


# ============================================================================
# Test Class 1: OpenAPI Schema Validation
# ============================================================================

class TestOpenAPISchema:
    """
    Verify that OpenAPI schema is correctly generated and matches models.
    """
    
    def test_openapi_schema_exists(self, client: TestClient) -> None:
        """
        API should expose OpenAPI schema at /openapi.json.
        
        SRP: Test schema endpoint availability.
        """
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema
    
    def test_openapi_schema_has_analyze_endpoint(self, client: TestClient) -> None:
        """
        Schema should document /analyze endpoint.
        
        SRP: Test endpoint documentation.
        """
        response = client.get("/openapi.json")
        schema = response.json()
        
        assert "/analyze" in schema["paths"]
        assert "post" in schema["paths"]["/analyze"]
    
    def test_openapi_schema_has_health_endpoint(self, client: TestClient) -> None:
        """
        Schema should document /health endpoint.
        
        SRP: Test health check documentation.
        """
        response = client.get("/openapi.json")
        schema = response.json()
        
        assert "/health" in schema["paths"]
        assert "get" in schema["paths"]["/health"]
    
    def test_openapi_schema_defines_risk_request_model(self, client: TestClient) -> None:
        """
        Schema should define RiskRequest model.
        
        SRP: Test request model documentation.
        """
        response = client.get("/openapi.json")
        schema = response.json()
        
        # Check components/schemas for RiskRequest
        assert "components" in schema
        assert "schemas" in schema["components"]
        assert "RiskRequest" in schema["components"]["schemas"]
        
        # Verify required fields
        request_schema = schema["components"]["schemas"]["RiskRequest"]
        assert "required" in request_schema
        assert "ticker" in request_schema["required"]
        assert "filing_year" in request_schema["required"]


# ============================================================================
# Test Class 2: Request Validation
# ============================================================================

class TestRequestValidation:
    """
    Verify that API validates input and returns 422 for bad requests.
    """
    
    def test_missing_required_field_returns_422(self, client: TestClient) -> None:
        """
        Request missing required field should return 422 Unprocessable Entity.
        
        SRP: Test required field validation.
        """
        # Missing filing_year
        response = client.post("/analyze", json={
            "ticker": "AAPL"
            # Missing filing_year
        })
        
        assert response.status_code == 422
        error = response.json()
        assert "detail" in error
    
    def test_invalid_ticker_format_returns_422(self, client: TestClient) -> None:
        """
        Invalid ticker format should return 422.
        
        SRP: Test ticker validation.
        """
        response = client.post("/analyze", json={
            "ticker": "invalid ticker",  # Lowercase with space
            "filing_year": 2025,
            "html_content": "<html>test</html>"
        })
        
        assert response.status_code == 422
        error = response.json()
        assert "detail" in error
    
    def test_invalid_year_returns_422(self, client: TestClient) -> None:
        """
        Year outside valid range should return 422.
        
        SRP: Test year validation.
        """
        response = client.post("/analyze", json={
            "ticker": "AAPL",
            "filing_year": 1800,  # Too old
            "html_content": "<html>test</html>"
        })
        
        assert response.status_code == 422
    
    def test_empty_ticker_returns_422(self, client: TestClient) -> None:
        """
        Empty ticker should return 422.
        
        SRP: Test empty string validation.
        """
        response = client.post("/analyze", json={
            "ticker": "",
            "filing_year": 2025,
            "html_content": "<html>test</html>"
        })
        
        assert response.status_code == 422
    
    def test_missing_html_content_and_path_returns_422(self, client: TestClient) -> None:
        """
        Request without html_content OR html_path should return 422.
        
        SRP: Test mutually exclusive field validation.
        """
        response = client.post("/analyze", json={
            "ticker": "AAPL",
            "filing_year": 2025
            # Missing both html_content and html_path
        })
        
        assert response.status_code == 422


# ============================================================================
# Test Class 3: Response Structure
# ============================================================================

class TestResponseStructure:
    """
    Verify that successful responses match the expected schema.
    """
    
    def test_successful_response_has_correct_structure(
        self,
        client: TestClient,
        sample_html_path: Path
    ) -> None:
        """
        Successful analysis should return properly structured response.
        
        SRP: Test response schema.
        """
        # Read with CP1252 encoding (SEC filings use this)
        html_content = sample_html_path.read_text(encoding='cp1252')
        
        response = client.post("/analyze", json={
            "ticker": "AAPL",
            "filing_year": 2025,
            "html_content": html_content
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Check top-level fields
        assert "ticker" in data
        assert "filing_year" in data
        assert "risks" in data
        assert "metadata" in data
        
        assert data["ticker"] == "AAPL"
        assert data["filing_year"] == 2025
        assert isinstance(data["risks"], list)
        assert len(data["risks"]) > 0
    
    def test_risk_entry_has_required_fields(
        self,
        client: TestClient,
        sample_html_path: Path
    ) -> None:
        """
        Each risk entry should have all required fields.
        
        SRP: Test risk schema.
        """
        html_content = sample_html_path.read_text(encoding='cp1252')
        
        response = client.post("/analyze", json={
            "ticker": "AAPL",
            "filing_year": 2025,
            "html_content": html_content
        })
        
        data = response.json()
        
        for risk in data["risks"]:
            assert "text" in risk
            assert "source_citation" in risk
            assert "severity" in risk
            assert "novelty" in risk
            assert "metadata" in risk
            
            # Check severity structure
            assert "value" in risk["severity"]
            assert "explanation" in risk["severity"]
            assert isinstance(risk["severity"]["value"], (int, float))
            assert 0.0 <= risk["severity"]["value"] <= 1.0
            
            # Check novelty structure
            assert "value" in risk["novelty"]
            assert "explanation" in risk["novelty"]
            assert isinstance(risk["novelty"]["value"], (int, float))
            assert 0.0 <= risk["novelty"]["value"] <= 1.0
    
    def test_metadata_includes_pipeline_info(
        self,
        client: TestClient,
        sample_html_path: Path
    ) -> None:
        """
        Metadata should include pipeline execution information.
        
        SRP: Test metadata schema.
        """
        html_content = sample_html_path.read_text(encoding='cp1252')
        
        response = client.post("/analyze", json={
            "ticker": "AAPL",
            "filing_year": 2025,
            "html_content": html_content
        })
        
        data = response.json()
        metadata = data["metadata"]
        
        # Should include timing/count info
        assert "total_latency_ms" in metadata or "chunks_indexed" in metadata


# ============================================================================
# Test Class 4: Error Handling
# ============================================================================

class TestErrorHandling:
    """
    Verify that API handles errors gracefully with proper status codes.
    """
    
    def test_invalid_html_handled_gracefully(self, client: TestClient) -> None:
        """
        Empty HTML should process with fallback (returns empty risks).
        
        SRP: Test HTML parsing robustness.
        """
        response = client.post("/analyze", json={
            "ticker": "AAPL",
            "filing_year": 2025,
            "html_content": ""  # Empty HTML
        })
        
        # Should succeed (BeautifulSoup/fallback handles it)
        assert response.status_code == 200
        data = response.json()
        # May have 0 risks if no content
        assert "risks" in data
    
    def test_missing_item_1a_handled_gracefully(self, client: TestClient) -> None:
        """
        HTML without Item 1A should return success with fallback.
        
        SRP: Test missing section handling.
        """
        html_without_1a = """
        <html>
            <body>
                <div>Some content but no Item 1A marker</div>
            </body>
        </html>
        """
        
        response = client.post("/analyze", json={
            "ticker": "TEST",
            "filing_year": 2025,
            "html_content": html_without_1a
        })
        
        # Should succeed with fallback (returns full text)
        assert response.status_code == 200
        data = response.json()
        assert len(data["risks"]) > 0
    
    def test_nonexistent_file_path_returns_404(self, client: TestClient) -> None:
        """
        Non-existent html_path should return 404 Not Found.
        
        SRP: Test file not found error.
        """
        response = client.post("/analyze", json={
            "ticker": "AAPL",
            "filing_year": 2025,
            "html_path": "/nonexistent/file.html"
        })
        
        # Should be 404 (file not found check in API)
        assert response.status_code == 404
        error = response.json()
        assert "detail" in error
        assert "not found" in error["detail"].lower()


# ============================================================================
# Test Class 5: Health Check
# ============================================================================

class TestHealthCheck:
    """
    Verify that health check endpoint works correctly.
    """
    
    def test_health_endpoint_returns_200(self, client: TestClient) -> None:
        """
        Health check should return 200 OK.
        
        SRP: Test health endpoint availability.
        """
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_health_endpoint_returns_status(self, client: TestClient) -> None:
        """
        Health check should return status information.
        
        SRP: Test health response structure.
        """
        response = client.get("/health")
        data = response.json()
        
        assert "status" in data
        assert data["status"] == "healthy"
    
    def test_health_endpoint_includes_version(self, client: TestClient) -> None:
        """
        Health check should include API version.
        
        SRP: Test version information.
        """
        response = client.get("/health")
        data = response.json()
        
        assert "version" in data
        assert isinstance(data["version"], str)


# ============================================================================
# Test Class 6: Type Safety
# ============================================================================

class TestTypeSafety:
    """
    Verify that all API models are properly typed.
    """
    
    def test_response_is_json_serializable(
        self,
        client: TestClient,
        sample_html_path: Path
    ) -> None:
        """
        Response should be valid JSON.
        
        SRP: Test JSON serialization.
        """
        html_content = sample_html_path.read_text(encoding='cp1252')
        
        response = client.post("/analyze", json={
            "ticker": "AAPL",
            "filing_year": 2025,
            "html_content": html_content
        })
        
        # Should be parseable JSON
        data = response.json()
        
        # Should be re-serializable
        json_str = json.dumps(data)
        assert isinstance(json_str, str)
        assert len(json_str) > 0
    
    def test_request_with_extra_fields_accepted(self, client: TestClient) -> None:
        """
        Pydantic should allow extra fields (or reject them based on config).
        
        SRP: Test extra field handling.
        """
        response = client.post("/analyze", json={
            "ticker": "AAPL",
            "filing_year": 2025,
            "html_content": "<html><body>ITEM 1A. Risk factors content</body></html>",
            "extra_field": "should be ignored or rejected"
        })
        
        # Either succeeds (ignores extra) or fails with 422 (forbids extra)
        assert response.status_code in [200, 422]


# ============================================================================
# Test Class 7: Optional Parameters
# ============================================================================

class TestOptionalParameters:
    """
    Verify that optional parameters work correctly.
    """
    
    def test_retrieve_top_k_parameter(
        self,
        client: TestClient,
        sample_html_path: Path
    ) -> None:
        """
        retrieve_top_k parameter should control number of results.
        
        SRP: Test optional parameter.
        """
        html_content = sample_html_path.read_text(encoding='cp1252')
        
        response = client.post("/analyze", json={
            "ticker": "AAPL",
            "filing_year": 2025,
            "html_content": html_content,
            "retrieve_top_k": 3
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Should return at most 3 risks
        assert len(data["risks"]) <= 3
    
    def test_default_retrieve_top_k(
        self,
        client: TestClient,
        sample_html_path: Path
    ) -> None:
        """
        Without retrieve_top_k, should use default (10).
        
        SRP: Test default parameter value.
        """
        html_content = sample_html_path.read_text(encoding='cp1252')
        
        response = client.post("/analyze", json={
            "ticker": "AAPL",
            "filing_year": 2025,
            "html_content": html_content
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Should return up to default (10) risks
        assert len(data["risks"]) <= 10
