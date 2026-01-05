"""
FastAPI REST API wrapper for SEC risk scoring engine (Issue #24).

This module provides RESTful endpoints for analyzing SEC filings and
computing severity/novelty scores with full type safety via Pydantic.

Endpoints:
- POST /analyze: Analyze a SEC filing (via HTML content or file path)
- GET /health: Health check endpoint
- GET /: API documentation redirect

Usage:
    # Start server
    uvicorn sec_risk_api.api:app --reload
    
    # Or with uv
    uv run uvicorn sec_risk_api.api:app --reload
    
    # Test endpoint
    curl -X POST "http://localhost:8000/analyze" \\
         -H "Content-Type: application/json" \\
         -d '{
           "ticker": "AAPL",
           "filing_year": 2025,
           "html_content": "<html>...</html>"
         }'
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
import tempfile
from pathlib import Path
import logging
import re

from sec_risk_api.integration import IntegrationPipeline, IntegrationError, RiskAnalysisResult

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API versioning
API_VERSION = "1.0.0"

# ============================================================================
# Pydantic Models (Request/Response Schemas)
# ============================================================================

class RiskRequest(BaseModel):
    """
    Request model for risk analysis endpoint.
    
    Either html_content or html_path must be provided (mutually exclusive).
    
    Attributes:
        ticker: Stock symbol (e.g., "AAPL", "BRK.B")
        filing_year: Filing year (1990-2030)
        html_content: Raw HTML content (optional)
        html_path: Path to HTML file (optional)
        retrieve_top_k: Number of top risks to analyze (default: 10)
    
    Examples:
        >>> # With HTML content
        >>> request = RiskRequest(
        ...     ticker="AAPL",
        ...     filing_year=2025,
        ...     html_content="<html>...</html>"
        ... )
        
        >>> # With file path
        >>> request = RiskRequest(
        ...     ticker="AAPL",
        ...     filing_year=2025,
        ...     html_path="data/sample_10k.html"
        ... )
    """
    ticker: str = Field(
        ...,
        description="Stock ticker symbol (uppercase, may contain dots/hyphens)",
        examples=["AAPL", "BRK.B", "ABC-D"]
    )
    filing_year: int = Field(
        ...,
        description="Filing year",
        ge=1990,
        le=2030,
        examples=[2025]
    )
    html_content: Optional[str] = Field(
        None,
        description="Raw HTML content of the filing"
    )
    html_path: Optional[str] = Field(
        None,
        description="Path to HTML file (alternative to html_content)"
    )
    retrieve_top_k: int = Field(
        10,
        description="Number of top risk factors to analyze",
        ge=1,
        le=50
    )
    
    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v: str) -> str:
        """Validate ticker format (uppercase alphanumeric + dots/hyphens)."""
        if not v or not v.strip():
            raise ValueError("Ticker cannot be empty")
        
        # Allow uppercase letters, numbers, dots, and hyphens
        if not re.match(r'^[A-Z0-9.\-]+$', v):
            raise ValueError(
                f"Invalid ticker format: {v}. "
                "Must contain only uppercase letters, numbers, dots, and hyphens."
            )
        
        return v
    
    @field_validator("html_content")
    @classmethod
    def validate_html_content(cls, v: Optional[str], info: Any) -> Optional[str]:
        """Ensure either html_content or html_path is provided."""
        # This is called before html_path validator, so we check in model_validator
        return v
    
    def model_post_init(self, __context: Any) -> None:
        """Validate that exactly one of html_content or html_path is provided."""
        if self.html_content is None and self.html_path is None:
            raise ValueError("Either html_content or html_path must be provided")
        
        if self.html_content is not None and self.html_path is not None:
            raise ValueError("Cannot provide both html_content and html_path")


class ScoreInfo(BaseModel):
    """
    Score information with value and explanation.
    
    Attributes:
        value: Normalized score in [0.0, 1.0]
        explanation: Human-readable explanation of the score
    """
    value: float = Field(..., ge=0.0, le=1.0, description="Score value [0.0-1.0]")
    explanation: str = Field(..., description="Explanation of the score calculation")


class RiskEntry(BaseModel):
    """
    Individual risk factor with scores and citation.
    
    Attributes:
        text: Full text of the risk disclosure
        source_citation: Excerpt from source (truncated for readability)
        severity: Severity score and explanation
        novelty: Novelty score and explanation
        metadata: Original chunk metadata (ticker, year, etc.)
    """
    text: str = Field(..., description="Full risk disclosure text")
    source_citation: str = Field(..., description="Source text citation")
    severity: ScoreInfo = Field(..., description="Severity score and explanation")
    novelty: ScoreInfo = Field(..., description="Novelty score and explanation")
    metadata: Dict[str, Any] = Field(..., description="Original metadata")


class RiskResponse(BaseModel):
    """
    Response model for risk analysis endpoint.
    
    Attributes:
        ticker: Stock symbol
        filing_year: Filing year
        risks: List of analyzed risk factors
        metadata: Pipeline execution metadata (timing, counts, etc.)
    """
    ticker: str = Field(..., description="Stock ticker symbol")
    filing_year: int = Field(..., description="Filing year")
    risks: List[RiskEntry] = Field(..., description="List of analyzed risk factors")
    metadata: Dict[str, Any] = Field(..., description="Pipeline execution metadata")


class HealthResponse(BaseModel):
    """
    Response model for health check endpoint.
    
    Attributes:
        status: Health status ("healthy" or "unhealthy")
        version: API version
    """
    status: str = Field(..., description="Health status", examples=["healthy"])
    version: str = Field(..., description="API version", examples=["1.0.0"])


class ErrorResponse(BaseModel):
    """
    Error response model.
    
    Attributes:
        detail: Error message
    """
    detail: str = Field(..., description="Error message")


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="SEC Risk Scoring API",
    description=(
        "RESTful API for analyzing SEC filings and computing severity/novelty scores. "
        "Extracts Item 1A risk factors, computes semantic embeddings, and scores each risk "
        "on severity (0.0-1.0) and novelty (0.0-1.0) with full source citation."
    ),
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Initialize integration pipeline (singleton)
pipeline = IntegrationPipeline()

# ============================================================================
# Endpoints
# ============================================================================

@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    """Redirect root to API documentation."""
    return RedirectResponse(url="/docs")


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check endpoint"
)
async def health_check() -> HealthResponse:
    """
    Check API health status.
    
    Returns health status and version information.
    Useful for monitoring and load balancer health checks.
    
    Returns:
        HealthResponse with status and version
    """
    return HealthResponse(status="healthy", version=API_VERSION)


@app.post(
    "/analyze",
    response_model=RiskResponse,
    tags=["Risk Analysis"],
    summary="Analyze SEC filing",
    responses={
        200: {"description": "Successful analysis"},
        400: {"description": "Bad request (invalid HTML)"},
        404: {"description": "File not found"},
        422: {"description": "Validation error (invalid input)"},
        500: {"description": "Internal server error"}
    }
)
async def analyze_filing(request: RiskRequest) -> RiskResponse:
    """
    Analyze a SEC filing and compute risk scores.
    
    Pipeline:
    1. Validate inputs (ticker, year, HTML source)
    2. Extract and chunk Item 1A risk factors
    3. Generate embeddings and index into vector database
    4. Retrieve top-k risk factors via semantic search
    5. Compute severity and novelty scores
    6. Return structured results with citations
    
    Args:
        request: RiskRequest with ticker, year, and HTML source
    
    Returns:
        RiskResponse with analyzed risks and metadata
    
    Raises:
        HTTPException 400: Invalid HTML content
        HTTPException 404: File not found
        HTTPException 422: Validation error (Pydantic)
        HTTPException 500: Internal processing error
    
    Example Request:
        ```json
        {
          "ticker": "AAPL",
          "filing_year": 2025,
          "html_content": "<html>...</html>",
          "retrieve_top_k": 10
        }
        ```
    
    Example Response:
        ```json
        {
          "ticker": "AAPL",
          "filing_year": 2025,
          "risks": [
            {
              "text": "Supply chain disruptions...",
              "source_citation": "Supply chain disruptions...",
              "severity": {
                "value": 0.75,
                "explanation": "High severity due to..."
              },
              "novelty": {
                "value": 0.82,
                "explanation": "High novelty compared to..."
              },
              "metadata": {"ticker": "AAPL", "filing_year": 2025}
            }
          ],
          "metadata": {
            "total_latency_ms": 2534.5,
            "chunks_indexed": 5
          }
        }
        ```
    """
    try:
        # Determine HTML source
        if request.html_content is not None:
            # Use provided HTML content - write to temp file
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.html',
                delete=False,
                encoding='utf-8'
            ) as temp_file:
                temp_file.write(request.html_content)
                temp_path = temp_file.name
            
            html_path_to_use = temp_path
        else:
            # Use provided file path (validated to exist in model)
            if request.html_path is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Either html_content or html_path must be provided"
                )
            
            html_path_to_use = request.html_path
            
            # Validate file exists
            if not Path(html_path_to_use).exists():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"HTML file not found at path: {html_path_to_use}"
                )
        
        # Run integration pipeline
        logger.info(f"Analyzing {request.ticker} {request.filing_year}")
        result: RiskAnalysisResult = pipeline.analyze_filing(
            html_path=html_path_to_use,
            ticker=request.ticker,
            filing_year=request.filing_year,
            retrieve_top_k=request.retrieve_top_k
        )
        
        # Clean up temp file if created
        if request.html_content is not None:
            try:
                Path(temp_path).unlink()
            except Exception:
                pass
        
        # Convert to API response format
        risk_entries = [
            RiskEntry(
                text=risk["text"],
                source_citation=risk["source_citation"],
                severity=ScoreInfo(
                    value=risk["severity"]["value"],
                    explanation=risk["severity"]["explanation"]
                ),
                novelty=ScoreInfo(
                    value=risk["novelty"]["value"],
                    explanation=risk["novelty"]["explanation"]
                ),
                metadata=risk["metadata"]
            )
            for risk in result.risks
        ]
        
        return RiskResponse(
            ticker=result.ticker,
            filing_year=result.filing_year,
            risks=risk_entries,
            metadata=result.metadata
        )
    
    except IntegrationError as e:
        # Handle integration pipeline errors
        error_msg = str(e).lower()
        
        if "file not found" in error_msg or "path" in error_msg:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(e)
            )
        elif "invalid" in error_msg or "empty" in error_msg:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(e)
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
    
    except HTTPException:
        # Re-raise HTTPExceptions (don't let generic catch-all mask these)
        raise
    
    except Exception as e:
        # Catch-all for unexpected errors
        logger.error(f"Unexpected error in analyze_filing: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event() -> None:
    """Initialize resources on startup."""
    logger.info(f"Starting SEC Risk Scoring API v{API_VERSION}")
    logger.info(f"Pipeline initialized at: {pipeline.persist_path}")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Clean up resources on shutdown."""
    logger.info("Shutting down SEC Risk Scoring API")
