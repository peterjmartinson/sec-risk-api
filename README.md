# Scream Sheet: Proprietary Risk Scoring API

[![CI](https://github.com/peterjmartinson/sec-risk-api/actions/workflows/ci.yml/badge.svg)](https://github.com/peterjmartinson/sec-risk-api/actions/workflows/ci.yml)

A RAG-powered pipeline designed to quantify novelty and severity in SEC **Item
1A: Risk Factors** disclosures. This system ingests structured HTML/XBRL
filings, extracts specific risk sections, and indexes them into a
high-dimensional vector space for semantic analysis.

## System Architecture

The application follows a modular "Extraction-to-Storage" flow:

1. **Ingestion Engine (`ingest.py`)**: Parses raw SEC HTM files using BeautifulSoup (lxml). It features robust encoding fallbacks (UTF-8/CP1252) and regex-based slicing to isolate "Item 1A."
2. **Processing Layer (`processing.py`)**: Implements recursive character text splitting. Chunks are normalized to ~800 characters with an 80-character overlap to preserve semantic context.
3. **Embedding Engine (`embeddings.py`)**: Converts text chunks into 384-dimensional semantic vectors using the `all-MiniLM-L6-v2` sentence transformer model.
4. **The Vault (`init_vector_db.py`)**: A persistent **Chroma DB** instance utilizing **HNSW** indexing with a **Cosine Similarity** metric. Storage path: `./chroma_db/`
5. **Reranking Layer (`reranking.py`)**: Cross-encoder model (`ms-marco-MiniLM-L-6-v2`) that reranks search results for improved relevance. Processes query-document pairs jointly for superior accuracy.
6. **Risk Classification (`risk_taxonomy.py`, `prompt_manager.py`)**: Proprietary 10-category risk taxonomy with version-controlled LLM prompts for semantic classification.
7. **Risk Scoring (`scoring.py`)**: Quantifies **Severity** (0.0-1.0) and **Novelty** (0.0-1.0) for risk disclosures using keyword analysis and semantic comparison with historical filings. Every score includes full source citation and human-readable explanation.
8. **Orchestration Layer (`indexing_pipeline.py`)**: End-to-end pipeline coordinator that integrates extraction, chunking, embedding, storage, and hybrid search with optional reranking.

### Risk Taxonomy

The system uses a proprietary 10-category classification for SEC risk factors:

| Category | Description | Examples |
|----------|-------------|----------|
| **OPERATIONAL** | Internal execution risks | Supply chain, manufacturing, IT infrastructure |
| **SYSTEMATIC** | Macroeconomic forces | Recession, inflation, market volatility |
| **GEOPOLITICAL** | International conflicts | War, trade disputes, sanctions |
| **REGULATORY** | Government compliance | New laws, regulations, policy changes |
| **COMPETITIVE** | Market rivalry | Competition, pricing pressure, new entrants |
| **TECHNOLOGICAL** | Innovation threats | Obsolescence, cybersecurity, disruption |
| **HUMAN_CAPITAL** | Workforce risks | Talent retention, labor disputes, skill gaps |
| **FINANCIAL** | Capital structure | Liquidity, debt, foreign exchange exposure |
| **REPUTATIONAL** | Brand and trust | PR crises, ESG controversies, trust erosion |
| **OTHER** | Miscellaneous | Company-specific edge cases |

**Prompt Engineering**: Version-controlled system prompts in `prompts/` directory enforce structured classification with source citation requirements.

### Storage & Retrieval

**Vector Database**: ChromaDB with SQLite backend
- **Storage Path**: `./chroma_db/` (configurable)
- **Collection**: `sec_risk_factors`
- **Distance Metric**: Cosine similarity (optimized for semantic search)
- **Index Type**: HNSW (Hierarchical Navigable Small World)

**Metadata Schema** (enforced on every chunk):
```python
{
    "ticker": str,        # Stock symbol (e.g., "AAPL")
    "filing_year": int,   # Filing year (e.g., 2025)
    "item_type": str      # SEC section (e.g., "Item 1A")
}
```

**Retrieval Logic**:
- **Semantic Search**: Natural language queries (e.g., "Geopolitical Instability") retrieve relevant chunks even without exact keyword matches.
- **Hybrid Filtering**: Combine semantic search with metadata filters (e.g., "Find AAPL risks from 2025 related to supply chain").
- **Two-Stage Reranking** (Optional): Retrieve broader candidates via bi-encoder, then rerank with cross-encoder for maximum precision.
- **Deterministic IDs**: Document IDs follow `{ticker}_{year}_{chunk_index}` pattern for safe upserts.

## Getting Started

### Prerequisites

* Python 3.10+
* `uv` for dependency management (preferred)
* WSL2 (for persistence on Windows)

### Installation

```bash
uv sync

```

### Initialization

Initialize the vector database infrastructure:

```bash
uv run init_vector_db.py
```

### Indexing a Filing

Index a 10-K HTML filing into the vector database:

```python
from sec_risk_api.indexing_pipeline import IndexingPipeline

# Initialize the pipeline
pipeline = IndexingPipeline(persist_path="./chroma_db")

# Index a filing
stats = pipeline.index_filing(
    html_path="data/sample_10k.html",
    ticker="AAPL",
    filing_year=2025,
    item_type="Item 1A"
)

print(f"Indexed {stats['chunks_indexed']} chunks in {stats['embedding_latency_ms']:.2f}ms")
```

### Semantic Search

Query the indexed filings using natural language:

```python
# Basic vector search (fast)
results = pipeline.semantic_search(
    query="Geopolitical risks and international conflicts",
    n_results=5
)

# Hybrid search with metadata filtering
results = pipeline.semantic_search(
    query="Supply chain disruptions",
    n_results=5,
    where={"ticker": "AAPL", "filing_year": 2025}
)

for result in results:
    print(f"[{result['metadata']['ticker']}] {result['text'][:100]}...")
    print(f"Distance: {result['distance']:.4f}\n")
```

### Reranked Search (Higher Precision)

For maximum relevance, enable cross-encoder reranking:

```python
# Two-stage search: vector retrieval → cross-encoder reranking
results = pipeline.semantic_search(
    query="regulatory compliance risks in technology sector",
    n_results=3,
    rerank=True  # Enable cross-encoder reranking
)

for result in results:
    print(f"[{result['metadata']['ticker']} {result['metadata']['filing_year']}]")
    print(f"Text: {result['text'][:150]}...")
    print(f"Vector Distance: {result['distance']:.4f}")
    print(f"Rerank Score: {result['rerank_score']:.4f}\n")

# Combine reranking with metadata filters
results = pipeline.semantic_search(
    query="market competition and pricing pressure",
    n_results=3,
    where={"filing_year": 2025},
    rerank=True
)
```

**Performance Trade-offs**:
- **Vector-only**: ~150ms, good for exploratory queries
- **Reranked**: ~170ms, optimal for high-precision results
- Reranking adds ~20ms overhead but significantly improves relevance

### Risk Scoring

Quantify the **Severity** and **Novelty** of risk disclosures:

```python
from sec_risk_api.scoring import RiskScorer

# Initialize scorer
scorer = RiskScorer()

# Score severity (how severe is this risk?)
chunk = {
    "text": "Catastrophic supply chain disruptions due to geopolitical conflicts",
    "metadata": {"ticker": "AAPL", "filing_year": 2025, "item_type": "Item 1A"}
}

severity = scorer.calculate_severity(chunk)
print(f"Severity: {severity.value:.2f}")
print(f"Explanation: {severity.explanation}")
print(f"Source: {severity.source_citation[:100]}...")

# Score novelty (how new is this vs. historical filings?)
historical_chunks = [
    {"text": "Standard competition risks", "metadata": {"ticker": "AAPL", "filing_year": 2024}}
]

novelty = scorer.calculate_novelty(chunk, historical_chunks)
print(f"Novelty: {novelty.value:.2f}")  # Higher = more novel
print(f"Explanation: {novelty.explanation}")

# Batch scoring for efficiency
chunks = [...]  # Multiple chunks
scores = scorer.calculate_severity_batch(chunks)
```

**Severity Scoring**:
- Analyzes keyword presence (catastrophic, severe, critical, etc.)
- Normalized to [0.0, 1.0] where 1.0 = catastrophic risk
- Every score includes source citation and explanation

**Novelty Scoring**:
- Compares current chunk with historical filing embeddings
- Measures semantic distance (cosine similarity)
- Novelty = 1 - max_similarity (higher = more novel)
- Edge case: No historical data → novelty = 1.0 (maximally novel)

### Integration Pipeline (End-to-End)

Run the complete analysis pipeline from HTML filing to structured risk scores:

```python
from sec_risk_api.integration import IntegrationPipeline

# Initialize integration pipeline
pipeline = IntegrationPipeline(persist_path="./chroma_db")

# Analyze a filing end-to-end
result = pipeline.analyze_filing(
    html_path="data/sample_10k.html",
    ticker="AAPL",
    filing_year=2025,
    retrieve_top_k=10  # Analyze top 10 risk factors
)

# Access structured results
print(f"Ticker: {result.ticker}")
print(f"Filing Year: {result.filing_year}")
print(f"Risks Analyzed: {len(result.risks)}")

# Each risk includes scores with full provenance
for risk in result.risks:
    print(f"\nRisk: {risk['text'][:100]}...")
    print(f"Severity: {risk['severity']['value']:.2f} - {risk['severity']['explanation']}")
    print(f"Novelty: {risk['novelty']['value']:.2f} - {risk['novelty']['explanation']}")
    print(f"Source: {risk['source_citation'][:80]}...")

# Export to JSON
json_output = result.to_json()
with open("risk_analysis.json", "w") as f:
    f.write(json_output)

# Access metadata
print(f"\nPipeline Metadata:")
print(f"Total latency: {result.metadata['total_latency_ms']:.2f}ms")
print(f"Chunks indexed: {result.metadata['chunks_indexed']}")
```

**Pipeline Flow**:
1. **Validation** - Check file exists, ticker format, year range
2. **Indexing** - Extract Item 1A → chunk → embed → store
3. **Retrieval** - Semantic search for top-k risk factors
4. **Scoring** - Compute severity and novelty for each
5. **Output** - Structured JSON with complete provenance

**Novelty Comparison**:
- Searches historical filings (up to 3 years back)
- First filing → novelty = 1.0 (no history)
- Repeated content → novelty ≈ 0.0 (identical to prior year)
- New risks → novelty > 0.7 (semantically distinct)

## REST API

The system exposes a FastAPI REST interface for production deployments.

### Starting the Server

```bash
# Development mode (auto-reload)
uv run uvicorn sec_risk_api.api:app --reload

# Production mode
uv run uvicorn sec_risk_api.api:app --host 0.0.0.0 --port 8000
```

### API Endpoints

#### POST /analyze

Analyze an SEC filing and return risk scores.

**Request Body**:
```json
{
  "ticker": "AAPL",
  "filing_year": 2025,
  "html_content": "<html>...</html>",
  "retrieve_top_k": 10
}
```

Or use a file path:
```json
{
  "ticker": "AAPL",
  "filing_year": 2025,
  "html_path": "/path/to/filing.html",
  "retrieve_top_k": 10
}
```

**Response**:
```json
{
  "ticker": "AAPL",
  "filing_year": 2025,
  "risks": [
    {
      "text": "Supply chain disruptions could materially impact...",
      "source_citation": "Supply chain disruptions...",
      "severity": {
        "value": 0.75,
        "explanation": "High severity due to keywords: severe, disruption"
      },
      "novelty": {
        "value": 0.82,
        "explanation": "High novelty - semantically distinct from 2024 filing"
      },
      "metadata": {
        "ticker": "AAPL",
        "filing_year": 2025,
        "item_type": "Item 1A"
      }
    }
  ],
  "metadata": {
    "total_latency_ms": 2534.5,
    "chunks_indexed": 5
  }
}
```

**Example - cURL**:
```bash
# With HTML content
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "AAPL",
    "filing_year": 2025,
    "html_content": "<html>...</html>",
    "retrieve_top_k": 5
  }'

# With file path
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "AAPL",
    "filing_year": 2025,
    "html_path": "data/sample_10k.html",
    "retrieve_top_k": 5
  }'
```

**Example - Python**:
```python
import requests

# With HTML content
with open("data/sample_10k.html", encoding="cp1252") as f:
    html_content = f.read()

response = requests.post("http://localhost:8000/analyze", json={
    "ticker": "AAPL",
    "filing_year": 2025,
    "html_content": html_content,
    "retrieve_top_k": 10
})

data = response.json()
for risk in data["risks"]:
    print(f"Severity: {risk['severity']['value']:.2f}")
    print(f"Novelty: {risk['novelty']['value']:.2f}")
    print(f"Text: {risk['text'][:100]}...\n")

# With file path (server must have access to file)
response = requests.post("http://localhost:8000/analyze", json={
    "ticker": "AAPL",
    "filing_year": 2025,
    "html_path": "/path/to/filing.html"
})
```

#### GET /health

Health check endpoint.

**Response**:
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "vector_db_initialized": true
}
```

#### GET /openapi.json

Auto-generated OpenAPI schema for API documentation.

**Interactive Documentation**:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Validation Rules

The API enforces strict validation via Pydantic:

- **ticker**: 1-10 uppercase alphanumeric characters
- **filing_year**: 1994-2050 (SEC EDGAR era)
- **html_content** XOR **html_path**: Exactly one must be provided
- **retrieve_top_k**: 1-100 (default: 10)

Invalid requests return **422 Unprocessable Entity** with detailed error messages.

## Testing

The project follows strict **Test-Driven Development (TDD)** and uses **mypy** for type safety.

```bash
# Run all unit tests
uv run python -m pytest

# Run tests with verbose output
uv run python -m pytest -v

# Debug tests in VS Code
# Use launch configuration "Python: Debug Tests (WSL)" or "Python: Debug Current Test File (WSL)"

# Check types
uv run mypy .

```
