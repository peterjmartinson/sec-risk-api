# SigmaK: Proprietary Risk Scoring API

[![CI](https://github.com/peterjmartinson/sigmak/actions/workflows/ci.yml/badge.svg)](https://github.com/peterjmartinson/sigmak/actions/workflows/ci.yml)

**SigmaK** is a RAG-powered pipeline designed to quantify novelty and severity in SEC **Item
1A: Risk Factors** disclosures. This system ingests structured HTML/XBRL
filings, extracts specific risk sections, and indexes them into a
high-dimensional vector space for semantic analysis.

> **Note**: Currently, the system uses **local ML models** (sentence-transformers) and **keyword-based algorithms** for risk scoring. No LLM API calls are made. See [LLM_API_INTEGRATION_STATUS.md](LLM_API_INTEGRATION_STATUS.md) for details on the current implementation and future LLM integration plans.

## System Architecture

The application follows a modular "Extraction-to-Storage" flow with **asynchronous task processing**:

1. **Ingestion Engine (`ingest.py`)**: Parses raw SEC HTM files using BeautifulSoup (lxml). It features robust encoding fallbacks (UTF-8/CP1252) and regex-based slicing to isolate "Item 1A."
2. **Processing Layer (`processing.py`)**: Implements recursive character text splitting. Chunks are normalized to ~800 characters with an 80-character overlap to preserve semantic context.
3. **Embedding Engine (`embeddings.py`)**: Converts text chunks into 384-dimensional semantic vectors using the `all-MiniLM-L6-v2` sentence transformer model.
4. **The Vault (`init_vector_db.py`)**: A persistent **Chroma DB** instance utilizing **HNSW** indexing with a **Cosine Similarity** metric. Storage path: `./chroma_db/`
5. **Reranking Layer (`reranking.py`)**: Cross-encoder model (`ms-marco-MiniLM-L-6-v2`) that reranks search results for improved relevance. Processes query-document pairs jointly for superior accuracy.
6. **Risk Classification (`risk_taxonomy.py`, `prompt_manager.py`)**: Proprietary 10-category risk taxonomy with version-controlled LLM prompts for semantic classification.
7. **Risk Scoring (`scoring.py`)**: Quantifies **Severity** (0.0-1.0) and **Novelty** (0.0-1.0) for risk disclosures using keyword analysis and semantic comparison with historical filings. Every score includes full source citation and human-readable explanation.
8. **Orchestration Layer (`indexing_pipeline.py`)**: End-to-end pipeline coordinator that integrates extraction, chunking, embedding, storage, and hybrid search with optional reranking.
9. **Async Task Queue (`tasks.py`)**: Celery + Redis background task processor for non-blocking API operations. All slow operations (ingestion, scoring) run in worker processes with progress tracking.
10. **REST API (`api.py`)**: FastAPI server with async endpoints, authentication, rate limiting, and real-time task status polling.

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

> **Current Implementation**: Risk scoring currently uses **keyword-based algorithms** and **semantic embeddings** (sentence-transformers). The prompt infrastructure is ready for future LLM classification integration. See [LLM_API_INTEGRATION_STATUS.md](LLM_API_INTEGRATION_STATUS.md) for details.

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

### Quick Start: Analyze a Filing

The fastest way to analyze a SEC filing is with the CLI utility:

```bash
# Download a 10-K HTML filing from SEC EDGAR
# Example: https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0001318605&type=10-K
# Save the main .htm file (NOT the -index.htm) to data/filings/

# Analyze the filing
uv run python analyze_filing.py data/filings/tsla-20221231.htm TSLA 2022
```

**Output**:
- ✅ Chunks indexed (e.g., 127 chunks from Tesla 2022)
- ✅ Top 5 risks with severity/novelty scores
- ✅ JSON export: `results_TSLA_2022.json`
- ✅ Persistent vector DB for future searches

**Batch Analysis**:
```bash
# Analyze multiple years for historical comparison
uv run python analyze_filing.py data/filings/tsla-20221231.htm TSLA 2022
uv run python analyze_filing.py data/filings/tsla-20231231.html TSLA 2023
uv run python analyze_filing.py data/filings/tsla-20241231.htm TSLA 2024
```

**Note**: Novelty scores improve with more historical data. The first filing gets novelty=1.0 (no baseline), subsequent years show real novelty detection by comparing against prior filings.

### Indexing a Filing (Programmatic)

Index a 10-K HTML filing into the vector database:

```python
from sigmak.indexing_pipeline import IndexingPipeline

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
from sigmak.scoring import RiskScorer

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
from sigmak.integration import IntegrationPipeline

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

The system exposes a FastAPI REST interface with **asynchronous task processing** for production deployments.

### Architecture: Async Task Queue

**Why Async?** Analysis tasks can take 5-30 seconds depending on filing size. Synchronous endpoints would block the API server, limiting throughput to 1-2 requests per second. With Celery + Redis, the API responds instantly (< 100ms) and processes tasks in parallel background workers.

**Components**:
- **FastAPI Server**: Accepts requests, validates input, submits tasks
- **Redis Broker**: Message queue for task distribution
- **Celery Workers**: Process tasks in parallel (scalable)
- **Redis Backend**: Stores task results and progress

**Task States**:
1. **PENDING**: Task queued but not started
2. **PROGRESS**: Task running (with progress updates)
3. **SUCCESS**: Task completed successfully
4. **FAILURE**: Task failed with error details

### Starting the System

```bash
# 1. Start Redis (message broker)
redis-server

# 2. Start Celery worker (in separate terminal)
celery -A sigmak.tasks worker --loglevel=info

# 3. Start API server (in separate terminal)
uv run uvicorn sigmak.api:app --reload --host 0.0.0.0 --port 8000
```

**Production Deployment**:
```bash
# Run multiple workers for horizontal scaling
celery -A sigmak.tasks worker --concurrency=4 --loglevel=info

# Optional: Use Flower for monitoring
celery -A sigmak.tasks flower
```

### API Endpoints

#### POST /analyze (Async)

Submit an analysis task and receive a task_id for polling.

**Authentication**: Requires `X-API-Key` header (see Authentication section below)

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

**Response (HTTP 202 Accepted)**:
```json
{
  "task_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status_url": "/tasks/a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "message": "Analysis task submitted successfully"
}
```

**Example - cURL**:
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "X-API-Key: your-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "AAPL",
    "filing_year": 2025,
    "html_content": "<html>...</html>",
    "retrieve_top_k": 5
  }'
```

**Example - Python**:
```python
import requests
import time

# Submit analysis task
response = requests.post(
    "http://localhost:8000/analyze",
    headers={"X-API-Key": "your-api-key-here"},
    json={
        "ticker": "AAPL",
        "filing_year": 2025,
        "html_path": "data/sample_10k.html",
        "retrieve_top_k": 10
    }
)

task_id = response.json()["task_id"]
print(f"Task submitted: {task_id}")

# Poll for results
while True:
    status_response = requests.get(
        f"http://localhost:8000/tasks/{task_id}",
        headers={"X-API-Key": "your-api-key-here"}
    )

    status_data = status_response.json()

    if status_data["status"] == "SUCCESS":
        result = status_data["result"]
        print(f"Analysis complete! {len(result['risks'])} risks found")
        break
    elif status_data["status"] == "FAILURE":
        print(f"Task failed: {status_data['error']}")
        break
    elif status_data["status"] == "PROGRESS":
        progress = status_data["progress"]
        print(f"Progress: {progress['current']}/{progress['total']} - {progress['status']}")

    time.sleep(2)  # Poll every 2 seconds
```

#### GET /tasks/{task_id}

Poll task status and retrieve results when complete.

**Authentication**: Requires `X-API-Key` header

**Response (PENDING)**:
```json
{
  "task_id": "a1b2c3d4-...",
  "status": "PENDING",
  "progress": null,
  "result": null,
  "error": null
}
```

**Response (PROGRESS)**:
```json
{
  "task_id": "a1b2c3d4-...",
  "status": "PROGRESS",
  "progress": {
    "current": 3,
    "total": 5,
    "status": "Computing severity scores..."
  },
  "result": null,
  "error": null
}
```

**Response (SUCCESS)**:
```json
{
  "task_id": "a1b2c3d4-...",
  "status": "SUCCESS",
  "progress": null,
  "result": {
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
  },
  "error": null
}
```

**Response (FAILURE)**:
```json
{
  "task_id": "a1b2c3d4-...",
  "status": "FAILURE",
  "progress": null,
  "result": null,
  "error": "File not found: /path/to/filing.html"
}
```

#### POST /index (Async)

Submit an indexing task (no scoring, just ingestion into vector DB).

**Authentication**: Requires `X-API-Key` header

**Response**: Same format as POST /analyze (returns task_id)

**Use Case**: Bulk ingestion of filings for later analysis.

#### GET /health

Health check endpoint (no authentication required).

**Response**:
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

### Authentication & Rate Limiting

**API Keys**: All endpoints (except /health) require authentication.

**Creating API Keys**:
```python
from sigmak.auth import APIKeyManager

manager = APIKeyManager()

# Create API key with rate limit
api_key = manager.create_api_key(user="client_name", rate_limit="10/minute")
print(f"API Key: {api_key}")

# Keys are stored in api_keys.json
```

**Rate Limits**:
- Default: 10 requests/minute per API key
- Configurable per-user via `rate_limit` parameter
- Exceeding limit returns HTTP 429 (Too Many Requests)

**Request Headers**:
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "X-API-Key: your-api-key-here" \
  -H "Content-Type: application/json" \
  -d '...'
```

### Validation Rules

The API enforces strict validation via Pydantic:

- **ticker**: 1-10 uppercase alphanumeric characters
- **filing_year**: 1994-2050 (SEC EDGAR era)
- **html_content** XOR **html_path**: Exactly one must be provided
- **retrieve_top_k**: 1-100 (default: 10)

Invalid requests return **422 Unprocessable Entity** with detailed error messages.

## Deployment

### Docker Deployment

The application includes full Docker support with multi-stage builds and health checks.

**Build and Run with Docker Compose**:
```bash
# Build all services
docker-compose build

# Start all services (API, Worker, Redis)
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f api
docker-compose logs -f worker

# Stop all services
docker-compose down
```

**Services**:
- **API**: FastAPI server on port 8000
- **Worker**: Celery worker for background tasks
- **Redis**: Message broker and result backend

**Environment Variables**:
```bash
# Set in docker-compose.yml or .env file
REDIS_URL=redis://redis:6379/0
LOG_LEVEL=INFO
ENVIRONMENT=production
CHROMA_PERSIST_PATH=/app/chroma_db
```

**Health Checks**:
- API: `http://localhost:8000/health` (basic health)
- API: `http://localhost:8000/ready` (readiness with dependency checks)
- API: `http://localhost:8000/live` (liveness probe)

### Cloud Deployment (Digital Ocean)

**Prerequisites**:
- Digital Ocean account
- Docker installed on Droplet
- Domain name (optional, for HTTPS)

**Setup Steps**:

1. **Create Droplet**:
```bash
# Recommended: Ubuntu 22.04 LTS, 4GB RAM minimum
doctl compute droplet create sigmak \
  --image ubuntu-22-04-x64 \
  --size s-2vcpu-4gb \
  --region nyc3 \
  --ssh-keys your-ssh-key-id
```

2. **SSH into Droplet**:
```bash
ssh root@your-droplet-ip
```

3. **Install Docker**:
```bash
# Update packages
apt-get update && apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install Docker Compose
apt-get install docker-compose-plugin -y
```

4. **Clone Repository**:
```bash
git clone https://github.com/your-username/sigmak.git
cd sigmak
```

5. **Configure Environment**:
```bash
# Create .env file
cat > .env <<EOF
REDIS_URL=redis://redis:6379/0
LOG_LEVEL=INFO
ENVIRONMENT=production
CHROMA_PERSIST_PATH=/app/chroma_db
EOF
```

6. **Deploy Application**:
```bash
# Build and start services
docker-compose up -d

# Verify deployment
curl http://localhost:8000/health
```

7. **Setup Firewall** (recommended):
```bash
ufw allow 22/tcp  # SSH
ufw allow 8000/tcp  # API
ufw enable
```

8. **Setup HTTPS with Nginx** (recommended for production):
```bash
# Install Nginx and Certbot
apt-get install nginx certbot python3-certbot-nginx -y

# Configure Nginx as reverse proxy
cat > /etc/nginx/sites-available/sigmak <<'EOF'
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
EOF

ln -s /etc/nginx/sites-available/sigmak /etc/nginx/sites-enabled/
systemctl reload nginx

# Get SSL certificate
certbot --nginx -d your-domain.com
```

### Monitoring & Observability

The application includes comprehensive monitoring:

**Structured Logging**:
- All logs are JSON-formatted for aggregation
- Includes: timestamps, log levels, request IDs, latency metrics
- LLM usage tracking: token counts, costs, model versions

**Metrics Collection**:
- Request counters by endpoint
- Latency histograms (p50, p95, p99)
- Error rates and categorization
- Celery task metrics

**Health Checks**:
- `/health`: Basic health check
- `/ready`: Readiness probe (checks Redis, ChromaDB)
- `/live`: Liveness probe (detects deadlocks)

**Graceful Shutdown**:
- Responds to SIGTERM signal
- Waits for in-flight requests to complete (30s timeout)
- Prevents new requests during shutdown

**View Logs**:
```bash
# Follow API logs
docker-compose logs -f api

# Follow worker logs
docker-compose logs -f worker

# View structured JSON logs
docker-compose logs api | jq .
```

### Performance Tuning

**Celery Workers**:
```bash
# Adjust concurrency (default: 2)
celery -A sigmak.tasks worker --concurrency=4
```

**API Server**:
```bash
# Run multiple workers with Gunicorn
gunicorn sigmak.api:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

**Redis Memory**:
```yaml
# In docker-compose.yml
redis:
  command: redis-server --maxmemory 512mb --maxmemory-policy allkeys-lru
```

### Maintenance

**Backup ChromaDB**:
```bash
# Create backup
tar -czf chroma_backup_$(date +%Y%m%d).tar.gz ./chroma_db/

# Restore backup
tar -xzf chroma_backup_20250106.tar.gz
```

**Update Application**:
```bash
# Pull latest code
git pull origin main

# Rebuild and restart
docker-compose down
docker-compose build
docker-compose up -d
```

**Monitor Resource Usage**:
```bash
# Docker stats
docker stats

# Disk usage
docker system df
```

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
