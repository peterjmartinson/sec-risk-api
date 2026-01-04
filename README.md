# Scream Sheet: Proprietary Risk Scoring API

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
5. **Orchestration Layer (`indexing_pipeline.py`)**: End-to-end pipeline coordinator that integrates extraction, chunking, embedding, and storage. Supports intelligent upserts and semantic retrieval.

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
# Search across all filings
results = pipeline.semantic_search(
    query="Geopolitical risks and international conflicts",
    n_results=5
)

# Search with metadata filtering
results = pipeline.semantic_search(
    query="Supply chain disruptions",
    n_results=5,
    where={"ticker": "AAPL", "filing_year": 2025}
)

for result in results:
    print(f"[{result['metadata']['ticker']}] {result['text'][:100]}...")
    print(f"Distance: {result['distance']:.4f}\n")
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
