# Scream Sheet: Proprietary Risk Scoring API

A RAG-powered pipeline designed to quantify novelty and severity in SEC **Item
1A: Risk Factors** disclosures. This system ingests structured HTML/XBRL
filings, extracts specific risk sections, and indexes them into a
high-dimensional vector space for semantic analysis.

## System Architecture

The application follows a modular "Extraction-to-Storage" flow:

1. **Ingestion Engine (`ingest.py`)**: Parses raw SEC HTM files using BeautifulSoup (lxml). It features robust encoding fallbacks (UTF-8/CP1252) and regex-based slicing to isolate "Item 1A."
2. **Processing Layer (`processing.py`)**: Implements recursive character text splitting. Chunks are normalized to ~800 characters with an 80-character overlap to preserve semantic context.
3. **The Vault (`init_vector_db.py`)**: A persistent **Chroma DB** instance utilizing **HNSW** indexing with a **Cosine Similarity** metric.

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

## Testing

The project follows strict **Test-Driven Development (TDD)** and uses **mypy** for type safety.

```bash
# Run unit tests
uv run python -m unittest discover tests

# Check types
uv run mypy .

```
