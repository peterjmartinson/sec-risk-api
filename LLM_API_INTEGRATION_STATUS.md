# LLM API Integration Status

## Executive Summary

**Currently, there are NO active LLM API calls in the SigmaK codebase.** 

The system has been designed with LLM integration in mind (prompt management, usage tracking, risk taxonomy), but the actual LLM API client code has not yet been implemented. Risk scoring is currently performed using **keyword-based algorithms** and **semantic embeddings** (via sentence-transformers), not LLM classification.

---

## What Steps Include LLM API Calls?

### ❌ None Currently

The following pipeline steps do **NOT** call LLM APIs:

1. **Ingestion** (`ingest.py`) - Pure HTML parsing with BeautifulSoup
2. **Processing** (`processing.py`) - Text chunking via LangChain
3. **Embedding** (`embeddings.py`) - Local sentence-transformers model (all-MiniLM-L6-v2)
4. **Indexing** (`indexing_pipeline.py`) - ChromaDB vector storage
5. **Retrieval** (`indexing_pipeline.py`) - Cosine similarity search
6. **Reranking** (`reranking.py`) - Cross-encoder model (ms-marco-MiniLM-L-6-v2)
7. **Severity Scoring** (`scoring.py`) - Keyword frequency analysis
8. **Novelty Scoring** (`scoring.py`) - Embedding distance comparison
9. **Risk Classification** - Not yet implemented (infrastructure in place)

---

## Infrastructure Ready for LLM Integration

The codebase has been architected to support future LLM integration:

### 1. Prompt Management (`prompt_manager.py`)

**Purpose**: Version-controlled system prompts for LLM classification

**Status**: ✅ Implemented, ready to use

**Features**:
- Loads prompts from `prompts/` directory
- Version tracking (e.g., `risk_classification_v1.txt`, `v2`, etc.)
- Automatic latest version resolution
- Metadata tracking (file size, path, version history)

**Example Usage** (when LLM integration is added):
```python
from sigmak.prompt_manager import PromptManager

manager = PromptManager()
system_prompt = manager.load_latest("risk_classification")
# Pass to OpenAI/Anthropic API
```

### 2. Risk Taxonomy (`risk_taxonomy.py`)

**Purpose**: 10-category proprietary risk classification schema

**Status**: ✅ Defined, ready for LLM consumption

**Categories**:
1. OPERATIONAL - Internal execution risks
2. SYSTEMATIC - Macroeconomic forces
3. GEOPOLITICAL - International conflicts
4. REGULATORY - Compliance and legal
5. COMPETITIVE - Market rivalry
6. TECHNOLOGICAL - Innovation threats
7. HUMAN_CAPITAL - Workforce risks
8. FINANCIAL - Capital structure
9. REPUTATIONAL - Brand and trust
10. OTHER - Miscellaneous

**Design**: Categories include keywords, severity multipliers, and descriptions that can be used in LLM prompts.

### 3. LLM Usage Tracking (`monitoring.py`)

**Purpose**: Log LLM API calls for cost tracking and performance monitoring

**Status**: ✅ Function implemented, not yet called

**Function Signature**:
```python
def log_llm_usage(
    model: str,              # e.g., "gpt-4-turbo"
    prompt_tokens: int,      # Token count
    completion_tokens: int,  # Response tokens
    total_cost: float,       # USD cost
    latency_ms: float        # API latency
) -> None
```

**Output**: Structured JSON logs for cost analysis:
```json
{
  "timestamp": "2026-01-10T23:45:00Z",
  "level": "INFO",
  "message": "LLM usage: gpt-4-turbo - 250 prompt + 100 completion tokens",
  "model": "gpt-4-turbo",
  "prompt_tokens": 250,
  "completion_tokens": 100,
  "total_tokens": 350,
  "cost_usd": 0.015,
  "latency_ms": 1234.5
}
```

### 4. Prompts Directory (`prompts/`)

**Status**: ✅ Created with initial classification prompt

**Contents**:
- `risk_classification_v1.txt` - System prompt for risk categorization
- `README.md` - Prompt design principles
- `CHANGELOG.md` - Version history

**Prompt Features**:
- Detailed category definitions with examples
- JSON output format specification
- Confidence scoring requirements
- Source citation enforcement
- Edge case handling instructions

---

## Current Risk Scoring Implementation

Since LLMs are not yet integrated, risk scoring uses alternative methods:

### Severity Scoring (Keyword-Based)

**Location**: `scoring.py` → `RiskScorer.calculate_severity()`

**Algorithm**:
1. Convert text to lowercase
2. Count severe keywords (catastrophic, existential, severe, etc.) × 2.0 weight
3. Count moderate keywords (challenge, risk, uncertainty, etc.) × 1.0 weight
4. Normalize by expected maximum (15 total weighted keywords)
5. Boost score if ≥3 severe keywords (compound risk indicator)
6. Clamp to [0.0, 1.0] range

**Example**:
```python
text = "Catastrophic supply chain failure could severely disrupt operations"
# severe_matches = 3 (catastrophic, severe, disrupt)
# moderate_matches = 1 (operations)
# raw_score = (3 × 2.0 + 1 × 1.0) / 15.0 = 0.47
# After boost (≥3 severe): 0.47 × 1.2 = 0.56
```

**No LLM API calls**: Pure keyword matching with NumPy

### Novelty Scoring (Embedding-Based)

**Location**: `scoring.py` → `RiskScorer.calculate_novelty()`

**Algorithm**:
1. Generate embedding for current chunk (sentence-transformers)
2. Retrieve historical chunks from past 3 years
3. Generate embeddings for all historical chunks
4. Compute cosine similarities
5. Novelty = 1 - max(similarities)
6. Higher distance = more novel

**Example**:
```python
current = "Quantum computing threats to encryption"
historical = ["Standard cybersecurity risks", "Network security"]
# High semantic distance → high novelty score (e.g., 0.85)

current = "Continued supply chain challenges"
historical = ["Supply chain disruptions due to...", "Supply chain risks"]
# Low semantic distance → low novelty score (e.g., 0.15)
```

**No LLM API calls**: Uses local sentence-transformers model (384-dimensional embeddings)

---

## When Would LLM APIs Be Called?

If LLM classification is implemented in the future, it would likely be integrated into:

### Proposed Integration Point: `scoring.py` or new `classification.py` module

**Hypothetical Flow**:
```python
from sigmak.prompt_manager import PromptManager
from sigmak.monitoring import log_llm_usage
import openai  # Not currently in dependencies

def classify_risk_with_llm(chunk: dict) -> dict:
    """Classify risk using LLM (NOT YET IMPLEMENTED)."""
    
    # 1. Load versioned prompt
    manager = PromptManager()
    system_prompt = manager.load_latest("risk_classification")
    
    # 2. Call LLM API
    start_time = time.time()
    response = openai.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": chunk["text"]}
        ],
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    latency_ms = (time.time() - start_time) * 1000
    
    # 3. Log usage
    log_llm_usage(
        model="gpt-4-turbo",
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
        total_cost=calculate_cost(response.usage),
        latency_ms=latency_ms
    )
    
    # 4. Parse response
    classification = json.loads(response.choices[0].message.content)
    return classification
```

**This code does not exist yet.**

---

## Dependencies Check

Current dependencies do **NOT** include LLM clients:

```toml
# pyproject.toml
dependencies = [
    "beautifulsoup4>=4.12.0",
    "lxml>=5.0.0",
    "chromadb>=0.4.0",
    "sentence-transformers>=5.2.0",  # Local embeddings only
    "numpy>=2.2.6",
    "scikit-learn>=1.7.2",
    "fastapi>=0.128.0",
    "celery>=5.3.0",
    "redis>=5.0.0",
    # NO openai, anthropic, langchain, etc.
]
```

To add LLM integration, dependencies would need:
```toml
"openai>=1.0.0",       # For GPT-4, etc.
# OR
"anthropic>=0.8.0",    # For Claude
# OR  
"langchain>=0.1.0",    # For multi-provider abstraction
```

---

## Summary

### What DOES use external APIs?
- **None** - All processing is local

### What uses ML models?
- **Embedding generation**: `sentence-transformers` (all-MiniLM-L6-v2) - runs locally on CPU/GPU
- **Reranking**: Cross-encoder (ms-marco-MiniLM-L-6-v2) - runs locally
- **Severity/novelty scoring**: Keyword algorithms + embedding similarity - no API calls

### What's ready for LLM integration?
1. ✅ Prompt management system (`prompt_manager.py`)
2. ✅ Risk taxonomy definitions (`risk_taxonomy.py`)
3. ✅ Usage tracking function (`monitoring.log_llm_usage()`)
4. ✅ Classification prompt templates (`prompts/risk_classification_v1.txt`)

### What's missing for LLM integration?
1. ❌ LLM client library (openai, anthropic, etc.)
2. ❌ API key management and configuration
3. ❌ Classification function that calls LLM
4. ❌ Error handling for API failures
5. ❌ Rate limiting and retry logic
6. ❌ Cost budgeting and alerts

---

## Recommended Next Steps (If LLM Integration Is Desired)

1. **Choose LLM Provider**: OpenAI (GPT-4), Anthropic (Claude), or open-source (Llama via Ollama)

2. **Add Dependencies**:
   ```bash
   uv add openai  # or anthropic, or langchain
   ```

3. **Implement Classification Module**:
   - Create `src/sigmak/classification.py`
   - Implement `classify_risk()` function
   - Use existing `prompt_manager` to load prompts
   - Call `log_llm_usage()` after each API call

4. **Integrate into Pipeline**:
   - Modify `integration.py` to optionally use LLM classification
   - Add `use_llm_classification` parameter to `analyze_filing()`
   - Fall back to keyword-based scoring if LLM fails

5. **Add Tests**:
   - Mock LLM responses for unit tests
   - Test API error handling
   - Validate JSON parsing from LLM responses

6. **Document Cost Implications**:
   - Estimate API costs per filing (e.g., $0.50 - $2.00 per 10-K)
   - Add cost budgeting and alerts

---

## Conclusion

**The SigmaK system does NOT currently make any LLM API calls.** 

All risk analysis is performed using:
- Local ML models (sentence-transformers, cross-encoders)
- Keyword-based algorithms
- Semantic similarity computations

However, the infrastructure is **well-architected for future LLM integration**, with prompt management, usage tracking, and risk taxonomy already in place. The system can function as a high-quality risk analysis tool without LLMs, and LLM classification can be added as an optional enhancement when needed.
