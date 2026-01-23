import types
from types import SimpleNamespace
from datetime import datetime

import pytest

from sigmak.integration import IntegrationPipeline


class EmbeddingObj:
    def __init__(self, vec):
        self._vec = vec

    def tolist(self):
        return self._vec


class DriftStub:
    def __init__(self, results):
        # results should be a list returned by similarity_search
        self._results = results

    def similarity_search(self, query_embedding, n_results=1):
        return self._results


class EmbeddingsStub:
    def __init__(self, vec):
        self._vec = vec

    def encode(self, texts):
        # Return a list of EmbeddingObj objects to mimic real encode behavior
        return [EmbeddingObj(self._vec) for _ in texts]


def make_pipeline_with_stubs(results, vec=None):
    p = IntegrationPipeline(persist_path="./database", db_only_classification=True)
    # Stub indexing pipeline embeddings
    if vec is None:
        vec = [0.0] * 384
    p.indexing_pipeline = SimpleNamespace(embeddings=EmbeddingsStub(vec))
    # Stub drift system
    p._drift_system = DriftStub(results)
    return p


def test_db_only_returns_vector_match_when_similarity_high():
    results = [{
        "category": "OPERATIONAL",
        "confidence": 0.92,
        "similarity_score": 0.85,
    }]

    p = make_pipeline_with_stubs(results)

    out = p._classify_risk_db_only({"text": "Supply chain disruptions may impact operations."})

    assert out["method"] == "vector_db"
    assert out["category"] == "OPERATIONAL"
    assert pytest.approx(out["confidence"], rel=1e-3) == 0.92


def test_db_only_returns_uncategorized_when_no_results():
    results = []
    p = make_pipeline_with_stubs(results)

    out = p._classify_risk_db_only({"text": "An obscure new risk phrase."})

    assert out["method"] == "db_only_no_match"
    assert out["category"] == "UNCATEGORIZED"
    assert out["confidence"] == 0.0


def test_db_only_returns_low_similarity_marker_when_below_threshold():
    # similarity below default threshold 0.8
    results = [{
        "category": "REGULATORY",
        "confidence": 0.6,
        "similarity_score": 0.5,
    }]

    p = make_pipeline_with_stubs(results)

    out = p._classify_risk_db_only({"text": "Minor regulatory wording change."})

    assert out["method"] == "db_only_low_similarity"
    assert out["category"] == "UNCATEGORIZED"
    assert pytest.approx(out["confidence"], rel=1e-3) == 0.6
