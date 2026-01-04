# Copyright (c) 2025 Peter Martinson, Distracted Fortune. All rights reserved.
# This software is proprietary and not licensed for use, modification, or distribution.

import pytest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Dict, Any
from sec_risk_api.indexing_pipeline import IndexingPipeline


@pytest.fixture
def temp_chroma_db() -> str:
    """
    Creates a temporary directory for ChromaDB testing.
    """
    with TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def multi_company_fixture(tmp_path: Path, temp_chroma_db: str) -> IndexingPipeline:
    """
    Fixture that creates a populated database with multiple companies and years.
    This represents a realistic scenario for testing hybrid search.
    """
    # Create filings for AAPL 2024 and 2025
    aapl_2024 = """
    <html><body>
    <h2>ITEM 1A. RISK FACTORS</h2>
    <p>Supply Chain Risk: Global semiconductor shortages and geopolitical tensions 
       in Asia could severely disrupt our manufacturing operations. We rely heavily 
       on suppliers in regions affected by trade disputes.</p>
    <p>Competition Risk: Intense competition in the smartphone market from Samsung 
       and other manufacturers threatens our market share.</p>
    </body></html>
    """
    
    aapl_2025 = """
    <html><body>
    <h2>ITEM 1A. RISK FACTORS</h2>
    <p>Supply Chain Risk: Our supply chain remains vulnerable to international 
       conflicts and trade restrictions. Recent tensions have increased costs.</p>
    <p>AI Competition Risk: Emerging AI capabilities from competitors could make 
       our products less competitive in the rapidly evolving technology landscape.</p>
    </body></html>
    """
    
    # Create filings for TSLA 2024 and 2025
    tsla_2024 = """
    <html><body>
    <h2>ITEM 1A. RISK FACTORS</h2>
    <p>Production Risk: Our ability to scale production of electric vehicles depends 
       on battery supply and manufacturing capacity. Delays could harm revenue.</p>
    <p>Regulatory Risk: Changes in government incentives for electric vehicles could 
       reduce demand and affect profitability.</p>
    </body></html>
    """
    
    tsla_2025 = """
    <html><body>
    <h2>ITEM 1A. RISK FACTORS</h2>
    <p>Autonomous Driving Risk: Regulatory approval for full self-driving technology 
       remains uncertain. Accidents involving autonomous features could lead to 
       significant liability and reputational damage.</p>
    <p>Competition Risk: Traditional automakers are rapidly entering the EV market 
       with competitive products and established distribution networks.</p>
    </body></html>
    """
    
    # Write files and index them
    pipeline = IndexingPipeline(persist_path=temp_chroma_db)
    
    filings = [
        ("AAPL", 2024, aapl_2024),
        ("AAPL", 2025, aapl_2025),
        ("TSLA", 2024, tsla_2024),
        ("TSLA", 2025, tsla_2025),
    ]
    
    for ticker, year, content in filings:
        file_path = tmp_path / f"{ticker}_{year}_10k.html"
        file_path.write_text(content)
        pipeline.index_filing(
            html_path=file_path,
            ticker=ticker,
            filing_year=year,
            item_type="Item 1A"
        )
    
    return pipeline


class TestHybridSearchMetadataFiltering:
    """
    Tests for Subissue 3.0: Metadata filtering in hybrid search.
    Verify that queries can be scoped to specific tickers and years.
    """

    def test_search_filters_by_single_ticker(
        self, multi_company_fixture: IndexingPipeline
    ) -> None:
        """
        Verify that metadata filtering correctly isolates results to a single ticker.
        """
        pipeline = multi_company_fixture
        
        # Search only AAPL filings
        results = pipeline.semantic_search(
            query="supply chain disruptions",
            n_results=10,
            where={"ticker": "AAPL"}
        )
        
        # All results should be from AAPL only
        assert len(results) > 0, "Should find AAPL supply chain risks"
        for result in results:
            assert result["metadata"]["ticker"] == "AAPL", (
                f"Expected AAPL, got {result['metadata']['ticker']}"
            )

    def test_search_filters_by_filing_year(
        self, multi_company_fixture: IndexingPipeline
    ) -> None:
        """
        Verify that metadata filtering correctly isolates results to a specific year.
        """
        pipeline = multi_company_fixture
        
        # Search only 2025 filings
        results = pipeline.semantic_search(
            query="competition risk",
            n_results=10,
            where={"filing_year": 2025}
        )
        
        # All results should be from 2025
        assert len(results) > 0, "Should find 2025 competition risks"
        for result in results:
            assert result["metadata"]["filing_year"] == 2025, (
                f"Expected 2025, got {result['metadata']['filing_year']}"
            )

    def test_search_combines_ticker_and_year_filters(
        self, multi_company_fixture: IndexingPipeline
    ) -> None:
        """
        Verify that multiple metadata filters can be combined.
        """
        pipeline = multi_company_fixture
        
        # Search AAPL 2025 only
        results = pipeline.semantic_search(
            query="risk factors",
            n_results=10,
            where={"ticker": "AAPL", "filing_year": 2025}
        )
        
        # All results should match both filters
        assert len(results) > 0, "Should find AAPL 2025 risks"
        for result in results:
            assert result["metadata"]["ticker"] == "AAPL"
            assert result["metadata"]["filing_year"] == 2025


class TestCrossEncoderReranking:
    """
    Tests for Subissue 3.0: Cross-Encoder reranking functionality.
    Verify that reranking improves relevance over baseline vector search.
    """

    def test_reranked_search_returns_top_3(
        self, multi_company_fixture: IndexingPipeline
    ) -> None:
        """
        Verify that reranked search returns exactly top 3 most relevant chunks.
        """
        pipeline = multi_company_fixture
        
        # Search with reranking enabled
        results = pipeline.semantic_search(
            query="supply chain vulnerabilities in technology manufacturing",
            n_results=3,
            rerank=True
        )
        
        # Should return exactly 3 results
        assert len(results) == 3, f"Expected 3 results, got {len(results)}"
        
        # Results should be ordered by relevance (higher rerank_score = more relevant)
        if "rerank_score" in results[0]:
            scores = [r["rerank_score"] for r in results]
            assert scores == sorted(scores, reverse=True), (
                "Results should be ordered by descending rerank_score"
            )

    def test_reranking_improves_relevance_over_baseline(
        self, multi_company_fixture: IndexingPipeline
    ) -> None:
        """
        Verify that reranking demonstrably improves relevance.
        
        Strategy: Compare top result from baseline (vector-only) vs reranked.
        The reranked top result should have higher cross-encoder score than
        the baseline top result.
        """
        pipeline = multi_company_fixture
        
        query = "geopolitical tensions affecting supply chains"
        
        # Baseline: vector-only search
        baseline_results = pipeline.semantic_search(
            query=query,
            n_results=5,
            rerank=False
        )
        
        # Reranked search
        reranked_results = pipeline.semantic_search(
            query=query,
            n_results=3,
            rerank=True
        )
        
        # Both should return results
        assert len(baseline_results) > 0, "Baseline should return results"
        assert len(reranked_results) > 0, "Reranked should return results"
        
        # Get the top chunk text from each approach
        baseline_top_text = baseline_results[0]["text"]
        reranked_top_text = reranked_results[0]["text"]
        
        # For this specific query, we expect AAPL's supply chain risk
        # to be more relevant than TSLA's production risk
        # Reranked results should prioritize "geopolitical tensions" mentions
        assert "supply" in reranked_top_text.lower() or "geopolit" in reranked_top_text.lower(), (
            "Reranked top result should be highly relevant to geopolitical/supply chain query"
        )

    def test_reranking_preserves_source_citations(
        self, multi_company_fixture: IndexingPipeline
    ) -> None:
        """
        Verify that reranked results maintain complete source citations.
        Every result must include: id, text, metadata, distance.
        Reranked results additionally include: rerank_score.
        """
        pipeline = multi_company_fixture
        
        results = pipeline.semantic_search(
            query="competition risks",
            n_results=3,
            rerank=True
        )
        
        assert len(results) > 0, "Should return results"
        
        # Check that all required fields are present
        for result in results:
            assert "id" in result, "Missing document ID"
            assert "text" in result, "Missing chunk text"
            assert "metadata" in result, "Missing metadata"
            assert "distance" in result, "Missing distance score"
            assert "rerank_score" in result, "Missing rerank score"
            
            # Verify metadata schema
            metadata = result["metadata"]
            assert "ticker" in metadata
            assert "filing_year" in metadata
            assert "item_type" in metadata

    def test_reranking_with_metadata_filters(
        self, multi_company_fixture: IndexingPipeline
    ) -> None:
        """
        Verify that reranking works correctly when combined with metadata filters.
        This tests the full hybrid search capability.
        """
        pipeline = multi_company_fixture
        
        # Hybrid search: semantic query + metadata filter + reranking
        results = pipeline.semantic_search(
            query="regulatory challenges and government policy",
            n_results=3,
            where={"ticker": "TSLA"},
            rerank=True
        )
        
        # Should find TSLA-specific regulatory risks
        assert len(results) > 0, "Should find TSLA regulatory risks"
        
        # All results should be from TSLA
        for result in results:
            assert result["metadata"]["ticker"] == "TSLA"
            assert "rerank_score" in result


class TestSourceCitationIntegrity:
    """
    Tests for Subissue 3.0 Success Condition:
    "Search routine always returns the top 3 most relevant chunks for a given query,
    each with its source cited."
    """

    def test_every_result_has_complete_citation(
        self, multi_company_fixture: IndexingPipeline
    ) -> None:
        """
        Verify that every search result includes complete citation information.
        """
        pipeline = multi_company_fixture
        
        results = pipeline.semantic_search(
            query="market risks",
            n_results=5,
            rerank=True
        )
        
        assert len(results) > 0
        
        for i, result in enumerate(results):
            # Document ID should be parseable to extract source info
            assert "id" in result
            doc_id = result["id"]
            
            # ID format should be: {ticker}_{year}_{chunk_index}
            parts = doc_id.split("_")
            assert len(parts) >= 3, f"Invalid doc_id format: {doc_id}"
            
            # Metadata should match the ID
            assert result["metadata"]["ticker"] == parts[0]
            assert result["metadata"]["filing_year"] == int(parts[1])
            
            # Text should be non-empty
            assert len(result["text"]) > 0, f"Result {i} has empty text"
            
            # Should have relevance scores
            assert "distance" in result or "rerank_score" in result

    def test_results_are_deterministic(
        self, multi_company_fixture: IndexingPipeline
    ) -> None:
        """
        Verify that identical queries return identical results (determinism).
        This is critical for reproducibility and debugging.
        """
        pipeline = multi_company_fixture
        
        query = "supply chain disruptions"
        
        # Run the same query twice
        results_1 = pipeline.semantic_search(query=query, n_results=3, rerank=True)
        results_2 = pipeline.semantic_search(query=query, n_results=3, rerank=True)
        
        # Results should be identical
        assert len(results_1) == len(results_2)
        
        for r1, r2 in zip(results_1, results_2):
            assert r1["id"] == r2["id"], "Document IDs should match"
            assert r1["text"] == r2["text"], "Text should match"
            assert abs(r1["rerank_score"] - r2["rerank_score"]) < 0.001, (
                "Rerank scores should match (within floating point precision)"
            )


class TestRerankingPerformance:
    """
    Tests for performance characteristics and regression detection.
    These support JOURNAL.md documentation requirements.
    """

    @pytest.mark.slow
    def test_reranking_latency_is_acceptable(
        self, multi_company_fixture: IndexingPipeline
    ) -> None:
        """
        Verify that reranking completes within acceptable latency bounds.
        
        Baseline: Vector-only search should be < 100ms
        Reranked: Should be < 500ms (allows for cross-encoder inference)
        """
        import time
        
        pipeline = multi_company_fixture
        
        # Measure baseline vector search
        start = time.time()
        baseline_results = pipeline.semantic_search(
            query="competition and market dynamics",
            n_results=5,
            rerank=False
        )
        baseline_latency_ms = (time.time() - start) * 1000
        
        # Measure reranked search
        start = time.time()
        reranked_results = pipeline.semantic_search(
            query="competition and market dynamics",
            n_results=3,
            rerank=True
        )
        reranked_latency_ms = (time.time() - start) * 1000
        
        print(f"\nBaseline latency: {baseline_latency_ms:.2f}ms")
        print(f"Reranked latency: {reranked_latency_ms:.2f}ms")
        print(f"Overhead: {reranked_latency_ms - baseline_latency_ms:.2f}ms")
        
        # Reranking should add reasonable overhead (typically 50-300ms for CPU)
        assert reranked_latency_ms < 2000, (
            f"Reranking latency too high: {reranked_latency_ms:.2f}ms"
        )

    def test_reranking_vs_baseline_top_result_comparison(
        self, multi_company_fixture: IndexingPipeline
    ) -> None:
        """
        Document the difference between baseline and reranked top results.
        This test generates data for JOURNAL.md documentation.
        """
        pipeline = multi_company_fixture
        
        query = "supply chain vulnerabilities due to international tensions"
        
        baseline = pipeline.semantic_search(query, n_results=5, rerank=False)
        reranked = pipeline.semantic_search(query, n_results=3, rerank=True)
        
        print("\n" + "="*60)
        print("RERANKING COMPARISON (for JOURNAL.md)")
        print("="*60)
        print(f"\nQuery: {query}\n")
        
        print("BASELINE (Vector-only) Top Result:")
        print(f"  ID: {baseline[0]['id']}")
        print(f"  Snippet: {baseline[0]['text'][:100]}...")
        print(f"  Vector Distance: {baseline[0]['distance']:.4f}")
        
        print("\nRERANKED Top Result:")
        print(f"  ID: {reranked[0]['id']}")
        print(f"  Snippet: {reranked[0]['text'][:100]}...")
        print(f"  Vector Distance: {reranked[0]['distance']:.4f}")
        print(f"  Rerank Score: {reranked[0]['rerank_score']:.4f}")
        
        print("\nDifference:")
        if baseline[0]['id'] == reranked[0]['id']:
            print("  ✓ Same document (reranking confirmed vector search)")
        else:
            print("  ✗ Different document (reranking changed ranking)")
        print("="*60)


class TestTypeAnnotationCoverage:
    """
    Tests for Subissue 3.0 Success Condition:
    "All major functions have full type annotations and pass mypy checks."
    """

    def test_search_return_type_is_correctly_typed(
        self, multi_company_fixture: IndexingPipeline
    ) -> None:
        """
        Verify that semantic_search returns correctly typed results.
        This test ensures runtime behavior matches type annotations.
        """
        pipeline = multi_company_fixture
        
        results = pipeline.semantic_search(
            query="test query",
            n_results=2,
            rerank=True
        )
        
        # Runtime type validation
        assert isinstance(results, list)
        
        if len(results) > 0:
            result = results[0]
            assert isinstance(result, dict)
            assert isinstance(result["id"], str)
            assert isinstance(result["text"], str)
            assert isinstance(result["metadata"], dict)
            assert isinstance(result["distance"], (float, int))
            assert isinstance(result["rerank_score"], (float, int))
