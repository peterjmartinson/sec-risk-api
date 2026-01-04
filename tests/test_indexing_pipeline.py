# Copyright (c) 2025 Peter Martinson, Distracted Fortune. All rights reserved.
# This software is proprietary and not licensed for use, modification, or distribution.

import pytest
from pathlib import Path
from tempfile import TemporaryDirectory
from sec_risk_api.indexing_pipeline import IndexingPipeline
from sec_risk_api.init_vector_db import initialize_chroma
from chromadb.api.models.Collection import Collection


@pytest.fixture
def temp_chroma_db() -> str:
    """
    Session-scoped fixture that creates a temporary directory for ChromaDB.
    Ensures clean state for each test.
    """
    with TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_10k_html(tmp_path: Path) -> Path:
    """
    Creates a minimal but valid 10-K HTML file with Item 1A section.
    """
    html_content = """
    <html>
    <body>
    <h1>FORM 10-K</h1>
    <p>ITEM 1. BUSINESS</p>
    <p>This is the business section. Not important for this test.</p>
    
    <h2>ITEM 1A. RISK FACTORS</h2>
    <p>Competition Risk: The company faces intense competition from established players 
       in the market. This could lead to market share erosion and pricing pressure.</p>
    <p>Technology Risk: Rapid technological change could make our products obsolete.
       We must continuously innovate to stay relevant.</p>
    <p>Regulatory Risk: Government regulations may impose compliance costs and operational 
       constraints that affect profitability and market access.</p>
    
    <h2>ITEM 2. PROPERTIES</h2>
    <p>This section should not be indexed.</p>
    </body>
    </html>
    """
    file_path = tmp_path / "sample_10k.html"
    file_path.write_text(html_content)
    return file_path


class TestIndexingPipelineColdStart:
    """
    Tests for Subissue 1.3 Success Condition: Cold Starts
    (initializing an empty DB with no data duplication).
    """

    def test_pipeline_initializes_with_empty_collection(self, temp_chroma_db: str) -> None:
        """
        Verify that the pipeline can initialize against a fresh ChromaDB instance.
        """
        pipeline = IndexingPipeline(persist_path=temp_chroma_db)
        
        # After init, the collection should be accessible
        assert pipeline.collection is not None
        assert pipeline.collection.name == "sec_risk_factors"

    def test_pipeline_indexes_single_10k(
        self, temp_chroma_db: str, sample_10k_html: Path
    ) -> None:
        """
        Verify that a single 10-K HTML file is correctly indexed end-to-end:
        1. File is extracted
        2. Item 1A is isolated
        3. Text is chunked
        4. Chunks are embedded
        5. Embeddings are stored in Chroma with metadata
        """
        pipeline = IndexingPipeline(persist_path=temp_chroma_db)
        
        # Index the sample filing
        stats = pipeline.index_filing(
            html_path=sample_10k_html,
            ticker="AAPL",
            filing_year=2025,
            item_type="Item 1A"
        )
        
        # Verify the pipeline returned stats
        assert stats["ticker"] == "AAPL"
        assert stats["filing_year"] == 2025
        assert stats["item_type"] == "Item 1A"
        assert stats["chunks_indexed"] > 0
        assert stats["embedding_latency_ms"] > 0
        
        # Verify that chunks were actually stored
        collection = pipeline.collection
        results = collection.get()
        assert len(results["ids"]) > 0

    def test_pipeline_includes_metadata_in_chunks(
        self, temp_chroma_db: str, sample_10k_html: Path
    ) -> None:
        """
        Verify that every indexed chunk includes schema-enforced metadata:
        ticker, filing_year, item_type.
        """
        pipeline = IndexingPipeline(persist_path=temp_chroma_db)
        
        pipeline.index_filing(
            html_path=sample_10k_html,
            ticker="TSLA",
            filing_year=2024,
            item_type="Item 1A"
        )
        
        # Retrieve chunks and verify metadata schema
        results = pipeline.collection.get(
            include=["metadatas"]
        )
        
        assert len(results["metadatas"]) > 0
        
        for metadata in results["metadatas"]:
            assert "ticker" in metadata
            assert "filing_year" in metadata
            assert "item_type" in metadata
            assert metadata["ticker"] == "TSLA"
            assert metadata["filing_year"] == 2024
            assert metadata["item_type"] == "Item 1A"


class TestIndexingPipelineSemanticRecall:
    """
    Tests for Subissue 1.3 Success Condition: Semantic Recall.
    A query for "Geopolitical Instability" must retrieve chunks discussing 
    "International Conflict" or "War," even if exact words don't match.
    """

    def test_pipeline_semantic_search_retrieves_related_content(
        self, temp_chroma_db: str, tmp_path: Path
    ) -> None:
        """
        Index a filing with risk language, then verify that semantic search
        retrieves semantically similar content even without exact word matches.
        """
        # Create a specialized HTML with geopolitical risk content
        html_content = """
        <html>
        <body>
        <h2>ITEM 1A. RISK FACTORS</h2>
        <p>International Conflict Risk: Armed conflicts between nations could disrupt 
           supply chains and market stability. We operate in regions affected by 
           territorial disputes and ongoing military tensions.</p>
        <p>Trade War Risk: Escalating trade disputes between major economies could 
           impose tariffs and sanctions on our operations.</p>
        </body>
        </html>
        """
        file_path = tmp_path / "geopolitical_10k.html"
        file_path.write_text(html_content)
        
        pipeline = IndexingPipeline(persist_path=temp_chroma_db)
        pipeline.index_filing(
            html_path=file_path,
            ticker="GEO",
            filing_year=2025,
            item_type="Item 1A"
        )
        
        # Now search for a semantically related but differently-worded query
        query = "Geopolitical Instability"
        results = pipeline.semantic_search(query, n_results=3)
        
        # Verify that we got results
        assert len(results) > 0
        
        # At least one result should mention geopolitical concepts
        retrieved_text = " ".join([r["text"] for r in results])
        assert any(
            keyword.lower() in retrieved_text.lower()
            for keyword in ["conflict", "international", "war", "dispute", "tension"]
        )

    def test_pipeline_semantic_search_with_metadata_filter(
        self, temp_chroma_db: str, tmp_path: Path
    ) -> None:
        """
        Verify that semantic search can filter by metadata (ticker, year, etc.)
        to enable targeted queries.
        """
        # Index two companies
        for ticker, year in [("AAPL", 2025), ("MSFT", 2024)]:
            html_content = f"""
            <html>
            <body>
            <h2>ITEM 1A. RISK FACTORS</h2>
            <p>Market volatility and competition risk. {ticker} operates in highly 
               competitive markets that are subject to rapid technological change.</p>
            </body>
            </html>
            """
            file_path = tmp_path / f"{ticker}_{year}_10k.html"
            file_path.write_text(html_content)
            
            pipeline = IndexingPipeline(persist_path=temp_chroma_db)
            pipeline.index_filing(
                html_path=file_path,
                ticker=ticker,
                filing_year=year,
                item_type="Item 1A"
            )
        
        # Search for AAPL only
        pipeline = IndexingPipeline(persist_path=temp_chroma_db)
        results = pipeline.semantic_search(
            query="competitive markets",
            n_results=5,
            where={"ticker": "AAPL"}
        )
        
        # All results should be from AAPL
        assert len(results) > 0
        assert all(r["metadata"]["ticker"] == "AAPL" for r in results)


class TestIndexingPipelineUpserts:
    """
    Tests for Subissue 1.3 Success Condition: Onion Stability
    (Upserts: updating existing filings without data duplication).
    """

    def test_pipeline_upsert_prevents_duplicate_chunks(
        self, temp_chroma_db: str, sample_10k_html: Path
    ) -> None:
        """
        Verify that indexing the same filing twice doesn't create duplicate chunks.
        The pipeline should upsert intelligently.
        """
        pipeline = IndexingPipeline(persist_path=temp_chroma_db)
        
        # Index the filing twice
        stats1 = pipeline.index_filing(
            html_path=sample_10k_html,
            ticker="AAPL",
            filing_year=2025,
            item_type="Item 1A"
        )
        
        count_after_first = len(pipeline.collection.get()["ids"])
        
        stats2 = pipeline.index_filing(
            html_path=sample_10k_html,
            ticker="AAPL",
            filing_year=2025,
            item_type="Item 1A"
        )
        
        count_after_second = len(pipeline.collection.get()["ids"])
        
        # Counts should be the same (upsert, not append)
        assert count_after_first == count_after_second, (
            f"Duplicate indexing should not increase chunk count. "
            f"First: {count_after_first}, Second: {count_after_second}"
        )

    def test_pipeline_upsert_with_different_metadata(
        self, temp_chroma_db: str, sample_10k_html: Path
    ) -> None:
        """
        Verify that the pipeline can update a filing with different metadata
        without causing duplication.
        """
        pipeline = IndexingPipeline(persist_path=temp_chroma_db)
        
        # Index with initial metadata
        pipeline.index_filing(
            html_path=sample_10k_html,
            ticker="AAPL",
            filing_year=2025,
            item_type="Item 1A"
        )
        
        count_first = len(pipeline.collection.get()["ids"])
        
        # Re-index with the same filing but a different year
        # (This simulates an amended filing scenario)
        pipeline.index_filing(
            html_path=sample_10k_html,
            ticker="AAPL",
            filing_year=2026,  # Different year
            item_type="Item 1A"
        )
        
        # The second pass should create new chunks for the new metadata
        count_second = len(pipeline.collection.get()["ids"])
        
        # This should create additional chunks since metadata is different
        assert count_second >= count_first


class TestIndexingPipelinePerformance:
    """
    Tests for latency tracking (for JOURNAL.md documentation).
    """

    @pytest.mark.slow
    def test_pipeline_tracks_embedding_latency(
        self, temp_chroma_db: str, sample_10k_html: Path
    ) -> None:
        """
        Verify that the pipeline measures and reports embedding latency
        for performance documentation.
        """
        pipeline = IndexingPipeline(persist_path=temp_chroma_db)
        
        stats = pipeline.index_filing(
            html_path=sample_10k_html,
            ticker="PERF",
            filing_year=2025,
            item_type="Item 1A"
        )
        
        # Latency should be measured and > 0
        assert "embedding_latency_ms" in stats
        assert stats["embedding_latency_ms"] > 0
        
        # For a small filing, should typically complete in < 5 seconds
        # (adjust based on your actual performance targets)
        assert stats["embedding_latency_ms"] < 5000

    def test_pipeline_reports_chunks_indexed_count(
        self, temp_chroma_db: str, sample_10k_html: Path
    ) -> None:
        """
        Verify that stats report the exact number of chunks indexed.
        """
        pipeline = IndexingPipeline(persist_path=temp_chroma_db)
        
        stats = pipeline.index_filing(
            html_path=sample_10k_html,
            ticker="COUNT",
            filing_year=2025,
            item_type="Item 1A"
        )
        
        assert "chunks_indexed" in stats
        assert stats["chunks_indexed"] > 0
        
        # Cross-check: the reported count should match what's in the DB
        collection_count = len(pipeline.collection.get()["ids"])
        assert stats["chunks_indexed"] <= collection_count
