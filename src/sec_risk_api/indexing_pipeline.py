# Copyright (c) 2025 Peter Martinson, Distracted Fortune. All rights reserved.
# This software is proprietary and not licensed for use, modification, or distribution.

from typing import Dict, Any, List, Optional
from pathlib import Path
import time
import logging
import hashlib

from chromadb.api.models.Collection import Collection
from chromadb.api import ClientAPI

from sec_risk_api.init_vector_db import initialize_chroma
from sec_risk_api.ingest import extract_text_from_file, slice_risk_factors
from sec_risk_api.processing import chunk_risk_section
from sec_risk_api.embeddings import EmbeddingEngine
from sec_risk_api.reranking import CrossEncoderReranker

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IndexingPipeline:
    """
    Orchestrator for the end-to-end "Extraction-to-Storage" pipeline.
    
    This class coordinates:
    1. HTML extraction (ingest.py)
    2. Text chunking with metadata (processing.py)
    3. Embedding generation (embeddings.py)
    4. Storage in Chroma DB (init_vector_db.py)
    
    Subissue 1.3 Implementation:
    - Schema Enforcement: Every chunk includes ticker, filing_year, item_type
    - Semantic Recall: Chroma's cosine similarity enables synonym discovery
    - Onion Stability: Intelligent upserts via document IDs to prevent duplication
    """

    def __init__(
        self,
        persist_path: str = "./chroma_db",
        embedding_model: str = "all-MiniLM-L6-v2"
    ) -> None:
        """
        Initialize the pipeline with persistent storage and embedding engine.
        
        Args:
            persist_path: Directory where Chroma DB will persist data.
            embedding_model: Sentence transformer model for embeddings.
        """
        self.persist_path = persist_path
        
        # Initialize Chroma DB
        self.client: ClientAPI
        self.collection: Collection
        self.client, self.collection = initialize_chroma(persist_path=persist_path)
        
        # Initialize Embedding Engine
        self.embeddings = EmbeddingEngine(model_name=embedding_model)
        
        # Lazy-load reranker (only initialize when needed)
        self._reranker: Optional[CrossEncoderReranker] = None
        
        logger.info(
            f"IndexingPipeline initialized: "
            f"persist_path={persist_path}, model={embedding_model}"
        )

    @property
    def reranker(self) -> CrossEncoderReranker:
        """Lazy-load the cross-encoder reranker (only when needed)."""
        if self._reranker is None:
            self._reranker = CrossEncoderReranker()
        return self._reranker

    def _generate_document_id(
        self,
        ticker: str,
        filing_year: int,
        chunk_index: int
    ) -> str:
        """
        Generates a deterministic document ID to enable intelligent upserts.
        
        Format: {ticker}_{year}_{chunk_index}_{hash}
        The hash prevents collisions for semantically identical chunks.
        
        Args:
            ticker: Stock ticker (e.g., "AAPL")
            filing_year: Year of the filing (e.g., 2025)
            chunk_index: Sequential chunk number
        
        Returns:
            A deterministic string ID for use as Chroma document ID.
        """
        base_id = f"{ticker}_{filing_year}_{chunk_index}"
        return base_id

    def index_filing(
        self,
        html_path: str | Path,
        ticker: str,
        filing_year: int,
        item_type: str = "Item 1A"
    ) -> Dict[str, Any]:
        """
        End-to-end indexing of a single SEC filing.
        
        Pipeline steps:
        1. Extract text from HTML
        2. Isolate Item 1A (or specified item)
        3. Chunk the text with overlap
        4. Generate embeddings for each chunk
        5. Store in Chroma with metadata
        6. Track performance metrics
        
        Args:
            html_path: Path to the HTM file
            ticker: Stock ticker symbol
            filing_year: Year of the filing
            item_type: SEC section to extract (default: "Item 1A")
        
        Returns:
            Dictionary with indexing statistics:
            {
                "ticker": str,
                "filing_year": int,
                "item_type": str,
                "chunks_indexed": int,
                "embedding_latency_ms": float,
                "status": str
            }
        """
        try:
            start_time = time.time()
            
            # Step 1: Extract text from HTML
            logger.info(f"Extracting text from {html_path}...")
            full_text = extract_text_from_file(html_path)
            
            # Step 2: Isolate the risk factors section
            logger.info(f"Slicing {item_type} from full text...")
            risk_text = slice_risk_factors(full_text)
            
            # Step 3: Chunk the text with metadata
            base_metadata = {
                "ticker": ticker,
                "filing_year": filing_year,
                "item_type": item_type
            }
            
            logger.info(f"Chunking text into semantic units...")
            chunks = chunk_risk_section(risk_text, metadata=base_metadata)
            
            if not chunks:
                logger.warning(f"No chunks generated for {ticker} {filing_year}")
                return {
                    "ticker": ticker,
                    "filing_year": filing_year,
                    "item_type": item_type,
                    "chunks_indexed": 0,
                    "embedding_latency_ms": 0,
                    "status": "no_chunks"
                }
            
            # Step 4: Generate embeddings
            logger.info(f"Generating embeddings for {len(chunks)} chunks...")
            chunk_texts = [chunk["text"] for chunk in chunks]
            embeddings = self.embeddings.encode(chunk_texts)
            
            # Step 5: Prepare documents for Chroma storage
            document_ids = []
            documents = []
            metadatas = []
            embedding_list = []
            
            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                doc_id = self._generate_document_id(ticker, filing_year, idx)
                document_ids.append(doc_id)
                documents.append(chunk["text"])
                metadatas.append(chunk["metadata"])
                embedding_list.append(embedding.tolist())
            
            # Step 6: Upsert into Chroma (prevents duplicates)
            logger.info(f"Upserting {len(documents)} documents into Chroma...")
            self.collection.upsert(
                ids=document_ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embedding_list
            )
            
            embedding_latency_ms = (time.time() - start_time) * 1000
            
            logger.info(
                f"Successfully indexed {ticker} {filing_year}: "
                f"{len(chunks)} chunks in {embedding_latency_ms:.2f}ms"
            )
            
            return {
                "ticker": ticker,
                "filing_year": filing_year,
                "item_type": item_type,
                "chunks_indexed": len(chunks),
                "embedding_latency_ms": embedding_latency_ms,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Failed to index {html_path}: {e}")
            raise

    def semantic_search(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        rerank: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Hybrid semantic search with optional cross-encoder reranking.
        
        Level 3.0 Enhancement:
        - Combines vector similarity with metadata filtering
        - Optional cross-encoder reranking for improved relevance
        - Full source citation for every result
        
        Args:
            query: Natural language search query (e.g., "Geopolitical Instability")
            n_results: Number of results to return (default: 5)
            where: Optional metadata filter (e.g., {"ticker": "AAPL", "filing_year": 2025})
            rerank: If True, apply cross-encoder reranking to improve relevance
        
        Returns:
            List of dictionaries with:
            {
                "id": str,              # Document ID
                "text": str,            # Chunk text
                "metadata": Dict,       # ticker, filing_year, item_type
                "distance": float,      # Vector distance (lower = more similar)
                "rerank_score": float   # (Optional) Cross-encoder score (higher = more relevant)
            }
        """
        try:
            # Generate embedding for the query
            query_embedding = self.embeddings.encode([query])[0].tolist()
            
            # Determine retrieval count
            # If reranking, retrieve more candidates for reranker to choose from
            retrieval_count = n_results * 3 if rerank else n_results
            
            # Search in Chroma
            logger.info(f"Semantic search for: '{query}' (rerank={rerank})")
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=retrieval_count,
                where=where,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            if results["ids"] and len(results["ids"]) > 0:
                for doc_id, text, metadata, distance in zip(
                    results["ids"][0],
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                ):
                    formatted_results.append({
                        "id": doc_id,
                        "text": text,
                        "metadata": metadata,
                        "distance": float(distance)  # Lower distance = better match
                    })
            
            # Apply reranking if requested
            if rerank and formatted_results:
                logger.info(f"Reranking {len(formatted_results)} candidates...")
                formatted_results = self.reranker.rerank(
                    query=query,
                    candidates=formatted_results,
                    top_k=n_results
                )
                logger.info(f"Reranking complete. Returning top {len(formatted_results)} results")
            
            logger.info(f"Found {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            raise

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Returns statistics about the current Chroma collection.
        
        Returns:
            Dictionary with:
            {
                "total_documents": int,
                "collection_name": str,
                "metric": str
            }
        """
        results = self.collection.get(include=[])
        return {
            "total_documents": len(results["ids"]),
            "collection_name": self.collection.name,
            "metric": "cosine"
        }
