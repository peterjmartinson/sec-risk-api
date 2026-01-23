#!/usr/bin/env python3
"""
CLI script to analyze SEC filings.

Usage:
    python analyze_filing.py data/filings/tsla_2024_10k.html TSLA 2024
"""

import sys
from pathlib import Path
from sigmak.integration import IntegrationPipeline
from sigmak.risk_classification_service import RiskClassificationService
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def main():
    if len(sys.argv) != 4:
        print("Usage: python analyze_filing.py <html_path> <ticker> <year>")
        print("Example: python analyze_filing.py data/sample_10k.html AAPL 2025")
        sys.exit(1)
    
    html_path = sys.argv[1]
    ticker = sys.argv[2].upper()
    year = int(sys.argv[3])
    
    # Initialize pipeline
    print(f"\n🔍 Initializing analysis pipeline...")
    pipeline = IntegrationPipeline(persist_path="./chroma_db")

    # Classification helper: prefer RiskClassificationService when GOOGLE_API_KEY set
    classification_service = None
    use_full_service = False
    if os.getenv('GOOGLE_API_KEY'):
        classification_service = RiskClassificationService(drift_system=pipeline.drift_system)
        use_full_service = True
    
    # Analyze filing
    print(f"📊 Analyzing {ticker} {year} from {html_path}...")
    result = pipeline.analyze_filing(
        html_path=html_path,
        ticker=ticker,
        filing_year=year,
        retrieve_top_k=5  # Start with top 5 risks
    )
    
    # Print results
    print(f"\n✅ Analysis Complete!")
    print(f"   • Chunks Indexed: {result.metadata['chunks_indexed']}")
    print(f"   • Risks Analyzed: {result.metadata['risks_analyzed']}")
    print(f"   • Total Time: {result.metadata['total_latency_ms']:.0f}ms")
    
    print(f"\n📋 Top Risks:\n")
    for i, risk in enumerate(result.risks, 1):
        print(f"━━━ Risk #{i} ━━━")
        print(f"Severity: {risk['severity']['value']:.2f} - {risk['severity']['explanation']}")
        print(f"Novelty:  {risk['novelty']['value']:.2f} - {risk['novelty']['explanation']}")
        print(f"Text: {risk['text'][:200]}...")
        print()

    # Enrich risks with similarity-first classification (vector DB -> LLM)
    print("\n🔎 Classifying identified risks using local vector DB (with LLM fallback)...")
    for i, risk in enumerate(result.risks, 1):
        try:
            # Skip if already classified in metadata
            if risk.get('metadata', {}).get('category'):
                risk['category'] = risk['metadata']['category']
                risk['category_confidence'] = risk['metadata'].get('confidence', 0.0)
                risk['classification_method'] = risk['metadata'].get('classification_source', 'vector')
                continue

            if use_full_service and classification_service:
                llm_result, source = classification_service.classify_with_cache_first(risk['text'])
                risk['category'] = llm_result.category.value
                risk['category_confidence'] = llm_result.confidence
                risk['classification_method'] = source
                print(f"Risk #{i}: classified as {risk['category']} (source={source})")
            else:
                # No API key: perform cache-only lookup against DriftDetectionSystem
                try:
                    embedding = pipeline.indexing_pipeline.embeddings.encode([risk['text']])[0].tolist()
                    cache_results = pipeline.drift_system.similarity_search(query_embedding=embedding, n_results=1)
                    if cache_results and cache_results[0].get('similarity_score', 0.0) >= 0.8:
                        top = cache_results[0]
                        risk['category'] = top.get('category', 'UNCATEGORIZED')
                        risk['category_confidence'] = float(top.get('confidence', 0.0))
                        risk['classification_method'] = 'vector_db'
                        print(f"Risk #{i}: classified from vector DB as {risk['category']}")
                    else:
                        risk['category'] = 'UNCATEGORIZED'
                        risk['category_confidence'] = 0.0
                        risk['classification_method'] = 'db_only_no_match'
                        print(f"Risk #{i}: no cached classification found; marked UNCATEGORIZED")
                except Exception as e:
                    print(f"Risk #{i}: cache-only classification failed: {e}")
                    risk['category'] = 'UNCATEGORIZED'
                    risk['category_confidence'] = 0.0
                    risk['classification_method'] = 'error'
        except Exception as e:
            print(f"Risk #{i}: classification failed: {e}")
    
    # Save to JSON
    output_file = f"output/results_{ticker}_{year}.json"
    with open(output_file, 'w') as f:
        f.write(result.to_json())
    print(f"💾 Full results saved to: {output_file}")

if __name__ == "__main__":
    main()
