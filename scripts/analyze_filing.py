#!/usr/bin/env python3
"""
CLI script to analyze SEC filings.

Usage:
    python analyze_filing.py data/filings/tsla_2024_10k.html TSLA 2024
"""

import sys
from pathlib import Path
from sigmak.integration import IntegrationPipeline

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
    
    # Save to JSON
    output_file = f"results_{ticker}_{year}.json"
    with open(output_file, 'w') as f:
        f.write(result.to_json())
    print(f"💾 Full results saved to: {output_file}")

if __name__ == "__main__":
    main()
