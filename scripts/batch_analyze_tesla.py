#!/usr/bin/env python3
"""
Batch analyze multiple Tesla filings to build historical context.

Usage:
    python batch_analyze_tesla.py
"""

from pathlib import Path
from sigmak.integration import IntegrationPipeline

def main():
    # Define Tesla filings
    filings = [
        ("data/filings/fc-20230831x10k.html", "FC", 2023),
        ("data/filings/fc-20240831x10k.html", "FC", 2024),
        ("data/filings/fc-20250831x10k.html", "FC", 2025),
    ]
    
    # Initialize pipeline once
    print("🔍 Initializing analysis pipeline...")
    pipeline = IntegrationPipeline(persist_path="./chroma_db")
    
    # Process each filing
    for html_path, ticker, year in filings:
        print(f"\n{'='*60}")
        print(f"📊 Analyzing {ticker} {year}")
        print(f"{'='*60}")
        
        if not Path(html_path).exists():
            print(f"⚠️  File not found: {html_path}")
            continue
        
        try:
            result = pipeline.analyze_filing(
                html_path=html_path,
                ticker=ticker,
                filing_year=year,
                retrieve_top_k=5
            )
            
            print(f"\n✅ Analysis Complete!")
            print(f"   • Chunks Indexed: {result.metadata['chunks_indexed']}")
            print(f"   • Risks Analyzed: {result.metadata['risks_analyzed']}")
            print(f"   • Time: {result.metadata['total_latency_ms']:.0f}ms")
            
            # Show top 3 risks
            print(f"\n📋 Top 3 Risks:")
            for i, risk in enumerate(result.risks[:3], 1):
                print(f"\n  {i}. Severity: {risk['severity']['value']:.2f} | Novelty: {risk['novelty']['value']:.2f}")
                print(f"     {risk['text'][:150]}...")
            
            # Save results
            output_file = f"results_{ticker}_{year}.json"
            with open(output_file, 'w') as f:
                f.write(result.to_json())
            print(f"\n💾 Saved: {output_file}")
            
        except Exception as e:
            print(f"❌ Error analyzing {ticker} {year}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print("🎉 Batch analysis complete!")
    print("{'='*60}")
    
    # Show what's now in the database
    print("\n📊 Database Summary:")
    try:
        # Query all Tesla chunks
        all_tesla = pipeline.indexing_pipeline.semantic_search(
            query="risk",
            n_results=100,
            where={"ticker": "FC"}
        )
        
        # Count by year
        year_counts = {}
        for chunk in all_tesla:
            year = chunk["metadata"]["filing_year"]
            year_counts[year] = year_counts.get(year, 0) + 1
        
        for year in sorted(year_counts.keys()):
            print(f"   • {year}: {year_counts[year]} chunks")
        
    except Exception as e:
        print(f"   Could not retrieve summary: {e}")

if __name__ == "__main__":
    main()
