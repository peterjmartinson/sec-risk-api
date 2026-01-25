Below is a conversation I had with a Claude Sonnet 4.5 model.  It will be implemented with a smaller, less capable model.  It will also be implemented over three branched iterations.  The issue text for each of these iterations are stored next to this file, in ISSUE84_PEER_DISCOVERY.md, ISSUE83_PEER_DOWNLOAD.md, ISSUE82_ANALYSIS_WITH_PEERS.md.

User said: A big issue is, I need to create a comparison between a target company's risk report (scripts/generate_yoy_report.py for example) and other similar companies in the same industry or market.  Can you help me figure out how we're gonna do this?
27 references
Copilot said: I'll help you design a comparison system
I'll help you design a comparison system for analyzing a target company's risk profile against industry peers. Let me first examine the current codebase structure to understand what we're working with.Excellent question! Building peer comparison and industry benchmarking is a natural evolution for SigmaK. Based on your existing architecture, here's a structured approach to tackle this challenge:

🎯 Strategic Approach: Industry Comparison Layer
The Goal
Enable analysts to answer questions like:

"How does HURC's risk profile compare to its industry peers?"
"What operational risks are HURC facing that competitors aren't?"
"Which companies in the semiconductor sector have the highest regulatory risk exposure?"
🏗️ Architecture Design
Phase 1: Industry Taxonomy & Metadata (Foundation)
1. Extend Filing Metadata with Industry Classification

Python
# src/sigmak/industry_taxonomy.py
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict

class IndustryCategory(str, Enum):
    TECHNOLOGY = "technology"
    SEMICONDUCTOR = "semiconductor"
    AUTOMOTIVE = "automotive"
    CONSUMER_GOODS = "consumer_goods"
    ENERGY = "energy"
    HEALTHCARE = "healthcare"
    FINANCIAL = "financial"
    # ... expand as needed

@dataclass
class IndustryMetadata:
    """Industry classification for peer grouping."""
    primary_industry: IndustryCategory
    sub_industries: List[str]
    market_cap_tier: str  # "mega", "large", "mid", "small"
    geography: str  # "global", "us", "emea", "apac"
2. Augment Chunk Metadata

Update your indexing pipeline to include industry tags:

Python
# In IndexingPipeline.index_filing()
chunk_metadata = {
    "ticker": ticker,
    "filing_year": filing_year,
    "chunk_index": chunk_index,
    "industry": self._get_industry(ticker),  # NEW
    "market_cap": self._get_market_cap(ticker),  # NEW
    # ... existing fields
}
3. Industry Lookup Service

Python
# src/sigmak/industry_service.py
class IndustryService:
    """Map tickers to industry classifications."""
    
    def __init__(self, mapping_file: str = "data/industry_mapping.json"):
        self.mapping = self._load_mapping(mapping_file)
    
    def get_industry(self, ticker: str) -> IndustryCategory:
        """Return primary industry for a ticker."""
        return self.mapping.get(ticker, IndustryCategory.OTHER)
    
    def get_peers(self, ticker: str, max_peers: int = 10) -> List[str]:
        """Return peer tickers in the same industry."""
        industry = self.get_industry(ticker)
        return [t for t, ind in self.mapping.items() 
                if ind == industry and t != ticker][:max_peers]
Phase 2: Peer Comparison Pipeline (Core Functionality)
1. Add Peer Retrieval to Integration Pipeline

Python
# src/sigmak/peer_comparison.py
from typing import List, Dict, Any
from dataclasses import dataclass
from sigmak.integration import RiskAnalysisResult

@dataclass
class PeerComparisonResult:
    """Results of comparing target company to peers."""
    target_ticker: str
    target_result: RiskAnalysisResult
    peer_results: List[RiskAnalysisResult]
    
    # Comparative metrics
    severity_percentile: float  # Where target ranks vs peers
    novelty_percentile: float
    category_divergence: Dict[str, float]  # Categories where target differs
    unique_risks: List[Dict[str, Any]]  # Risks target has that peers don't
    shared_risks: List[Dict[str, Any]]  # Common industry risks

class PeerComparisonPipeline:
    def __init__(
        self,
        integration_pipeline: IntegrationPipeline,
        industry_service: IndustryService
    ):
        self.pipeline = integration_pipeline
        self.industry_service = industry_service
    
    def compare_to_peers(
        self,
        target_ticker: str,
        target_year: int,
        peer_tickers: List[str] = None,
        max_peers: int = 5
    ) -> PeerComparisonResult:
        """Compare target company's risks to industry peers."""
        
        # 1. Get target company's analysis
        target_result = self._get_or_analyze(target_ticker, target_year)
        
        # 2. Get peer list if not provided
        if peer_tickers is None:
            peer_tickers = self.industry_service.get_peers(
                target_ticker, max_peers
            )
        
        # 3. Analyze each peer
        peer_results = []
        for peer in peer_tickers:
            try:
                result = self._get_or_analyze(peer, target_year)
                peer_results.append(result)
            except FileNotFoundError:
                logger.warning(f"No filing found for {peer} {target_year}")
        
        # 4. Calculate comparative metrics
        return self._calculate_comparison(target_result, peer_results)
    
    def _calculate_comparison(
        self,
        target: RiskAnalysisResult,
        peers: List[RiskAnalysisResult]
    ) -> PeerComparisonResult:
        """Calculate percentile rankings and divergence."""
        
        # Severity percentile
        all_severities = [
            r['severity']['value'] 
            for result in peers 
            for r in result.risks
        ]
        target_avg_severity = np.mean([
            r['severity']['value'] for r in target.risks
        ])
        severity_percentile = self._calculate_percentile(
            target_avg_severity, all_severities
        )
        
        # Category divergence: identify categories where target
        # has significantly higher concentration than peers
        category_divergence = self._calculate_category_divergence(
            target, peers
        )
        
        # Unique vs shared risks
        unique_risks = self._find_unique_risks(target, peers)
        shared_risks = self._find_shared_risks(target, peers)
        
        return PeerComparisonResult(
            target_ticker=target.ticker,
            target_result=target,
            peer_results=peers,
            severity_percentile=severity_percentile,
            novelty_percentile=...,  # Similar calculation
            category_divergence=category_divergence,
            unique_risks=unique_risks,
            shared_risks=shared_risks
        )
2. Category Divergence Analysis

Python
def _calculate_category_divergence(
    self,
    target: RiskAnalysisResult,
    peers: List[RiskAnalysisResult]
) -> Dict[str, float]:
    """
    Identify categories where target's concentration differs from peers.
    
    Returns:
        Dict mapping category -> z-score of divergence
    """
    from collections import defaultdict
    
    # Count category distributions for target
    target_dist = defaultdict(int)
    for risk in target.risks:
        cat = risk.get('category', 'UNCATEGORIZED')
        target_dist[cat] += 1
    
    # Normalize to percentages
    total_target = len(target.risks)
    target_pcts = {
        cat: count / total_target 
        for cat, count in target_dist.items()
    }
    
    # Calculate peer averages
    peer_dists = []
    for peer in peers:
        peer_dist = defaultdict(int)
        for risk in peer.risks:
            cat = risk.get('category', 'UNCATEGORIZED')
            peer_dist[cat] += 1
        total_peer = len(peer.risks)
        peer_pcts = {
            cat: count / total_peer 
            for cat, count in peer_dist.items()
        }
        peer_dists.append(peer_pcts)
    
    # Calculate z-scores for each category
    divergence = {}
    for cat in set(target_pcts.keys()).union(
        *[set(d.keys()) for d in peer_dists]
    ):
        target_val = target_pcts.get(cat, 0.0)
        peer_vals = [d.get(cat, 0.0) for d in peer_dists]
        
        mean_peer = np.mean(peer_vals)
        std_peer = np.std(peer_vals) + 1e-6  # Avoid div by zero
        
        z_score = (target_val - mean_peer) / std_peer
        divergence[cat] = z_score
    
    return divergence
Phase 3: Enhanced Report Generation
Update generate_yoy_report.py to include peer comparison:

Python
def generate_markdown_report_with_peers(
    ticker: str,
    results: List[RiskAnalysisResult],
    peer_comparison: PeerComparisonResult,
    output_file: str
) -> None:
    """Generate report with peer benchmarking section."""
    
    # ... existing report sections ...
    
    # NEW: Peer Comparison Section
    report_lines.append("## Peer Benchmarking")
    report_lines.append("")
    report_lines.append(f"**Industry Peers**: {', '.join([p.ticker for p in peer_comparison.peer_results])}")
    report_lines.append("")
    
    # Severity ranking
    severity_pct = peer_comparison.severity_percentile
    report_lines.append(f"**Severity Ranking**: {ticker} ranks at the **{severity_pct:.0f}th percentile** among peers")
    if severity_pct >= 75:
        report_lines.append("🔴 **ALERT**: Significantly higher risk profile than industry average")
    elif severity_pct <= 25:
        report_lines.append("🟢 **STRENGTH**: Below-average risk exposure vs peers")
    report_lines.append("")
    
    # Category divergence
    report_lines.append("### Risk Category Divergence")
    report_lines.append("")
    report_lines.append("| Category | Target % | Peer Avg % | Z-Score |")
    report_lines.append("|----------|----------|------------|---------|")
    
    for cat, z_score in sorted(
        peer_comparison.category_divergence.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:5]:  # Top 5 divergent categories
        target_pct = ...  # Calculate from target_result
        peer_avg = ...    # Calculate from peer_results
        
        flag = "🔴" if abs(z_score) >= 2.0 else "🟡" if abs(z_score) >= 1.0 else ""
        report_lines.append(f"| {cat} | {target_pct:.1f}% | {peer_avg:.1f}% | {z_score:+.2f} {flag} |")
    
    report_lines.append("")
    report_lines.append("_Z-score > 2.0 indicates significant divergence from peer average_")
    report_lines.append("")
    
    # Unique risks
    if peer_comparison.unique_risks:
        report_lines.append("### Unique Risk Exposures")
        report_lines.append("")
        report_lines.append(f"{ticker} discloses the following risks not commonly seen in peer filings:")
        report_lines.append("")
        
        for risk in peer_comparison.unique_risks[:5]:
            text_preview = risk['text'][:150]
            category = risk.get('category', 'UNCATEGORIZED')
            report_lines.append(f"- **[{category}]** {text_preview}...")
        
        report_lines.append("")
Phase 4: Script Integration
Create a new script: scripts/generate_peer_comparison_report.py

Python
#!/usr/bin/env python3
"""
Generate peer comparison risk analysis report.

Usage:
    python scripts/generate_peer_comparison_report.py HURC 2025
    python scripts/generate_peer_comparison_report.py HURC 2025 --peers INTC AMD NVDA
"""

import argparse
from sigmak.integration import IntegrationPipeline
from sigmak.peer_comparison import PeerComparisonPipeline
from sigmak.industry_service import IndustryService

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ticker', help='Target company ticker')
    parser.add_argument('year', type=int, help='Filing year')
    parser.add_argument('--peers', nargs='+', help='Explicit peer list (optional)')
    parser.add_argument('--max-peers', type=int, default=5, help='Max number of peers')
    
    args = parser.parse_args()
    
    # Initialize pipelines
    integration_pipeline = IntegrationPipeline(persist_path="./chroma_db")
    industry_service = IndustryService(mapping_file="data/industry_mapping.json")
    peer_pipeline = PeerComparisonPipeline(integration_pipeline, industry_service)
    
    # Run comparison
    result = peer_pipeline.compare_to_peers(
        target_ticker=args.ticker,
        target_year=args.year,
        peer_tickers=args.peers,
        max_peers=args.max_peers
    )
    
    # Generate report
    output_file = f"output/{args.ticker}_Peer_Comparison_{args.year}.md"
    generate_markdown_report_with_peers(
        ticker=args.ticker,
        results=[result.target_result],
        peer_comparison=result,
        output_file=output_file
    )

if __name__ == "__main__":
    main()
📋 Implementation Checklist
Immediate Next Steps
Create Issue: "Add peer comparison and industry benchmarking"
Build Industry Mapping:
Create data/industry_mapping.json with ticker → industry mappings
Start with companies you already have filings for (HURC, FC, etc.)
Implement IndustryService:
Write unit tests first (tests/test_industry_service.py)
Implement lookup and peer discovery logic
Extend Indexing Pipeline:
Add industry field to chunk metadata
Backfill existing ChromaDB entries (migration script)
Build PeerComparisonPipeline:
Implement comparison metrics (percentiles, divergence)
Write unit tests with mock peer data
Update Report Generation:
Add peer benchmarking section to Markdown reports
Include z-score visualizations and unique risk callouts
Data Requirements
Industry Mapping File (data/industry_mapping.json):

JSON
{
  "HURC": {
    "primary_industry": "semiconductor",
    "sub_industries": ["mining", "materials"],
    "market_cap_tier": "mid",
    "peers": ["ALB", "SQM", "LTHM"]
  },
  ...
}
Peer Filing Coverage: Ensure you have filings for peer companies (use your downloader to batch-fetch)

🚀 Advanced Enhancements (Future Phases)
Dynamic Peer Discovery: Use SEC's company_tickers.json + SIC codes to auto-discover peers
Time-Series Comparison: Track how peer rankings evolve over multiple years
Risk Correlation Analysis: Identify which risks co-occur across industry (e.g., "supply chain + geopolitical" in semiconductors)
API Endpoint: POST /compare_peers to expose this via REST API
💡 Recommendation
Start with Phase 1 + Phase 2 to get the core functionality working. This will let you:

Answer "How does HURC compare to peers?" questions
Identify unique vs shared risks
Generate investment-grade comparative reports
Would you like me to:

Create a GitHub issue for this feature with the full spec?
Generate the test file first (tests/test_peer_comparison.py)?
Build the industry mapping file for your existing tickers?
Let me know how you'd like to proceed! 🎯

You said: one thing - I'm not downloading filings
one thing - I'm not downloading filings for every company right now.  I want to pick a target, like NVDA, and run a report for that one.  So, I'll want some kind of automated way to ALSO identify, download, and run analysis on filings for the industry peers.
Copilot said: Perfect! Let's design an **end-to-end automated peer
Perfect! Let's design an end-to-end automated peer comparison workflow that handles everything from peer discovery to report generation.

🎯 Automated Peer Comparison Workflow
High-Level Flow
Code
User Input: "NVDA, 2025"
    ↓
1. Identify Peers (via SIC code lookup)
    ↓
2. Download Target + Peer Filings (batch download)
    ↓
3. Analyze All Filings (parallel processing)
    ↓
4. Generate Peer Comparison Report
🏗️ Architecture Design
Component 1: Automated Peer Discovery
Use SEC's SIC (Standard Industrial Classification) codes to find similar companies:

Python
# src/sigmak/peer_discovery.py
"""
Automated peer discovery using SEC EDGAR data.

This module uses SIC codes and market cap to identify comparable companies.
"""

import requests
import json
from typing import List, Dict, Optional
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class CompanyInfo:
    """Company metadata from SEC."""
    ticker: str
    cik: str
    name: str
    sic: str  # Standard Industrial Classification code
    sic_description: str
    market_cap_tier: Optional[str] = None  # Estimated tier

class PeerDiscoveryService:
    """Discover peer companies using SEC EDGAR data."""
    
    SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
    SEC_CIK_LOOKUP_URL = "https://data.sec.gov/submissions/CIK{cik_padded}.json"
    
    def __init__(self, cache_dir: str = "./data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.company_tickers = self._load_company_tickers()
    
    def _load_company_tickers(self) -> Dict:
        """Load SEC's company tickers JSON (cached)."""
        cache_file = self.cache_dir / "company_tickers.json"
        
        if cache_file.exists():
            logger.info("Loading company tickers from cache")
            with open(cache_file) as f:
                return json.load(f)
        
        logger.info("Downloading company tickers from SEC...")
        headers = {"User-Agent": "SigmaK Risk Analysis info@example.com"}
        response = requests.get(self.SEC_COMPANY_TICKERS_URL, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        
        # Cache for future use
        with open(cache_file, 'w') as f:
            json.dump(data, f)
        
        return data
    
    def get_company_info(self, ticker: str) -> CompanyInfo:
        """
        Get company info including SIC code.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            CompanyInfo with SIC code and metadata
        """
        # Find company in tickers index
        ticker_upper = ticker.upper()
        company_entry = None
        
        for entry in self.company_tickers.values():
            if entry.get('ticker', '').upper() == ticker_upper:
                company_entry = entry
                break
        
        if not company_entry:
            raise ValueError(f"Ticker {ticker} not found in SEC database")
        
        cik = str(company_entry['cik_str']).zfill(10)
        
        # Fetch detailed company data from SEC submissions endpoint
        cache_file = self.cache_dir / f"cik_{cik}.json"
        
        if cache_file.exists():
            with open(cache_file) as f:
                company_data = json.load(f)
        else:
            logger.info(f"Fetching company data for CIK {cik}...")
            headers = {"User-Agent": "SigmaK Risk Analysis info@example.com"}
            url = self.SEC_CIK_LOOKUP_URL.format(cik_padded=cik)
            
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            company_data = response.json()
            
            # Cache
            with open(cache_file, 'w') as f:
                json.dump(company_data, f)
        
        # Extract SIC info
        sic = company_data.get('sic', 'Unknown')
        sic_description = company_data.get('sicDescription', 'Unknown')
        
        return CompanyInfo(
            ticker=ticker_upper,
            cik=cik,
            name=company_data.get('name', company_entry['title']),
            sic=sic,
            sic_description=sic_description
        )
    
    def find_peers(
        self,
        ticker: str,
        max_peers: int = 10,
        same_sic_only: bool = True
    ) -> List[CompanyInfo]:
        """
        Find peer companies based on SIC code.
        
        Args:
            ticker: Target company ticker
            max_peers: Maximum number of peers to return
            same_sic_only: If True, only return exact SIC matches
            
        Returns:
            List of CompanyInfo for peer companies
        """
        target_info = self.get_company_info(ticker)
        target_sic = target_info.sic
        
        logger.info(
            f"Finding peers for {ticker} (SIC: {target_sic} - {target_info.sic_description})"
        )
        
        # Find all companies with same SIC
        peer_candidates = []
        for entry in self.company_tickers.values():
            candidate_ticker = entry.get('ticker', '')
            if candidate_ticker.upper() == ticker.upper():
                continue  # Skip target company
            
            try:
                # Check if we have cached data (avoid rate limiting)
                cik = str(entry['cik_str']).zfill(10)
                cache_file = self.cache_dir / f"cik_{cik}.json"
                
                if cache_file.exists():
                    with open(cache_file) as f:
                        company_data = json.load(f)
                    
                    candidate_sic = company_data.get('sic', '')
                    
                    if same_sic_only and str(candidate_sic) == str(target_sic):
                        peer_candidates.append(
                            CompanyInfo(
                                ticker=candidate_ticker.upper(),
                                cik=cik,
                                name=company_data.get('name', entry['title']),
                                sic=candidate_sic,
                                sic_description=company_data.get('sicDescription', '')
                            )
                        )
                
            except Exception as e:
                logger.debug(f"Skipping {candidate_ticker}: {e}")
                continue
        
        logger.info(f"Found {len(peer_candidates)} potential peers")
        
        # Return top N peers (you could add ranking logic here)
        return peer_candidates[:max_peers]
    
    def enrich_peers_with_sic(
        self,
        target_ticker: str,
        max_peers: int = 10,
        fetch_missing: bool = True
    ) -> List[CompanyInfo]:
        """
        Find peers and optionally fetch their SIC data if not cached.
        
        Args:
            target_ticker: Target company
            max_peers: Max number of peers
            fetch_missing: If True, fetch SIC data for uncached companies
            
        Returns:
            List of peer companies with SIC codes
        """
        import time
        
        if not fetch_missing:
            return self.find_peers(target_ticker, max_peers)
        
        # Get target SIC first
        target_info = self.get_company_info(target_ticker)
        target_sic = target_info.sic
        
        logger.info("Enriching company database with SIC codes (this may take a few minutes)...")
        
        # Fetch SIC codes for all companies (rate-limited)
        fetched = 0
        for i, entry in enumerate(self.company_tickers.values()):
            cik = str(entry['cik_str']).zfill(10)
            cache_file = self.cache_dir / f"cik_{cik}.json"
            
            if not cache_file.exists():
                try:
                    headers = {"User-Agent": "SigmaK Risk Analysis info@example.com"}
                    url = self.SEC_CIK_LOOKUP_URL.format(cik_padded=cik)
                    
                    response = requests.get(url, headers=headers)
                    response.raise_for_status()
                    company_data = response.json()
                    
                    with open(cache_file, 'w') as f:
                        json.dump(company_data, f)
                    
                    fetched += 1
                    
                    # SEC rate limit: 10 requests/second
                    time.sleep(0.12)
                    
                    if fetched % 50 == 0:
                        logger.info(f"Fetched {fetched} company records...")
                
                except Exception as e:
                    logger.warning(f"Failed to fetch CIK {cik}: {e}")
                    continue
        
        logger.info(f"Enrichment complete. Fetched {fetched} new records.")
        
        # Now find peers with full SIC data
        return self.find_peers(target_ticker, max_peers, same_sic_only=True)
Component 2: Orchestration Script
Create scripts/generate_peer_comparison_report.py:

Python
#!/usr/bin/env python3
"""
Automated peer comparison workflow.

This script:
1. Discovers peers using SIC codes
2. Downloads filings for target + peers
3. Analyzes all filings
4. Generates comparative report

Usage:
    # Auto-discover peers
    python scripts/generate_peer_comparison_report.py NVDA 2024
    
    # Manual peer list
    python scripts/generate_peer_comparison_report.py NVDA 2024 --peers AMD INTC
    
    # Limit number of auto-discovered peers
    python scripts/generate_peer_comparison_report.py NVDA 2024 --max-peers 5
"""

import argparse
import sys
from pathlib import Path
from typing import List
import logging

from sigmak.downloads.tenk_downloader import TenKDownloader
from sigmak.peer_discovery import PeerDiscoveryService
from sigmak.integration import IntegrationPipeline
from sigmak.peer_comparison import PeerComparisonPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Automated peer comparison analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('ticker', help='Target company ticker')
    parser.add_argument('year', type=int, help='Filing year')
    parser.add_argument(
        '--peers',
        nargs='+',
        help='Explicit peer list (skips auto-discovery)'
    )
    parser.add_argument(
        '--max-peers',
        type=int,
        default=5,
        help='Max number of auto-discovered peers (default: 5)'
    )
    parser.add_argument(
        '--download-dir',
        default='./data/filings',
        help='Directory for downloaded filings'
    )
    parser.add_argument(
        '--db-path',
        default='./database/sec_filings.db',
        help='Path to filings database'
    )
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip download step (use existing filings)'
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"🔍 Automated Peer Comparison: {args.ticker} {args.year}")
    print(f"{'='*60}\n")
    
    # Step 1: Discover peers (if not provided)
    if args.peers:
        peer_tickers = [p.upper() for p in args.peers]
        print(f"📋 Using provided peers: {', '.join(peer_tickers)}")
    else:
        print(f"🔎 Discovering peers for {args.ticker}...")
        peer_service = PeerDiscoveryService()
        
        try:
            target_info = peer_service.get_company_info(args.ticker)
            print(f"   Target: {target_info.name}")
            print(f"   Industry: {target_info.sic_description} (SIC: {target_info.sic})")
            print(f"   Searching for similar companies...")
            
            # Find peers (may need to enrich database)
            peers = peer_service.find_peers(
                args.ticker,
                max_peers=args.max_peers
            )
            
            if not peers:
                print(f"\n⚠️  No cached peers found. Enriching database (this may take a few minutes)...")
                peers = peer_service.enrich_peers_with_sic(
                    args.ticker,
                    max_peers=args.max_peers,
                    fetch_missing=True
                )
            
            if not peers:
                print(f"\n❌ Could not find any peers for {args.ticker}")
                sys.exit(1)
            
            peer_tickers = [p.ticker for p in peers]
            print(f"\n✅ Found {len(peer_tickers)} peers:")
            for peer in peers:
                print(f"   • {peer.ticker} - {peer.name}")
        
        except Exception as e:
            print(f"\n❌ Peer discovery failed: {e}")
            sys.exit(1)
    
    # Step 2: Download filings
    all_tickers = [args.ticker.upper()] + peer_tickers
    
    if not args.skip_download:
        print(f"\n{'='*60}")
        print(f"📥 Downloading 10-K Filings")
        print(f"{'='*60}\n")
        
        downloader = TenKDownloader(
            db_path=args.db_path,
            download_dir=args.download_dir
        )
        
        for ticker in all_tickers:
            try:
                print(f"Downloading {ticker} {args.year}...")
                records = downloader.download_10k(
                    ticker=ticker,
                    years=1,  # Just the target year
                    force_refresh=False
                )
                
                if records:
                    record = records[0]
                    print(f"   ✅ {record.local_path}")
                else:
                    print(f"   ⚠️  No filing found for {ticker} {args.year}")
            
            except Exception as e:
                print(f"   ❌ Download failed: {e}")
                continue
    else:
        print(f"\n⏭️  Skipping download (using existing filings)")
    
    # Step 3: Analyze all filings
    print(f"\n{'='*60}")
    print(f"🔬 Analyzing Risk Disclosures")
    print(f"{'='*60}\n")
    
    pipeline = IntegrationPipeline(persist_path="./chroma_db")
    
    results = []
    failed_tickers = []
    
    for ticker in all_tickers:
        # Find the filing file
        filing_path = find_filing_path(ticker, args.year, args.download_dir)
        
        if not filing_path:
            print(f"⚠️  No filing found for {ticker} {args.year}")
            failed_tickers.append(ticker)
            continue
        
        try:
            print(f"Analyzing {ticker} {args.year}...")
            result = pipeline.analyze_filing(
                html_path=str(filing_path),
                ticker=ticker,
                filing_year=args.year,
                retrieve_top_k=10,
                rerank=True
            )
            results.append(result)
            print(f"   ✅ {len(result.risks)} risks analyzed")
        
        except Exception as e:
            print(f"   ❌ Analysis failed: {e}")
            failed_tickers.append(ticker)
            continue
    
    if not results or len(results) < 2:
        print(f"\n❌ Need at least 2 analyzed filings for comparison")
        print(f"   Analyzed: {len(results)}")
        print(f"   Failed: {', '.join(failed_tickers)}")
        sys.exit(1)
    
    # Step 4: Generate comparison report
    print(f"\n{'='*60}")
    print(f"📊 Generating Peer Comparison Report")
    print(f"{'='*60}\n")
    
    # Separate target from peers
    target_result = next(r for r in results if r.ticker == args.ticker.upper())
    peer_results = [r for r in results if r.ticker != args.ticker.upper()]
    
    # Build comparison (using your existing peer_comparison module)
    from sigmak.peer_comparison import PeerComparisonPipeline
    comparison_pipeline = PeerComparisonPipeline(pipeline, None)
    
    peer_comparison = comparison_pipeline._calculate_comparison(
        target_result,
        peer_results
    )
    
    # Generate report
    output_file = f"output/{args.ticker}_Peer_Comparison_{args.year}.md"
    generate_markdown_report_with_peers(
        ticker=args.ticker,
        results=[target_result],
        peer_comparison=peer_comparison,
        output_file=output_file
    )
    
    print(f"\n✅ Report generated: {output_file}")
    print(f"\n{'='*60}")
    print(f"📈 Comparison Summary")
    print(f"{'='*60}")
    print(f"Target: {args.ticker}")
    print(f"Peers: {', '.join([r.ticker for r in peer_results])}")
    print(f"Severity Percentile: {peer_comparison.severity_percentile:.0f}th")
    print(f"Novelty Percentile: {peer_comparison.novelty_percentile:.0f}th")
    print(f"{'='*60}\n")


def find_filing_path(ticker: str, year: int, base_dir: str) -> Path | None:
    """
    Find filing file for a ticker/year.
    
    Searches in:
    - data/filings/{ticker}/{year}/*.htm
    - data/filings/{ticker}/*{year}*.htm
    - data/filings/*{ticker}*{year}*.htm
    """
    base = Path(base_dir)
    ticker_lower = ticker.lower()
    ticker_upper = ticker.upper()
    year_str = str(year)
    
    # Search patterns (in order of preference)
    search_paths = [
        base / ticker_upper / year_str,
        base / ticker_upper,
        base / ticker_lower / year_str,
        base / ticker_lower,
        base
    ]
    
    for search_path in search_paths:
        if not search_path.exists():
            continue
        
        # Look for .htm files with ticker and year
        matches = list(search_path.glob(f"*{ticker_lower}*{year_str}*.htm"))
        matches += list(search_path.glob(f"*{ticker_upper}*{year_str}*.htm"))
        
        if matches:
            # Prefer 10-K files
            for match in matches:
                if "10k" in match.name.lower():
                    return match
            # Otherwise return first match
            return matches[0]
    
    return None


def generate_markdown_report_with_peers(
    ticker: str,
    results: List,
    peer_comparison,
    output_file: str
) -> None:
    """Generate markdown report with peer comparison section."""
    # Import your existing report generation logic
    from scripts.generate_yoy_report import generate_markdown_report
    
    # TODO: Extend your existing report generator to include peer comparison
    # For now, call existing function and append peer section
    
    # ... (implementation depends on your current report structure)
    pass


if __name__ == "__main__":
    main()
📋 Implementation Steps
Phase 1: Core Infrastructure (Do This First)
Create src/sigmak/peer_discovery.py (as shown above)

Create tests:

Python
# tests/test_peer_discovery.py
def test_get_company_info():
    service = PeerDiscoveryService()
    info = service.get_company_info("NVDA")
    assert info.ticker == "NVDA"
    assert info.sic is not None

def test_find_peers():
    service = PeerDiscoveryService()
    peers = service.find_peers("NVDA", max_peers=3)
    assert len(peers) <= 3
    assert all(p.sic == peers[0].sic for p in peers)
Test peer discovery manually:

bash
python -c "
from sigmak.peer_discovery import PeerDiscoveryService
service = PeerDiscoveryService()
info = service.get_company_info('NVDA')
print(f'{info.name}: SIC {info.sic} - {info.sic_description}')
peers = service.find_peers('NVDA', max_peers=5)
for p in peers:
    print(f'  - {p.ticker}: {p.name}')
"
Phase 2: Orchestration Script
Create scripts/generate_peer_comparison_report.py (as shown above)
Add helper function find_filing_path() to handle flexible file discovery
Test end-to-end:
bash
# Run full workflow
python scripts/generate_peer_comparison_report.py NVDA 2024 --max-peers 3
Phase 3: Report Enhancement
Extend your existing generate_markdown_report() function to include peer comparison sections
Add peer benchmarking visualizations (severity percentiles, category divergence)
🚀 Usage Examples
bash
# Auto-discover peers and generate report
python scripts/generate_peer_comparison_report.py NVDA 2024

# Limit to 3 peers
python scripts/generate_peer_comparison_report.py NVDA 2024 --max-peers 3

# Manual peer selection
python scripts/generate_peer_comparison_report.py NVDA 2024 --peers AMD INTC

# Skip download if filings already exist
python scripts/generate_peer_comparison_report.py NVDA 2024 --skip-download
⚠️ Important Notes
SEC Rate Limiting: The peer enrichment process respects SEC's 10 requests/second limit
Caching: Company SIC data is cached in ./data/cache/ to avoid repeated API calls
Filing Availability: Not all peers may have filings for the target year (script handles gracefully)
SIC Code Limitations: SIC codes are broad (e.g., "3674 - Semiconductors"). You might want to add additional filtering by market cap or revenue tier.