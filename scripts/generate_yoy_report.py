#!/usr/bin/env python3
"""
Generate Investment-Grade Risk Analysis Report for SEC 10-K Filings.

This script analyzes SEC 10-K Risk Factor disclosures and produces an
investment-focused report for hedge fund analysts and portfolio managers:
- Material risk assessment for the latest filing
- Risk category concentration analysis
- Identification of emerging vs. resolved risks
- Business impact implications

Usage:
    python generate_yoy_report.py

For custom companies:
    python generate_yoy_report.py <ticker> <year1> <year2> <year3>

Example:
    python generate_yoy_report.py HURC 2023 2024 2025
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple, TYPE_CHECKING
import json
from collections import defaultdict
from dotenv import load_dotenv
from sigmak import filings_db

if TYPE_CHECKING:
    from sigmak.integration import RiskAnalysisResult

# Load environment variables from .env file
load_dotenv()


def load_or_analyze_filing(
    pipeline,
    html_path: str,
    ticker: str,
    year: int,
    retrieve_top_k: int = 10,
    use_llm: bool = False
) -> "RiskAnalysisResult":
    """
    Load cached results or analyze filing fresh.
    
    Args:
        pipeline: Integration pipeline instance
        html_path: Path to HTML filing
        ticker: Stock ticker symbol
        year: Filing year
        retrieve_top_k: Number of top risks to retrieve
        use_llm: Whether LLM classification is enabled (bypasses cache)
        
    Returns:
        RiskAnalysisResult with full risk analysis
    """
    # Import here to avoid heavy imports at module load-time during tests
    from sigmak.integration import IntegrationPipeline, RiskAnalysisResult

    cache_file = Path(f"output/results_{ticker}_{year}.json")
    
    # Skip cache when LLM is enabled to ensure fresh enrichment
    if not use_llm and cache_file.exists():
        print(f"📂 Loading cached results for {ticker} {year}...")
        with open(cache_file, 'r') as f:
            data = json.load(f)
            return RiskAnalysisResult(
                ticker=data['ticker'],
                filing_year=data['filing_year'],
                risks=data['risks'],
                metadata=data['metadata']
            )
    
    # Analyze fresh
    print(f"🔍 Analyzing {ticker} {year} from {html_path}...")
    result = pipeline.analyze_filing(
        html_path=html_path,
        ticker=ticker,
        filing_year=year,
        retrieve_top_k=retrieve_top_k
    )
    
    # Cache results
    with open(cache_file, 'w') as f:
        f.write(result.to_json())
    print(f"   ✅ Indexed {result.metadata['chunks_indexed']} chunks in {result.metadata['total_latency_ms']:.0f}ms")
    
    return result


def calculate_risk_similarity(risk1_text: str, risk2_text: str) -> float:
    """
    Calculate simple word overlap similarity between two risk texts.
    
    Args:
        risk1_text: First risk text
        risk2_text: Second risk text
        
    Returns:
        Similarity score (0.0 to 1.0)
    """
    # Simple word-based similarity (good enough for year-over-year comparison)
    words1 = set(risk1_text.lower().split())
    words2 = set(risk2_text.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    return intersection / union if union > 0 else 0.0


def identify_risk_changes(
    results: List["RiskAnalysisResult"],
    similarity_threshold: float = 0.4
) -> Dict[str, Any]:
    """
    Identify new, disappeared, and persistent risks across years.
    
    Args:
        results: List of RiskAnalysisResult ordered by year
        similarity_threshold: Minimum similarity to consider risks "the same"
        
    Returns:
        Dictionary with new_risks, disappeared_risks, and persistent_risks
    """
    changes = {
        'new_risks': [],
        'disappeared_risks': [],
        'persistent_risks': []
    }
    
    if len(results) < 2:
        return changes
    
    # Compare consecutive years
    for i in range(len(results) - 1):
        prev_result = results[i]
        curr_result = results[i + 1]
        
        prev_risks = prev_result.risks
        curr_risks = curr_result.risks
        
        # Find new risks (in current year but not in previous)
        for curr_risk in curr_risks:
            is_new = True
            for prev_risk in prev_risks:
                similarity = calculate_risk_similarity(
                    curr_risk['text'],
                    prev_risk['text']
                )
                if similarity >= similarity_threshold:
                    is_new = False
                    # Track as persistent if it's evolved
                    changes['persistent_risks'].append({
                        'year': curr_result.filing_year,
                        'prev_year': prev_result.filing_year,
                        'risk': curr_risk,
                        'similarity': similarity
                    })
                    break
            
            if is_new:
                changes['new_risks'].append({
                    'year': curr_result.filing_year,
                    'risk': curr_risk
                })
        
        # Find disappeared risks (in previous year but not in current)
        for prev_risk in prev_risks:
            is_disappeared = True
            for curr_risk in curr_risks:
                similarity = calculate_risk_similarity(
                    prev_risk['text'],
                    curr_risk['text']
                )
                if similarity >= similarity_threshold:
                    is_disappeared = False
                    break
            
            if is_disappeared:
                changes['disappeared_risks'].append({
                    'year': prev_result.filing_year,
                    'risk': prev_risk
                })
    
    return changes


def calculate_category_distribution(result: "RiskAnalysisResult") -> Dict[str, List[Dict[str, Any]]]:
    """
    Calculate distribution of risk categories with associated risks.
    
    Args:
        result: Risk analysis result
        
    Returns:
        Dictionary mapping category names to list of risks in that category
    """
    distribution = defaultdict(list)
    
    for risk in result.risks:
        # Extract category from risk metadata if available
        category = risk.get('category', 'UNCATEGORIZED')
        distribution[category].append(risk)
    
    return dict(distribution)


def extract_category_from_text(risk_text: str) -> str:
    """
    Extract category hint from risk text (fallback if category not in metadata).
    
    Args:
        risk_text: Risk disclosure text
        
    Returns:
        Best guess category name
    """
    text_lower = risk_text.lower()
    
    # Simple keyword matching for category inference
    category_keywords = {
        'OPERATIONAL': ['supply chain', 'manufacturing', 'operations', 'production', 'facilities'],
        'FINANCIAL': ['debt', 'liquidity', 'credit', 'capital', 'financial'],
        'REGULATORY': ['regulation', 'compliance', 'law', 'government', 'legal'],
        'COMPETITIVE': ['competition', 'competitor', 'market share', 'pricing pressure'],
        'TECHNOLOGICAL': ['technology', 'cybersecurity', 'innovation', 'digital', 'it systems'],
        'GEOPOLITICAL': ['international', 'trade', 'tariff', 'sanctions', 'war', 'political'],
        'HUMAN_CAPITAL': ['employee', 'talent', 'workforce', 'labor', 'personnel'],
        'REPUTATIONAL': ['reputation', 'brand', 'customer', 'trust', 'public perception'],
        'SYSTEMATIC': ['economic', 'recession', 'inflation', 'market volatility', 'macroeconomic']
    }
    
    for category, keywords in category_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            return category
    
    return 'OTHER'


def suggest_categories_from_keywords(risk_text: str) -> Tuple[List[str], List[str]]:
    """
    Suggest categories based on keyword matches in risk text.
    
    Args:
        risk_text: Risk disclosure text
        
    Returns:
        Tuple of (detected_keywords, suggested_categories)
    """
    text_lower = risk_text.lower()
    
    category_keywords = {
        'GEOPOLITICAL': ['tariff', 'trade', 'international', 'export', 'import', 'sanctions', 'war', 'political'],
        'OPERATIONAL': ['supply chain', 'supplier', 'manufacturing', 'production', 'facilities', 'operations'],
        'FINANCIAL': ['debt', 'liquidity', 'credit', 'capital', 'cash flow', 'interest'],
        'REGULATORY': ['regulation', 'compliance', 'law', 'government', 'sec', 'regulatory'],
        'COMPETITIVE': ['competition', 'competitor', 'market share', 'pricing', 'rival'],
        'TECHNOLOGICAL': ['technology', 'cyber', 'digital', 'it', 'systems', 'innovation'],
        'HUMAN_CAPITAL': ['employee', 'talent', 'workforce', 'labor', 'personnel', 'hiring'],
        'REPUTATIONAL': ['reputation', 'brand', 'customer', 'trust', 'image'],
        'SYSTEMATIC': ['economic', 'recession', 'inflation', 'market', 'macro']
    }
    
    detected_keywords = []
    category_scores = {}
    
    for category, keywords in category_keywords.items():
        matches = [kw for kw in keywords if kw in text_lower]
        if matches:
            category_scores[category] = len(matches)
            detected_keywords.extend(matches[:3])  # Limit to first 3 per category
    
    # Get top 2 suggested categories
    suggested = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)[:2]
    suggested_categories = [cat for cat, _ in suggested]
    
    return detected_keywords[:3], suggested_categories  # Return max 3 keywords


def load_filing_provenance(ticker: str, filing_year: int, filings_db_path: str | None = None) -> Dict[str, str]:
    """
    Load filing provenance metadata from JSON file if available.
    
    Args:
        ticker: Stock ticker symbol
        filing_year: Filing year
        
    Returns:
        Dictionary with accession, cik, and sec_url (or placeholders)
    """
    # First, attempt to load identifiers from the filings_index SQLite DB
    db_result = filings_db.get_identifiers(filings_db_path, ticker, filing_year)

    # If DB provided values (not missing token), return them
    if db_result and all(v != filings_db.MISSING_TOKEN for v in db_result.values()):
        return db_result

    # Otherwise, try the legacy JSON search as a fallback
    search_dirs = [Path("data/filings"), Path("data"), Path("data/samples")]

    # Common filename patterns
    patterns = [
        f"{ticker.lower()}-{filing_year}*.json",
        f"{ticker.upper()}-{filing_year}*.json",
    ]

    metadata = {
        'accession': db_result.get('accession', filings_db.MISSING_TOKEN),
        'cik': db_result.get('cik', filings_db.MISSING_TOKEN),
        'sec_url': db_result.get('sec_url', filings_db.MISSING_TOKEN)
    }

    # Look for JSON files matching the patterns in all search directories
    for data_dir in search_dirs:
        if not data_dir.exists():
            continue
        for pattern in patterns:
            matches = list(data_dir.glob(pattern))
            if matches:
                try:
                    with open(matches[0], 'r') as f:
                        data = json.load(f)
                        metadata['accession'] = data.get('accession', metadata['accession'])
                        metadata['cik'] = data.get('cik', metadata['cik'])
                        metadata['sec_url'] = data.get('sec_url', metadata['sec_url'])
                        return metadata  # Return immediately on first match
                except (json.JSONDecodeError, IOError):
                    pass

    return metadata


def generate_markdown_report(
    ticker: str,
    results: List["RiskAnalysisResult"],
    output_file: str = "risk_analysis_report.md",
    filings_db_path: str | None = None,
) -> None:
    """
    Generate investment-grade markdown report focusing on latest year's risks.
    
    Args:
        ticker: Stock ticker symbol
        results: List of RiskAnalysisResult ordered by year (oldest to newest)
        output_file: Output markdown file path
    """
    results_sorted = sorted(results, key=lambda r: r.filing_year)
    years = [r.filing_year for r in results_sorted]
    latest_result = results_sorted[-1]
    latest_year = latest_result.filing_year
    changes = identify_risk_changes(results_sorted)
    
    # Build the report
    report_lines = []
    
    # Header
    report_lines.append(f"# {ticker} — Risk Factor Analysis")
    report_lines.append(f"## Item 1A Deep Dive: {latest_year} 10-K")
    report_lines.append("")
    report_lines.append(f"**Report Date:** {datetime.now().strftime('%B %d, %Y')}")
    
    # Add provenance metadata
    provenance = load_filing_provenance(ticker, latest_year, filings_db_path)
    sec_url = provenance['sec_url']
    if not sec_url.startswith('http') and sec_url != '<SEC_URL>':
        sec_url = f"https://www.sec.gov{sec_url if sec_url.startswith('/') else '/' + sec_url}"
    
    report_lines.append(f"**Filing:** 10-K (Accession: {provenance['accession']}) • CIK: {provenance['cik']} • SEC URL: {sec_url}")
    report_lines.append(f"**Historical Comparison:** {years[0]}–{years[-1]}")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    
    # Executive Summary
    report_lines.append("## Executive Summary")
    report_lines.append("")
    
    # Calculate stats for latest year
    latest_severities = [r['severity']['value'] for r in latest_result.risks if 'severity' in r]
    latest_novelties = [r['novelty']['value'] for r in latest_result.risks if 'novelty' in r]
    avg_severity = sum(latest_severities) / len(latest_severities) if latest_severities else 0
    avg_novelty = sum(latest_novelties) / len(latest_novelties) if latest_novelties else 0
    
    total_new = len([r for r in changes['new_risks'] if r['year'] == latest_year])
    total_disappeared = len(changes['disappeared_risks'])
    
    # High/medium/low severity buckets
    high_severity = len([s for s in latest_severities if s >= 0.7])
    medium_severity = len([s for s in latest_severities if 0.4 <= s < 0.7])
    low_severity = len([s for s in latest_severities if s < 0.4])
    
    report_lines.append("### Key Takeaways")
    report_lines.append("")
    
    # Generate PM takeaway based on top risks and changes
    category_dist = calculate_category_distribution(latest_result)
    sorted_cats = sorted(
        category_dist.items(),
        key=lambda x: len(x[1]) * (sum(r.get('severity', {}).get('value', 0) for r in x[1]) / len(x[1]) if x[1] else 0),
        reverse=True
    )
    
    top_category = sorted_cats[0][0] if sorted_cats else "OPERATIONAL"
    top_cat_count = len(sorted_cats[0][1]) if sorted_cats else 0
    
    # Build concise takeaway
    if total_new > 0 and high_severity >= 3:
        what = f"{total_new} new {top_category.lower()} risk(s)"
        why = f"{high_severity} material disclosures"
        action = f"track {top_category.lower()} metrics"
    elif high_severity >= len(latest_result.risks) * 0.3:
        what = f"{high_severity} material {top_category.lower()} risks"
        why = f"{top_cat_count} disclosures in category"
        action = f"monitor {top_category.lower()} exposure"
    else:
        what = f"{top_cat_count} {top_category.lower()} disclosures"
        why = f"primary risk concentration"
        action = f"watch {top_category.lower()} trends"
    
    takeaway = f"**TAKEAWAY:** {what}; {why}; {action}."
    if len(takeaway) > 150:  # Fallback if too long
        takeaway = f"**TAKEAWAY:** {high_severity} material risks in {top_category}; monitor closely."
    
    report_lines.append(takeaway)
    report_lines.append("")
    
    # Alert-style summary bullets
    report_lines.append(f"**FILING SNAPSHOT:** {len(latest_result.risks)} total risk factors | {high_severity} material (≥0.70 severity) | Avg severity: {avg_severity:.2f}")
    report_lines.append("")
    
    if total_new > 0:
        report_lines.append(f"🚨 **NEW:** {total_new} risk factor(s) added — signals shifting exposure; review disclosure language for operational changes.")
    
    if total_disappeared > 0:
        report_lines.append(f"✅ **RESOLVED:** {total_disappeared} risk factor(s) dropped from disclosure — potential business improvement or strategic pivot.")
    
    if high_severity >= len(latest_result.risks) * 0.4:
        report_lines.append(f"⚠️ **ALERT:** {100*high_severity/len(latest_result.risks):.0f}% of disclosures are high severity — elevated risk profile requires close portfolio monitoring.")
    
    report_lines.append("")
    report_lines.append("### Risk Severity Distribution")
    report_lines.append("")
    report_lines.append("| Severity Level | Count | % of Total |")
    report_lines.append("|----------------|-------|------------|")
    total_risks = len(latest_severities)
    report_lines.append(f"| High (≥0.70) | {high_severity} | {100*high_severity/total_risks:.0f}% |")
    report_lines.append(f"| Medium (0.40-0.69) | {medium_severity} | {100*medium_severity/total_risks:.0f}% |")
    report_lines.append(f"| Low (<0.40) | {low_severity} | {100*low_severity/total_risks:.0f}% |")
    report_lines.append("")
    
    # Add severity legend and percentile
    report_lines.append("**Severity Legend:** 🔴 Critical ≥0.75 | 🟠 High 0.50–0.74 | 🟡 Medium 0.30–0.49 | 🟢 Low <0.30")
    report_lines.append("")
    report_lines.append(f"**{ticker} avg severity percentile vs. universe:** <placeholder percentile>")
    report_lines.append("")
    
    # Category distribution
    category_dist = calculate_category_distribution(latest_result)
    
    report_lines.append("### Risk Concentration by Category")
    report_lines.append("")
    
    # Sort categories by weighted severity (count × avg severity)
    sorted_categories = sorted(
        category_dist.items(),
        key=lambda x: len(x[1]) * (sum(r.get('severity', {}).get('value', 0) for r in x[1]) / len(x[1]) if x[1] else 0),
        reverse=True
    )
    
    # Find most concentrated category
    if sorted_categories:
        top_category = sorted_categories[0][0]
        top_count = len(sorted_categories[0][1])
        report_lines.append(f"**Primary Risk Exposure**: {top_category} ({top_count} disclosures)")
        report_lines.append("")
    
    report_lines.append("| Category | Risk Count | Avg Severity | Material Risks (≥0.70) |")
    report_lines.append("|----------|------------|--------------|------------------------|")
    
    for category, risks in sorted_categories:
        count = len(risks)
        severities = [r.get('severity', {}).get('value', 0) for r in risks]
        avg_cat_severity = sum(severities) / len(severities) if severities else 0
        material_count = len([s for s in severities if s >= 0.7])
        
        # Handle uncategorized with suggestions
        display_category = category
        if category in ['UNCATEGORIZED', 'OTHER']:
            # Get keyword suggestions from first risk in category
            first_risk = risks[0] if risks else None
            if first_risk:
                keywords, suggestions = suggest_categories_from_keywords(first_risk['text'])
                if suggestions:
                    display_category = f"Uncategorized (low classifier confidence) — suggested: {', '.join(suggestions)}"
                else:
                    display_category = "Uncategorized (low classifier confidence)"
        
        report_lines.append(f"| {display_category} | {count} | {avg_cat_severity:.2f} | {material_count} |")
    
    # Add keyword mapping note for uncategorized if present
    has_uncategorized = any(cat in ['UNCATEGORIZED', 'OTHER'] for cat, _ in sorted_categories)
    if has_uncategorized:
        for category, risks in sorted_categories:
            if category in ['UNCATEGORIZED', 'OTHER'] and risks:
                keywords, suggestions = suggest_categories_from_keywords(risks[0]['text'])
                if keywords:
                    kw_str = ', '.join(keywords)
                    sugg_str = '/'.join(suggestions) if suggestions else 'N/A'
                    report_lines.append("")
                    report_lines.append(f"_Keywords detected: {kw_str} → suggest {sugg_str}_")
                break
    
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    
    # Top 10 Risks by Severity (Latest Year)
    report_lines.append(f"## Material Risk Factors — {latest_year}")
    report_lines.append("")
    report_lines.append("The following represents the most severe risk disclosures requiring investor attention:")
    report_lines.append("")
    
    # Load provenance for evidence links
    provenance = load_filing_provenance(ticker, latest_year, filings_db_path)
    base_sec_url = provenance['sec_url']
    if base_sec_url == '<SEC_URL>':
        base_sec_url = f"https://www.sec.gov/Archives/edgar/data/{provenance['cik']}/{provenance['accession'].replace('-', '')}/{provenance['accession']}.htm"
    
    sorted_risks = sorted(
        latest_result.risks,
        key=lambda r: r.get('severity', {}).get('value', 0),
        reverse=True
    )[:10]
    
    for i, risk in enumerate(sorted_risks, 1):
        severity = risk.get('severity', {})
        novelty = risk.get('novelty', {})
        category = risk.get('category', extract_category_from_text(risk['text']))
        text_preview = risk['text'][:250].replace('\n', ' ')
        
        severity_val = severity.get('value', 0)
        novelty_val = novelty.get('value', 0)
        
        # Determine risk label
        if severity_val >= 0.8:
            risk_label = "🔴 CRITICAL"
        elif severity_val >= 0.7:
            risk_label = "🟠 HIGH"
        elif severity_val >= 0.5:
            risk_label = "🟡 MODERATE"
        else:
            risk_label = "🟢 LOW"
        
        # Determine status badge
        status_badge = ""
        if novelty_val >= 0.50:
            status_badge = " ★ NEW"
        
        # Check if this risk was in previous years (dropped status)
        for dropped in changes['disappeared_risks']:
            dropped_text = dropped['risk']['text']
            similarity = calculate_risk_similarity(risk['text'], dropped_text)
            if similarity >= 0.4:
                dropped_year = dropped['year']
                status_badge = f" — DROPPED (previously disclosed in {dropped_year})"
                break
        
        # Determine confidence level
        if severity_val >= 0.80 and novelty_val >= 0.20:
            confidence = "High"
        elif (0.50 <= severity_val < 0.80) or (0.10 <= novelty_val <= 0.49):
            confidence = "Medium"
        else:
            confidence = "Low"
        
        # Count supporting quotes (default to 1 for the current quote)
        supporting_quotes = risk.get('metadata', {}).get('chunk_count', 1)
        
        # Build evidence URL with anchor
        evidence_url = f"{base_sec_url}#para{i}"
        
        # Generate impact label from severity explanation
        severity_explanation = severity.get('explanation', 'material business impact')
        impact_keywords = {
            'revenue': 'revenue impact',
            'earnings': 'earnings impact',
            'profit': 'margin compression',
            'cash': 'liquidity risk',
            'operational': 'operational disruption',
            'supply chain': 'supply disruption',
            'competition': 'competitive pressure',
            'regulatory': 'compliance risk',
            'market': 'market volatility',
            'customer': 'customer attrition',
            'financial': 'financial stress'
        }
        
        impact_label = "material business impact"
        for keyword, label in impact_keywords.items():
            if keyword in severity_explanation.lower():
                impact_label = label
                break
        
        # Severity-based qualifier
        if severity_val >= 0.8:
            impact_label = f"{impact_label} (High)"
        elif severity_val >= 0.7:
            impact_label = f"{impact_label} (Med-High)"
        else:
            impact_label = f"{impact_label} (Moderate)"
        
        # Category-based monitoring signals
        monitoring_signals = {
            'OPERATIONAL': 'production metrics, supplier performance',
            'FINANCIAL': 'liquidity ratios, credit ratings',
            'REGULATORY': 'regulatory filings, policy announcements',
            'COMPETITIVE': 'market share data, pricing trends',
            'TECHNOLOGICAL': 'cybersecurity incidents, tech investments',
            'GEOPOLITICAL': 'trade policy, tariff announcements',
            'HUMAN_CAPITAL': 'turnover rates, hiring metrics',
            'REPUTATIONAL': 'brand sentiment, customer reviews',
            'SYSTEMATIC': 'macro indicators, interest rates',
            'OTHER': 'company disclosures, industry trends'
        }
        
        monitoring = monitoring_signals.get(category, 'quarterly filings, management guidance')
        
        # Build short evidence link
        evidence_short = f"para {i}"
        
        report_lines.append(f"### {i}. {risk_label} — {category}{status_badge}")
        report_lines.append("")
        report_lines.append(f"_Severity: {severity_val:.2f} | Novelty: {novelty_val:.2f}_")
        report_lines.append("")
        report_lines.append(f"> {text_preview}...")
        report_lines.append("")
        report_lines.append(f"**Impact:** {impact_label} | **Confidence:** {confidence} | **Evidence:** [{evidence_short}]({evidence_url}) | **Monitoring:** {monitoring}")
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")
    
    # Category Deep Dives
    report_lines.append("## Risk Category Breakdown")
    report_lines.append("")
    
    for category, risks in sorted_categories[:5]:  # Top 5 categories only
        if not risks:
            continue
            
        severities = [r.get('severity', {}).get('value', 0) for r in risks]
        avg_severity = sum(severities) / len(severities) if severities else 0
        
        # Handle uncategorized display
        display_category = category
        if category in ['UNCATEGORIZED', 'OTHER']:
            keywords, suggestions = suggest_categories_from_keywords(risks[0]['text'])
            if suggestions:
                display_category = f"Uncategorized (low classifier confidence) — suggested mapping: {', '.join(suggestions)}"
                report_lines.append(f"### {display_category}")
                report_lines.append("")
                report_lines.append(f"**{len(risks)} risks, avg severity: {avg_severity:.2f}**")
                if keywords:
                    kw_str = ', '.join(keywords)
                    sugg_str = '/'.join(suggestions)
                    report_lines.append("")
                    report_lines.append(f"_Keywords detected: {kw_str} → suggest {sugg_str}_")
            else:
                report_lines.append(f"### Uncategorized (low classifier confidence)")
                report_lines.append("")
                report_lines.append(f"**{len(risks)} risks, avg severity: {avg_severity:.2f}**")
        else:
            report_lines.append(f"### {category} ({len(risks)} risks, avg severity: {avg_severity:.2f})")
        
        report_lines.append("")
        
        # Get top 2 risks in this category
        top_risks = sorted(
            risks,
            key=lambda r: r.get('severity', {}).get('value', 0),
            reverse=True
        )[:2]
        
        for risk in top_risks:
            severity = risk.get('severity', {}).get('value', 0)
            text_preview = risk['text'][:180].replace('\n', ' ')
            
            report_lines.append(f"- **[{severity:.2f}]** {text_preview}...")
        
        report_lines.append("")
    
    # Disappeared Risks Section (from historical context)
    if changes['disappeared_risks']:
        report_lines.append("## Resolved or De-emphasized Risks")
        report_lines.append("")
        report_lines.append(f"The following risks were disclosed in prior filings but removed in {latest_year}, potentially signaling improved business conditions or strategic pivots:")
        report_lines.append("")
        
        for disappeared_risk in changes['disappeared_risks'][:8]:  # Limit to top 8
            year = disappeared_risk['year']
            risk = disappeared_risk['risk']
            category = risk.get('category', extract_category_from_text(risk['text']))
            text_preview = risk['text'][:180].replace('\n', ' ')
            
            report_lines.append(f"**[{category}] Last seen: {year}**")
            report_lines.append(f"> {text_preview}...")
            report_lines.append("")
    
    # Comparison Table
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## Multi-Year Disclosure Trends")
    report_lines.append("")
    report_lines.append("| Year | Risk Factors | High Severity (≥0.70) | Avg Severity |")
    report_lines.append("|------|--------------|----------------------|--------------|")
    
    for result in results_sorted:
        year = result.filing_year
        total = len(result.risks)
        sev_values = [r.get('severity', {}).get('value', 0) for r in result.risks]
        high_sev = len([s for s in sev_values if s >= 0.7])
        avg_sev = sum(sev_values) / len(sev_values) if sev_values else 0
        
        report_lines.append(f"| {year} | {total} | {high_sev} | {avg_sev:.2f} |")
    
    report_lines.append("")
    report_lines.append("")
    
    # Methodology
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## About This Analysis")
    report_lines.append("")
    report_lines.append("This report analyzes Item 1A (Risk Factors) disclosures from SEC 10-K filings using proprietary natural language processing and semantic analysis. Each risk factor is:")
    report_lines.append("")
    report_lines.append("1. **Extracted** from the company's official SEC filing")
    report_lines.append("2. **Scored** for severity (potential business impact) and novelty (uniqueness vs. historical patterns)")
    report_lines.append("3. **Classified** into standard risk categories for cross-company comparison")
    report_lines.append("4. **Compared** against prior year disclosures to identify emerging or resolved risks")
    report_lines.append("")
    report_lines.append("**Severity scores** (0.0–1.0) reflect the potential magnitude of business impact, with ≥0.70 considered material.")
    report_lines.append("")
    report_lines.append("**Novelty scores** (0.0–1.0) indicate how unprecedented or unique a disclosure is relative to the company's historical risk profile and industry norms.")
    report_lines.append("")
    report_lines.append("### Risk Category Framework")
    report_lines.append("")
    report_lines.append("- **OPERATIONAL**: Supply chain, manufacturing, operations")
    report_lines.append("- **FINANCIAL**: Liquidity, debt, capital structure")
    report_lines.append("- **REGULATORY**: Compliance, legal, policy")
    report_lines.append("- **COMPETITIVE**: Market position, pricing, rivalry")
    report_lines.append("- **TECHNOLOGICAL**: Cybersecurity, innovation, disruption")
    report_lines.append("- **GEOPOLITICAL**: Trade, sanctions, international exposure")
    report_lines.append("- **HUMAN_CAPITAL**: Workforce, talent, labor relations")
    report_lines.append("- **REPUTATIONAL**: Brand, trust, public perception")
    report_lines.append("- **SYSTEMATIC**: Macro conditions, market volatility")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append(f"*Analysis powered by SigmaK Risk Intelligence Platform | {datetime.now().strftime('%B %Y')}*")
    
    # Write to file
    output_path = Path(output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\n✨ Report generated: {output_path.absolute()}")


def main():
    """Main entry point for report generation."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Generate YoY Risk Analysis Report for SEC 10-K filings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_yoy_report.py                          # Default: HURC 2023-2025
  python generate_yoy_report.py TSLA 2022 2023 2024     # Custom ticker and years
  python generate_yoy_report.py --use-llm               # Enable LLM classification
  python generate_yoy_report.py HURC 2023 2024 2025 --use-llm
        """
    )
    parser.add_argument(
        'ticker',
        nargs='?',
        default='HURC',
        help='Stock ticker symbol (default: HURC)'
    )
    parser.add_argument(
        'years',
        nargs='*',
        type=int,
        help='Filing years (default: 2023 2024 2025)'
    )
    parser.add_argument(
        '--use-llm',
        action='store_true',
        help='Enable LLM-based risk classification (increases latency and cost)'
    )
    
    args = parser.parse_args()
    
    # Set defaults
    ticker = args.ticker.upper()
    years = args.years if args.years else [2023, 2024, 2025]
    use_llm = args.use_llm
    
    # Validate years
    if len(years) < 2:
        parser.error("At least 2 years required for YoY comparison")
    
    # Build file paths
    filings = []
    for year in years:
        # Try common naming patterns (search filings/ first, then data/, then samples/)
        candidates = [
            f"data/filings/{ticker.lower()}-{year}1031x10k.htm",
            f"data/filings/{ticker.lower()}-{year}1231x10k.htm",
            f"data/filings/{ticker.lower()}-{year}0930x10k.htm",
            f"data/filings/{ticker.lower()}-{year}0630x10k.htm",
            f"data/{ticker.lower()}-{year}1031x10k.htm",
            f"data/{ticker.lower()}-{year}1231x10k.htm",
            f"data/{ticker.lower()}-{year}0930x10k.htm",
            f"data/{ticker.lower()}-{year}0630x10k.htm",
            f"data/samples/{ticker.lower()}-{year}1031x10k.htm",
            f"data/samples/{ticker.lower()}-{year}1231x10k.htm",
        ]
        
        found = False
        for candidate in candidates:
            if Path(candidate).exists():
                filings.append((candidate, ticker, year))
                found = True
                break
        
        if not found:
            print(f"⚠️  Could not find filing for {ticker} {year}")
            print(f"    Tried: {', '.join(candidates)}")
            sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"📊 SigmaK Year-over-Year Risk Analysis")
    print(f"{'='*60}")
    print(f"Company: {ticker}")
    print(f"Years: {', '.join(map(str, years))}")
    print(f"LLM Classification: {'Enabled' if use_llm else 'Disabled'}")
    print(f"{'='*60}\n")
    
    # Initialize pipeline
    pipeline = IntegrationPipeline(persist_path="./chroma_db", use_llm=use_llm)
    
    # Analyze each filing
    results = []
    for html_path, ticker_sym, year in filings:
        result = load_or_analyze_filing(
            pipeline=pipeline,
            html_path=html_path,
            ticker=ticker_sym,
            year=year,
            retrieve_top_k=10,
            use_llm=use_llm
        )
        results.append(result)
    
    # Generate report
    print(f"\n{'='*60}")
    print("📝 Generating Markdown Report...")
    print(f"{'='*60}\n")
    
    output_file = f"output/{ticker}_YoY_Risk_Analysis_{min(years)}_{max(years)}.md"
    generate_markdown_report(ticker, results, output_file)
    
    print(f"\n{'='*60}")
    print("✅ Analysis Complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
