#!/usr/bin/env python3
"""Generate a Markdown peer comparison report for a target ticker and year.

Behavior:
- Loads downloaded 10-K HTML files from `data/filings/{TICKER}/{YEAR}`.
- If a peer's filing for the requested year is missing, attempts to download it.
- Runs the `IntegrationPipeline` analysis for each filing and computes simple
  comparison metrics (average severity percentile, category divergence,
  unique/shared risks).
- Writes a Markdown report to `output/{TICKER}_Peer_Comparison_{YEAR}.md`.

The script exits with an informative message if a required download fails.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from statistics import mean, median
import re
from collections import Counter
from typing import Dict, List, Tuple
from dotenv import load_dotenv

from sigmak.integration import IntegrationPipeline, IntegrationError, RiskAnalysisResult
from sigmak.text_utils import clean_text
from sigmak.downloads.tenk_downloader import TenKDownloader, resolve_ticker_to_cik, fetch_company_submissions
from sigmak.peer_discovery import PeerDiscoveryService

logger = logging.getLogger(__name__)

# Load environment variables from .env so GOOGLE_API_KEY is available
load_dotenv()


def find_html_in_dir(dirpath: Path) -> Path | None:
    if not dirpath.exists():
        return None
    for ext in ("*.htm", "*.html"):
        matches = list(dirpath.glob(ext))
        if matches:
            return matches[0]
    return None


def locate_filing_html(ticker_dir: Path, year: int) -> Path | None:
    """Locate an HTML filing for a ticker directory, tolerant of mis-filed years.

    Search strategy (in order):
    1. Exact year dir (e.g., ticker_dir / "2024/")
    2. Any file under ticker_dir whose filename or parent dir name contains the year
    3. Any HTML under ticker_dir whose content contains a filing date with the year
    4. Most recently modified HTML under ticker_dir
    """
    # 1) exact year folder
    exact = find_html_in_dir(ticker_dir / str(year))
    if exact:
        return exact

    # 2) filename or parent folder hint
    candidates: List[Path] = list(ticker_dir.rglob("*.htm")) + list(ticker_dir.rglob("*.html"))
    if not candidates:
        return None

    for c in candidates:
        if str(year) in c.name or str(year) in str(c.parent.name):
            return c

    # 3) inspect file contents for a date containing the year
    import re

    date_re = re.compile(r"(\d{4})-\d{2}-\d{2}")
    for c in candidates:
        try:
            txt = c.read_text(encoding="utf-8", errors="ignore")[:4096]
        except OSError:
            continue
        m = date_re.search(txt)
        if m and int(m.group(1)) == year:
            return c

    # 4) fallback to most recently modified
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def ensure_filing(downloader: TenKDownloader, ticker: str, year: int) -> Path:
    """Ensure a 10-K HTML exists for ticker/year; download if missing.

    Returns path to the HTML file or raises RuntimeError on failure.
    """
    ticker_dir = Path(downloader.download_dir) / ticker
    html = locate_filing_html(ticker_dir, year)
    if html:
        return html

    # Attempt to locate filing via SEC submissions and download the requested year
    try:
        cik = resolve_ticker_to_cik(ticker)
    except ValueError as e:
        raise RuntimeError(f"Ticker resolution failed for {ticker}: {e}")

    try:
        filings = fetch_company_submissions(cik, form_type="10-K")
    except Exception as e:
        raise RuntimeError(f"Failed to fetch submissions for {ticker}: {e}")

    chosen = None
    for f in filings:
        if str(year) == f.filing_date.split("-")[0]:
            chosen = f
            break
    if not chosen:
        if filings:
            chosen = sorted(filings, key=lambda x: x.filing_date, reverse=True)[0]
            logger.info("No exact-year 10-K for %s; falling back to most recent %s", ticker, chosen.filing_date)
        else:
            raise RuntimeError(f"No 10-K filings found for {ticker}")

    # Record filing and download via downloader
    filing_id = downloader.db.insert_filing(chosen)
    try:
        downloader.download_filing(chosen, filing_id)
    except Exception as e:
        raise RuntimeError(f"Download failed for {ticker}: {e}")

    html = find_html_in_dir(ticker_dir)
    if not html:
        raise RuntimeError(f"Downloaded filing for {ticker} but no HTML found at {ticker_dir}")
    return html


def compute_severity_avg(result: RiskAnalysisResult) -> float:
    vals = [r.get("severity", {}).get("value", 0.0) for r in result.risks]
    return float(mean(vals)) if vals else 0.0


def compute_category_distribution(result: RiskAnalysisResult) -> Dict[str, float]:
    counts: Dict[str, int] = {}
    for r in result.risks:
        cat = r.get("category", "UNCATEGORIZED")
        counts[cat] = counts.get(cat, 0) + 1
    total = sum(counts.values()) or 1
    return {k: v / total for k, v in counts.items()}


def generate_markdown_report(target: str, year: int, results: List[RiskAnalysisResult], outpath: Path) -> None:
    target_result = next(r for r in results if r.ticker.upper() == target.upper())
    peer_results = [r for r in results if r.ticker.upper() != target.upper()]

    lines: List[str] = []
    lines.append(f"# Peer Comparison Report — {target} {year}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")

    # Severity percentiles
    target_sev = compute_severity_avg(target_result)
    peer_sevs = [compute_severity_avg(r) for r in peer_results]
    all_sevs = peer_sevs + [target_sev]
    # percentile
    higher = sum(1 for v in peer_sevs if v < target_sev)
    pct = (higher / max(1, len(peer_sevs))) * 100 if peer_sevs else 100.0

    lines.append(f"**Severity average (target):** {target_sev:.3f}")
    lines.append(f"**Severity percentile vs peers:** {pct:.0f}th")
    lines.append("")

    # Category divergence (simple top categories)
    target_dist = compute_category_distribution(target_result)
    lines.append("## Category Distribution (Target)")
    lines.append("")
    for cat, pctv in sorted(target_dist.items(), key=lambda x: -x[1])[:10]:
        lines.append(f"- {cat}: {pctv*100:.1f}%")
    lines.append("")

    # List companies compared with (peers)
    lines.append("## Compared Companies")
    lines.append("")
    if peer_results:
        for pr in peer_results:
            lines.append(f"- {pr.ticker.upper()}")
    else:
        lines.append("- (no peers analyzed)")
    lines.append("")

    # Unique / Shared risks (token-overlap Jaccard matching)
    SHARED_THRESHOLD = 0.35

    def tokenize(s: str):
        return set(re.findall(r"\w+", (s or "").lower()))

    target_risks = [clean_text(r.get("text", "")) for r in target_result.risks]
    peer_risks = []
    for pr in peer_results:
        for r in pr.risks:
            peer_risks.append((pr.ticker.upper(), clean_text(r.get("text", ""))))

    unique_previews = []
    shared_entries = []

    for t in target_risks:
        t_tokens = tokenize(t)
        best_score = 0.0
        best_peer = None
        best_peer_text = ""
        for peer_ticker, ptext in peer_risks:
            p_tokens = tokenize(ptext)
            if not t_tokens and not p_tokens:
                score = 1.0
            else:
                inter = len(t_tokens & p_tokens)
                uni = len(t_tokens | p_tokens)
                score = inter / uni if uni else 0.0
            if score > best_score:
                best_score = score
                best_peer = peer_ticker
                best_peer_text = ptext

        if best_score >= SHARED_THRESHOLD:
            shared_entries.append({
                "target_preview": (t or "").strip()[:200],
                "peer": best_peer,
                "peer_preview": (best_peer_text or "").strip()[:200],
                "score": best_score,
            })
        else:
            unique_previews.append((t or "").strip()[:200])

    lines.append("## Unique Risks (target, preview)")
    lines.append("")
    for u in unique_previews[:10]:
        lines.append(f"- {u}")
    lines.append("")

    lines.append("## Shared Risks (target vs peers)")
    lines.append("")
    if shared_entries:
        for s in sorted(shared_entries, key=lambda x: -x["score"])[:20]:
            lines.append(f"- {s['target_preview']}  — shared with **{s['peer']}** (score {s['score']:.3f})")
            if s['peer_preview']:
                lines.append(f"  - peer snippet: {s['peer_preview']}")
    else:
        lines.append("- (no shared risks detected at current threshold)")
    lines.append("")

    # defer writing file until all sections are appended (specific comparisons + histogram)

    # --- Additional comparisons requested in ISSUE #82
    # Aggregate texts
    def normalize_text(s: str) -> str:
        # Use the central cleaner then collapse whitespace
        return clean_text(s)

    def sentences(text: str):
        # naive sentence split
        s = re.split(r"(?<=[.!?])\s+", text)
        return [re.sub(r"[^\w\s]", "", x).strip().lower() for x in s if x.strip()]

    def words(text: str):
        return re.findall(r"\w+", text.lower())

    company_texts = {r.ticker.upper(): "\n".join([clean_text(k.get("text", "")) for k in r.risks]) for r in results}

    target_text = company_texts.get(target.upper(), "")

    # 1) Textual Novelty (Year-over-Year)
    prior_path = Path(f"output/results_{target}_{year-1}.json")
    novelty_line = "Textual novelty: N/A (prior year not available)"
    if prior_path.exists():
        try:
            import json
            prev = json.loads(prior_path.read_text())
            prev_text = "\n".join([r.get("text", "") for r in prev.get("risks", [])])
            cur_sents = set(sentences(target_text))
            prev_sents = set(sentences(prev_text))
            if cur_sents:
                new = cur_sents - prev_sents
                pct_new = (len(new) / len(cur_sents)) * 100.0
                novelty_line = f"Textual novelty (YoY): {pct_new:.1f}% new sentences ({len(new)} of {len(cur_sents)})"
            else:
                novelty_line = "Textual novelty: no sentences detected in current filing"
        except Exception:
            novelty_line = "Textual novelty: error computing novelty"

    lines.append("## Specific Comparisons")
    lines.append("")
    lines.append(f"- **{novelty_line}**")

    # 2) Peer Similarity Score (Jaccard + Cosine)
    # Build simple token sets and TF vectors
    all_companies = list(company_texts.keys())
    token_sets = {c: set(words(company_texts[c])) for c in all_companies}
    def jaccard(a: set, b: set) -> float:
        if not a and not b:
            return 1.0
        inter = len(a & b)
        uni = len(a | b)
        return inter / uni if uni else 0.0

    peer_scores = []
    peer_texts = [company_texts[c] for c in all_companies if c != target.upper()]
    peer_tokens = [token_sets[c] for c in all_companies if c != target.upper()]

    # target vs peer average (Jaccard)
    for pt in peer_tokens:
        peer_scores.append(jaccard(token_sets.get(target.upper(), set()), pt))

    if peer_scores:
        avg_jaccard = sum(peer_scores) / len(peer_scores)
        lines.append(f"- **Peer similarity (avg Jaccard):** {avg_jaccard:.3f} — lower means more company-specific")
    else:
        lines.append(f"- **Peer similarity (avg Jaccard):** N/A (no peers analyzed)")

    # 3) Risk Density & Volume (word count, paragraph count)
    def paragraphs(text: str):
        paras = [p.strip() for p in text.split('\n') if p.strip()]
        return paras

    stats = {}
    for c, txt in company_texts.items():
        w = words(txt)
        p = paragraphs(txt)
        stats[c] = {"words": len(w), "paras": len(p)}

    peer_words = [v["words"] for k, v in stats.items() if k != target.upper()]
    peer_paras = [v["paras"] for k, v in stats.items() if k != target.upper()]
    tgt_words = stats.get(target.upper(), {}).get("words", 0)
    tgt_paras = stats.get(target.upper(), {}).get("paras", 0)

    if peer_words:
        med_words = median(peer_words)
        med_paras = median(peer_paras)
        lines.append(f"- **Risk density & volume:** target words={tgt_words}, peers median words={med_words}; target paras={tgt_paras}, peers median paras={med_paras}")
    else:
        lines.append(f"- **Risk density & volume:** target words={tgt_words}, paras={tgt_paras} (no peers)")

    # 4) Linguistic Tone (keyword density per 1k words)
    risk_keywords = ["litigation", "volatility", "uncertainty", "adverse", "risk", "loss", "decline", "regulatory", "compliance"]
    tone = {}
    for c, txt in company_texts.items():
        w = words(txt)
        cnt = sum(1 for t in w if t in risk_keywords)
        per_k = (cnt / max(1, len(w))) * 1000
        tone[c] = {"count": cnt, "per_k": per_k}

    tgt_tone = tone.get(target.upper(), {"count": 0, "per_k": 0})
    if peer_words:
        peer_perks = [tone[c]["per_k"] for c in tone if c != target.upper()]
        med_perk = median(peer_perks) if peer_perks else 0.0
        lines.append(f"- **Linguistic tone (risk keywords/1k words):** target={tgt_tone['per_k']:.2f}, peers median={med_perk:.2f}")
    else:
        lines.append(f"- **Linguistic tone (risk keywords/1k words):** target={tgt_tone['per_k']:.2f} (no peers)")

    # append final note
    lines.append("")
    lines.append("*End of specific comparisons — computed with simple token-based metrics (see ISSUE #82 for desired refinements).*")

    # --- Histogram of pairwise Jaccard scores (target risk snippets vs peer snippets)
    # Build score list using the same tokenize() helper above
    try:
        pair_scores: List[float] = []
        for t in target_risks:
            t_tokens = tokenize(t)
            for peer_ticker, ptext in peer_risks:
                p_tokens = tokenize(ptext)
                if not t_tokens and not p_tokens:
                    score = 1.0
                else:
                    inter = len(t_tokens & p_tokens)
                    uni = len(t_tokens | p_tokens)
                    score = inter / uni if uni else 0.0
                pair_scores.append(score)

        # create 10 bins (0.0-0.1, 0.1-0.2, ... 0.9-1.0)
        bins = [0] * 10
        for s in pair_scores:
            idx = min(int(s * 10), 9)
            bins[idx] += 1

        lines.append("## Jaccard Histogram (target risks vs peer risks)")
        lines.append("")
        if pair_scores:
            total = len(pair_scores)
            max_bar = 40
            for i, count in enumerate(bins):
                lo = i / 10.0
                hi = (i + 1) / 10.0
                barlen = int((count / total) * max_bar)
                bar = '#' * barlen
                lines.append(f"- {lo:.1f}-{hi:.1f}: {count} {' ' if count<10 else ''}{bar}")
            lines.append("")
        else:
            lines.append("- No pairwise scores available (insufficient data)")
            lines.append("")
    except Exception:
        # don't fail report generation if histogram errors
        lines.append("")
        lines.append("- Jaccard histogram: error computing histogram")
        lines.append("")

    # finally write the completed Markdown report
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text("\n".join(lines), encoding="utf-8")

    # Also emit a machine-readable YAML artifact alongside the Markdown so
    # downstream consumers can parse reports deterministically.
    yaml_path = outpath.with_suffix(".yaml")

    # Build structured data matching the key sections above
    data = {
        "ticker": target.upper(),
        "year": int(year),
        "summary": {
            "severity_average": float(target_sev),
            "severity_percentile": int(pct),
        },
        "category_distribution": {k: float(v) for k, v in target_dist.items()},
        "compared_companies": [pr.ticker.upper() for pr in peer_results],
        "unique_risks": [u for u in unique_previews],
        "shared_risks": [
            {
                "target_preview": s["target_preview"],
                "peer": s["peer"],
                "peer_preview": s["peer_preview"],
                "score": float(s["score"]),
            }
            for s in shared_entries
        ],
        "specific_comparisons": {
            "textual_novelty": novelty_line,
            "peer_similarity_avg_jaccard": float(avg_jaccard) if peer_scores else None,
            "risk_density": {
                "target_words": int(tgt_words),
                "peers_median_words": float(med_words) if peer_words else None,
                "target_paras": int(tgt_paras),
                "peers_median_paras": float(med_paras) if peer_paras else None,
            },
            "linguistic_tone_per_k": {
                "target": float(tgt_tone.get("per_k", 0.0)),
                "peers_median": float(med_perk) if peer_words else None,
            },
        },
        "jaccard_histogram": {},
    }

    # populate histogram mapping using the bins computed above (if present)
    try:
        if pair_scores:
            total = len(pair_scores)
            for i, count in enumerate(bins):
                lo = i / 10.0
                hi = (i + 1) / 10.0
                key = f"{lo:.1f}-{hi:.1f}"
                data["jaccard_histogram"][key] = int(count)
        else:
            data["jaccard_histogram"] = {}
    except Exception:
        data["jaccard_histogram"] = {}

    # Try to use PyYAML if available, otherwise emit a simple YAML serializer
    try:
        import yaml

        with yaml_path.open("w", encoding="utf-8") as fh:
            # allow_unicode ensures characters like bullets and non-ASCII
            # punctuation are written as Unicode rather than \x escapes.
            yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True)
    except Exception:
        def _dump_yaml(obj, indent=0):
            pad = "  " * indent
            if isinstance(obj, dict):
                lines = []
                for k, v in obj.items():
                    if isinstance(v, (dict, list)):
                        lines.append(f"{pad}{k}:")
                        lines.append(_dump_yaml(v, indent + 1))
                    else:
                        if v is None:
                            lines.append(f"{pad}{k}: null")
                        elif isinstance(v, bool):
                            lines.append(f"{pad}{k}: {str(v).lower()}")
                        elif isinstance(v, (int, float)):
                            lines.append(f"{pad}{k}: {v}")
                        else:
                            sval = str(v)
                            if "\n" in sval:
                                # block literal for multi-line
                                block = "\n".join([f"{pad}  {l}" for l in sval.splitlines()])
                                lines.append(f"{pad}{k}: |")
                                lines.append(block)
                            else:
                                # simple scalar (escape leading/trailing spaces)
                                lines.append(f"{pad}{k}: {sval}")
                return "\n".join(lines)
            elif isinstance(obj, list):
                lines = []
                for it in obj:
                    if isinstance(it, (dict, list)):
                        lines.append(f"{pad}- ")
                        lines.append(_dump_yaml(it, indent + 1))
                    else:
                        lines.append(f"{pad}- {it}")
                return "\n".join(lines)
            else:
                return f"{pad}{obj}\n"

        yaml_path.write_text(_dump_yaml(data), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a peer comparison report")
    parser.add_argument("ticker", help="Target ticker (e.g., NVDA)")
    parser.add_argument("year", type=int, help="Target filing year (e.g., 2024)")
    parser.add_argument("--peers", nargs="+", help="Explicit peer tickers (optional)")
    parser.add_argument("--max-peers", type=int, default=6, help="Max peers to include")
    parser.add_argument("--download-dir", type=str, default="./data/filings", help="Base download dir")
    parser.add_argument("--db-path", type=str, default="./database/sec_filings.db", help="Filings DB path")
    parser.add_argument("--output", type=str, default=None, help="Output markdown path (default: output/{TICKER}_Peer_Comparison_{YEAR}.md)")
    parser.add_argument("--db-only-classification", action="store_true", help="Use DB-only classification (no LLM calls)")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    downloader = TenKDownloader(db_path=args.db_path, download_dir=args.download_dir)
    svc = PeerDiscoveryService(db_path=args.db_path)
    # The pipeline expects a directory path for vector DB persistence. If the
    # user passed a sqlite DB file (e.g. ./database/sec_filings.db), use its
    # parent directory so we don't attempt to create a file path as a folder.
    from pathlib import Path
    persist_dir = str(Path(args.db_path).parent)

    pipeline = IntegrationPipeline(
        persist_path=persist_dir,
        db_only_classification=bool(args.db_only_classification),
    )
    if args.db_only_classification:
        logger.info("DB-only classification enabled for pipeline (LLM calls prevented)")
    else:
        logger.info("Default classification: vector-store-first, then LLM fallback")

    target = args.ticker.upper()
    year = args.year

    # Analyze target first — if target filing is unavailable, abort.
    try:
        target_html = ensure_filing(downloader, target, year)
    except RuntimeError as e:
        logger.error("Failed to ensure filing for target %s: %s", target, e)
        print(f"Error: failed to obtain filing for target {target}: {e}")
        sys.exit(2)

    try:
        target_result = pipeline.analyze_filing(str(target_html), target, year)
    except IntegrationError as e:
        logger.error("Analysis failed for target %s: %s", target, e)
        print(f"Error: analysis failed for target {target}: {e}")
        sys.exit(3)

    # Build candidate peer list. If user supplied explicit peers, prefer those
    # (but attempt replacements if they lack filings). Otherwise request a
    # larger candidate pool from the discovery service and pick replacements
    # as needed.
    desired = args.max_peers
    collected: List[RiskAnalysisResult] = []
    attempted: set = set()

    if args.peers:
        candidates = [p.upper() for p in args.peers if p.upper() != target]
    else:
        # request a larger pool to allow replacements when some peers lack filings
        candidates = svc.find_peers_for_ticker(target, top_n=max(desired * 3, desired + 6))

    for cand in candidates:
        if len(collected) >= desired:
            break
        cand_u = cand.upper()
        if cand_u == target:
            continue
        if cand_u in attempted:
            continue
        attempted.add(cand_u)

        try:
            html_path = ensure_filing(downloader, cand_u, year)
        except RuntimeError as e:
            logger.warning("Could not obtain filing for peer %s: %s — trying next candidate", cand_u, e)
            continue

        try:
            res = pipeline.analyze_filing(str(html_path), cand_u, year)
            collected.append(res)
        except IntegrationError as e:
            logger.warning("Analysis failed for peer %s: %s — skipping", cand_u, e)
            continue

    results = [target_result] + collected

    outpath = Path(args.output) if args.output else Path("output") / f"{target}_Peer_Comparison_{year}.md"
    generate_markdown_report(target, year, results, outpath)
    print(f"Wrote report: {outpath}")


if __name__ == "__main__":
    main()
