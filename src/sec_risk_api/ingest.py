from bs4 import BeautifulSoup
from pathlib import Path
import logging
import re
from sec_risk_api.processing import chunk_risk_section

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_sec_html(html_content: str) -> str:
    """
    Pure logic: Extracts clean text from raw HTML content.
    Targeted for unit testing and modular use.
    """
    # Use lxml for speed and better handling of malformed SEC tags
    soup = BeautifulSoup(html_content, "lxml")

    # Remove script and style elements that contaminate RAG context
    for script_or_style in soup(["script", "style"]):
        script_or_style.decompose()

    # get_text with a separator prevents words from mashing together
    # when <td> or <div> tags end.
    text: str = soup.get_text(separator=" ", strip=True)
    return text

def extract_text_from_file(html_path: str | Path) -> str:
    """
    IO Wrapper: Handles file loading, encoding fallbacks, and calls the parser.
    """
    path = Path(html_path)
    if not path.exists():
        logger.error(f"File not found: {html_path}")
        raise FileNotFoundError(f"No HTM file found at {html_path}")

    try:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # Attempt 2: CP1252 (The legacy Windows fallback common in SEC filings)
            logger.warning(f"UTF-8 decode failed for {html_path}. Falling back to CP1252.")
            with open(path, 'r', encoding='cp1252') as f:
                content = f.read()

        full_text = parse_sec_html(content)

        logger.info(f"Successfully extracted {len(full_text)} characters from {html_path}")
        return full_text

    except Exception as e:
        logger.error(f"Failed to process HTM file {html_path}: {e}")
        raise

def slice_risk_factors(text: str) -> str:
    """
    Slices the 'Item 1A' section from the full text.
    Gracefully falls back to full text if markers are missing.
    """
    # Regex to find Item 1A, ignoring case and handling varied spacing/punctuation
    # The [^>]{0,50} helps skip over HTML remnants or short TOC entries
    start_pattern = re.compile(r"ITEM\s+1A[\.\s\-:]+RISK\s+FACTORS", re.IGNORECASE)

    # End markers: Item 1B or Item 2 are the most common next sections
    end_pattern = re.compile(r"ITEM\s+(1B|2)[\.\s\-:]", re.IGNORECASE)

    # Use search to find the FIRST occurrence (usually the body, not the TOC)
    # Note: If the first match IS the TOC, we may need to find the second match.
    # For the walking skeleton, we start with the first valid match.
    matches = list(start_pattern.finditer(text))
    if not matches:
        logger.warning("Item 1A marker not found. Fallback: returning full text.")
        return text

    # Typically, the first match in a text-only dump is the section header.
    # (HTML scrapers usually strip the TOC links that cause double-matches).
    start_match = matches[0]

    # Find the end marker ONLY after the start marker
    end_match = end_pattern.search(text, pos=start_match.end())

    if end_match:
        logger.info("Successfully sliced Item 1A section.")
        return text[start_match.start() : end_match.start()].strip()

    logger.info("Item 1A found but no end marker detected. Slicing from 1A to EOF.")
    return text[start_match.start() :].strip()

def run_ingestion_pipeline(html_content: str, ticker: str, year: int) -> str:
    """
    Orchestrates the flow from raw HTML to Vector DB atoms.
    """
    # 1. Extraction: From raw HTML to clean 'Item 1A' text
    # We modularize the calls to your existing logic
    full_text = parse_sec_html(html_content)
    risk_text = slice_risk_factors(full_text)

    if not risk_text or len(risk_text) < 100:
        logger.warning(f"Risk section for {ticker} seems suspiciously short or missing.")
        # Return or handle error as per your preference

    # 2. Chunking: From long text to metadata-tagged atoms (Subissue 1.0)
    metadata = {
        "ticker": ticker,
        "year": year,
        "item_type": "1A"
    }
    chunks = chunk_risk_section(risk_text, metadata)

    # 3. Storage & Embeddings (Subissue 1.1 & 1.2)
    # Placeholder for the upcoming 'Vault' logic
    # upsert_to_vault(chunks)

    logger.info(f"Pipeline: Processed {len(chunks)} chunks for {ticker} {year}")
    return f"Processed {len(chunks)} chunks for {ticker}"
