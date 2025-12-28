from bs4 import BeautifulSoup
from pathlib import Path
import logging

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
