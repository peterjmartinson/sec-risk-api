from bs4 import BeautifulSoup
from pathlib import Path
import logging

# Setup basic logging for the Armory
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_text_from_html(html_path: str) -> str:
    """
    Extracts clean text from an SEC EDGAR HTM filing.

    This is the refined base layer of the Ingestion Spark.
    """
    path = Path(html_path)
    if not path.exists():
        logger.error(f"File not found: {html_path}")
        raise FileNotFoundError(f"No HTM file found at {html_path}")

    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Use lxml for speed and better handling of malformed SEC tags
        soup = BeautifulSoup(content, "lxml")

        # Remove script and style elements that contaminate RAG context
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()

        # get_text with a separator prevents words from mashing together
        # when <td> or <div> tags end.
        full_text = soup.get_text(separator=" ", strip=True)

        logger.info(f"Successfully extracted {len(full_text)} characters from {html_path}")
        return full_text

    except Exception as e:
        logger.error(f"Failed to parse HTM: {e}")
        raise
