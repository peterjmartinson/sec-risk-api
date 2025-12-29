from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Any

def chunk_risk_section(text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Atomic function: Takes raw text and returns a list of
    chunked dictionaries with metadata attached.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    return [
        {"text": chunk, "metadata": metadata}
        for chunk in splitter.split_text(text)
    ]
