"""Text cleaning utilities for downstream reports and ingestion.

Provide a small, well-typed helper to remove unprintable and invisible
Unicode characters, collapse whitespace, and safely unescape common
escaped Unicode sequences that sometimes appear in extracted text.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Callable


def _remove_format_and_control_chars(s: str) -> str:
    # Remove only control (Cc) and format (Cf) characters which are typically
    # invisible (e.g., zero-width space U+200B, BOM U+FEFF, etc.). Keep other
    # Unicode categories (letters, punctuation, symbols) intact.
    return "".join(ch for ch in s if unicodedata.category(ch) not in ("Cf", "Cc"))


def clean_text(text: str) -> str:
    """Return a cleaned version of *text*.

    Cleaning steps (best-effort, non-destructive):
    - Normalize Unicode using NFKC.
    - Attempt to decode literal escape sequences (e.g. "\\u200B") safely.
    - Remove format & control characters (zero-width spaces, BOM, control chars).
    - Replace backslash-line-continuations and escaped underscores.
    - Collapse runs of whitespace to a single space and strip edges.

    This function is conservative: it avoids removing normal punctuation and
    letters; it's intended for preparing human-readable report text.
    """
    if not text:
        return text

    # Normalize to canonical form
    txt = unicodedata.normalize("NFKC", text)

    # Convert common escaped unicode sequences like "\\u200B" into the
    # actual character, if present. Use a best-effort decode — failure
    # shouldn't raise.
    # Some extracted strings contain literal hex byte escape sequences
    # like "\xE2\x80\xA2" (UTF-8 for •). Convert runs of \xHH into bytes
    # and then decode as UTF-8 when possible.
    try:
        def _hex_unescape(match: re.Match) -> str:
            seq = match.group(0)
            bytes_vals = bytes(int(x, 16) for x in re.findall(r"\\x([0-9A-Fa-f]{2})", seq))
            try:
                return bytes_vals.decode("utf-8")
            except Exception:
                return seq

        txt = re.sub(r"(?:\\x[0-9A-Fa-f]{2})+", _hex_unescape, txt)

        txt = txt.encode("utf-8", errors="surrogatepass").decode("unicode_escape")
        # Replace any remaining literal hex-escapes like "\xE2" that couldn't
        # be decoded into valid UTF-8 sequences with a single space so we don't
        # accidentally join neighboring words.
        txt = re.sub(r"(?:\\x[0-9A-Fa-f]{2})+", " ", txt)
    except Exception:
        # decode may fail on malformed sequences; keep original
        pass

    # Remove backslash-newline line continuations introduced by some serializations
    txt = re.sub(r"\\\s*\n\s*", " ", txt)

    # Replace literal escaped underscores '\\_' with plain underscore
    txt = txt.replace("\\_", "_")

    # Remove invisible/format/control characters
    txt = _remove_format_and_control_chars(txt)

    # Heuristic: fix common mojibake where UTF-8 bytes were decoded as
    # Latin-1 (results in sequences like 'Ã', 'Â', 'â'). Try a round-trip
    # re-encode/decode to recover proper Unicode if such patterns appear.
    try:
        if re.search(r"[ÃÂâ]", txt):
            txt = txt.encode("latin-1", errors="replace").decode("utf-8", errors="replace")
    except Exception:
        pass

    # Replace Unicode replacement characters (�, U+FFFD) and any remaining
    # literal "\uXXXX" escapes with spaces so they don't produce garbled
    # tokens in reports. Collapsing to space preserves word boundaries.
    txt = txt.replace("\ufffd", " ")
    # Some replacement characters become the three-character mojibake
    # sequence 'ï¿½' when bytes are mis-decoded; replace that too.
    txt = txt.replace("ï¿½", " ")
    txt = re.sub(r"(?:\\u[0-9A-Fa-f]{4})+", " ", txt)

    # Collapse whitespace
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


# Export a compact alias for convenience in call sites
sanitize = clean_text
