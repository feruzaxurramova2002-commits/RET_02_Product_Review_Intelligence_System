"""Text preprocessing and noise sanitizer for raw product-review text.

Real review text is messy: HTML remnants, emoji, ALL-CAPS shouting, and —
critically — reviews so short ("ok", "???") that naive vectorizers can
silently degenerate. `ReviewSanitizer` normalizes text and flags data-quality
edge cases so downstream code can route defensively instead of crashing or
silently producing a confident-looking prediction on garbage input.
"""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Optional

import pandas as pd

try:
    import emoji as emoji_lib
    _HAS_EMOJI_LIB = True
except ImportError:  # pragma: no cover - optional dependency
    _HAS_EMOJI_LIB = False


class ReviewSanitizer:
    """Cleans and normalizes raw review text, with explicit edge-case detection.

    Attributes:
        min_alpha_ratio: Minimum fraction of alphabetic characters required for
            a review to be considered non-gibberish.
        min_token_count: Reviews with fewer tokens than this are flagged
            'ultra_short' (still processed, but routed for human review).
    """

    _HTML_TAG_RE = re.compile(r"<[^>]+>")
    _URL_RE = re.compile(r"https?://\S+|www\.\S+")
    _NON_ALPHA_RE = re.compile(r"[^a-zA-Z\s]")
    _MULTI_SPACE_RE = re.compile(r"\s+")
    _REPEATED_CHAR_RE = re.compile(r"(.)\1{2,}")  # e.g. "sooooo" -> "soo"

    def __init__(self, min_alpha_ratio: float = 0.4, min_token_count: int = 3) -> None:
        self.min_alpha_ratio = min_alpha_ratio
        self.min_token_count = min_token_count

    @staticmethod
    def _demojize(text: str) -> str:
        """Converts emoji characters to a readable text token."""
        if _HAS_EMOJI_LIB:
            return emoji_lib.demojize(text, delimiters=(" ", " "))
        return "".join(ch if unicodedata.category(ch)[0] != "So" else " " for ch in text)

    def clean_text(self, text: Optional[str]) -> str:
        """Applies the full cleaning chain to a single raw text field.

        Args:
            text: Raw review text. May be None, NaN, empty, or malformed.

        Returns:
            A normalized, lowercase string. Returns an empty string (never
            None/NaN) for missing input, so downstream calls never crash.
        """
        if text is None:
            return ""
        if isinstance(text, float) and math.isnan(text):
            return ""
        text = str(text)
        if not text.strip():
            return ""

        text = self._HTML_TAG_RE.sub(" ", text)
        text = self._URL_RE.sub(" ", text)
        text = self._demojize(text)
        text = unicodedata.normalize("NFKC", text)
        text = text.lower()
        text = self._REPEATED_CHAR_RE.sub(r"\1\1", text)
        text = self._NON_ALPHA_RE.sub(" ", text)
        text = self._MULTI_SPACE_RE.sub(" ", text).strip()
        return text

    def classify_edge_case(self, cleaned_text: str) -> str:
        """Classifies a cleaned review into a data-quality edge-case bucket.

        Args:
            cleaned_text: Output of `clean_text`.

        Returns:
            One of: 'empty', 'ultra_short', 'gibberish', 'ok'.
        """
        if not cleaned_text:
            return "empty"
        tokens = cleaned_text.split()
        if len(tokens) == 0:
            return "empty"
        alpha_chars = sum(c.isalpha() for c in cleaned_text)
        alpha_ratio = alpha_chars / max(len(cleaned_text.replace(" ", "")), 1)
        if alpha_ratio < self.min_alpha_ratio:
            return "gibberish"
        if len(tokens) < self.min_token_count:
            return "ultra_short"
        return "ok"

    def sanitize_dataframe(self, df: pd.DataFrame, text_col: str = "review_body") -> pd.DataFrame:
        """Applies cleaning + edge-case classification to an entire DataFrame column.

        Args:
            df: Input DataFrame containing `text_col`.
            text_col: Name of the raw text column to sanitize.

        Returns:
            A copy of `df` with two new columns: `<text_col>_clean` and
            `<text_col>_edge_case`.
        """
        out = df.copy()
        out[f"{text_col}_clean"] = out[text_col].apply(self.clean_text)
        out[f"{text_col}_edge_case"] = out[f"{text_col}_clean"].apply(self.classify_edge_case)
        return out
