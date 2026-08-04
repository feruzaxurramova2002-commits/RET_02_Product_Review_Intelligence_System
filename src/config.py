"""Central, immutable configuration for the RET-02 Product Review Intelligence System.

Every tunable constant used across the sanitizer, model manager, inference engine,
and Streamlit UI lives here so the system stays easy to audit, tune, and reproduce.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Tuple

# --------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------
SRC_DIR: str = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT: str = os.path.dirname(SRC_DIR)
MODELS_DIR: str = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

BASELINE_ARTIFACT_PATH: str = os.path.join(MODELS_DIR, "baseline_pipeline.joblib")


@dataclass(frozen=True)
class Config:
    """Immutable configuration shared by every module in the pipeline.

    Attributes:
        seed: Global random seed for reproducibility.
        n_samples: Number of synthetic reviews generated to train/demo the models.
        issue_categories: Controlled vocabulary of issue/aspect labels.
        sentiment_classes: Controlled vocabulary of sentiment labels.
        confidence_threshold: Minimum overall confidence required to auto-process
            a review; below this the review is routed to HUMAN_REVIEW_REQUIRED.
        min_alpha_ratio: Minimum fraction of alphabetic characters for text to be
            considered non-gibberish (used by the sanitizer).
        min_token_count: Reviews with fewer tokens than this are flagged ultra-short.
        max_tfidf_features: Vocabulary cap for the TF-IDF vectorizer.
        urgent_actionability_threshold: Actionability score above which a review is
            considered an "urgent" item in the executive summary.
    """

    seed: int = 42
    n_samples: int = 900
    issue_categories: Tuple[str, ...] = (
        "Delivery", "Packaging", "Product Quality", "Defect",
        "Price/Value", "Customer Support", "None",
    )
    sentiment_classes: Tuple[str, ...] = ("Positive", "Neutral", "Negative")
    confidence_threshold: float = 0.65
    min_alpha_ratio: float = 0.4
    min_token_count: int = 3
    max_tfidf_features: int = 5000
    urgent_actionability_threshold: float = 0.6


CFG = Config()

# --------------------------------------------------------------------------------
# Adaptive-design profile themes.
# Each profile maps to a CSS color palette applied globally in app.py.
# --------------------------------------------------------------------------------
PROFILE_THEMES = {
    "pm_cx": {
        "label_key": "profile_pm_cx",
        "background": "#0B0F19",
        "surface": "#111827",
        "surface_alt": "#1F2937",
        "border": "#374151",
        "text": "#E2E8F0",
        "text_muted": "#9CA3AF",
        "accent": "#3B82F6",
        "accent_2": "#60A5FA",
        "success": "#22C55E",
        "warning": "#F59E0B",
        "danger": "#EF4444",
        "font": "'Inter', -apple-system, BlinkMacSystemFont, sans-serif",
    },
    "data_ml": {
        "label_key": "profile_data_ml",
        "background": "#05070A",
        "surface": "#0D1117",
        "surface_alt": "#161B22",
        "border": "#30363D",
        "text": "#F0F6FC",
        "text_muted": "#8B949E",
        "accent": "#39D353",
        "accent_2": "#00E5FF",
        "success": "#39D353",
        "warning": "#E3B341",
        "danger": "#F85149",
        "font": "'JetBrains Mono', 'Fira Code', 'Courier New', monospace",
    },
    "exec_biz": {
        "label_key": "profile_exec_biz",
        "background": "#F7FAFC",
        "surface": "#FFFFFF",
        "surface_alt": "#F1F5F9",
        "border": "#E2E8F0",
        "text": "#1E293B",
        "text_muted": "#64748B",
        "accent": "#2563EB",
        "accent_2": "#059669",
        "success": "#059669",
        "warning": "#D97706",
        "danger": "#DC2626",
        "font": "'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif",
    },
    "genz_student": {
        "label_key": "profile_genz_student",
        "background": "#12081F",
        "surface": "#1E0E33",
        "surface_alt": "#2A1245",
        "border": "#7C3AED",
        "text": "#F5F3FF",
        "text_muted": "#C4B5FD",
        "accent": "#EC4899",
        "accent_2": "#22D3EE",
        "success": "#34D399",
        "warning": "#FBBF24",
        "danger": "#FB7185",
        "font": "'Poppins', 'Inter', sans-serif",
    },
}

DEFAULT_PROFILE = "pm_cx"
