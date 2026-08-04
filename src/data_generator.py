"""Synthetic e-commerce product-review dataset generator.

No public dataset ships with a ready-made issue_category x sentiment x
actionability label set at the granularity this project needs, so this module
generates a realistic, internally-consistent synthetic corpus that is used to
train and demo the baseline model. See README.md for the rationale.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import CFG


class SyntheticReviewDataGenerator:
    """Generates a realistic synthetic e-commerce product-review dataset.

    Mirrors real marketplace data along four axes: (1) rating inflation (skew
    toward 4-5 stars), (2) category-conditioned vocabulary, (3) correlated
    multi-label issues and sentiment (not sampled independently), and (4)
    realistic text noise (short reviews, HTML remnants, emoji, ALL-CAPS).

    Attributes:
        n_samples: Number of review rows to generate.
        categories: Product categories to sample from.
        issue_categories: Controlled issue-label vocabulary.
        sentiment_classes: Controlled sentiment vocabulary.
        rng: A dedicated NumPy random Generator seeded independently.
    """

    _POSITIVE_SNIPPETS = [
        "works perfectly", "exceeded my expectations", "great value for the price",
        "fast shipping and well packaged", "exactly as described", "very happy with this purchase",
        "highly recommend", "solid build quality", "love it", "would buy again",
    ]
    _NEGATIVE_SNIPPETS: Dict[str, List[str]] = {
        "Delivery": ["arrived two weeks late", "delivery was delayed repeatedly", "shipment got lost",
                     "took forever to arrive", "courier never showed up"],
        "Packaging": ["box was crushed on arrival", "packaging was flimsy and torn",
                      "no protective padding at all", "item was loose inside a huge box"],
        "Product Quality": ["material feels cheap", "not as durable as expected",
                             "quality is far below the price point", "stitching came apart after a week"],
        "Defect": ["stopped working after two days", "arrived broken", "defective unit, does not turn on",
                   "battery does not hold a charge", "screen was cracked out of the box"],
        "Price/Value": ["overpriced for what you get", "cheaper alternatives perform just as well",
                         "not worth the money", "price does not match the quality"],
        "Customer Support": ["support never responded to my emails", "refund process was a nightmare",
                              "customer service was rude and unhelpful", "no one answered my complaint"],
    }
    _NOISE_SNIPPETS = ["ok", "bad", "good price", "???", "meh", "fine i guess", "5 stars", "n/a", "..."]

    def __init__(
        self,
        n_samples: int = 900,
        categories: Optional[List[str]] = None,
        issue_categories: Optional[Tuple[str, ...]] = None,
        sentiment_classes: Optional[Tuple[str, ...]] = None,
        seed: int = 42,
    ) -> None:
        self.n_samples = n_samples
        self.categories = categories or [
            "Electronics", "Home & Kitchen", "Apparel", "Beauty", "Sports & Outdoors", "Toys & Games",
        ]
        self.issue_categories = issue_categories or CFG.issue_categories
        self.sentiment_classes = sentiment_classes or CFG.sentiment_classes
        self.rng = np.random.default_rng(seed)

    def _sample_rating(self) -> int:
        """Samples a 1-5 star rating with realistic positive skew (rating inflation)."""
        probs = np.array([0.07, 0.06, 0.10, 0.27, 0.50])
        return int(self.rng.choice([1, 2, 3, 4, 5], p=probs))

    def _sample_issues(self, rating: int) -> List[str]:
        """Samples 0-3 correlated issue labels conditioned on the star rating."""
        real_issues = [c for c in self.issue_categories if c != "None"]
        if rating <= 2:
            k = int(self.rng.choice([1, 2, 3], p=[0.45, 0.35, 0.20]))
            return list(self.rng.choice(real_issues, size=k, replace=False))
        elif rating == 3:
            if self.rng.random() < 0.6:
                return list(self.rng.choice(real_issues, size=1))
            return ["None"]
        else:  # rating 4-5
            if self.rng.random() < 0.12:  # occasional mismatch: high rating, minor gripe
                return list(self.rng.choice(real_issues, size=1))
            return ["None"]

    def _sample_sentiment(self, rating: int, issues: List[str]) -> str:
        """Derives sentiment correlated with rating and issues, with intentional noise."""
        base = {1: "Negative", 2: "Negative", 3: "Neutral", 4: "Positive", 5: "Positive"}[rating]
        if issues != ["None"] and rating >= 4 and self.rng.random() < 0.4:
            return "Neutral"  # sarcastic / mixed-signal review
        if self.rng.random() < 0.08:  # 8% label noise to emulate annotation inconsistency
            return str(self.rng.choice(self.sentiment_classes))
        return base

    def _compute_actionability(self, rating: int, issues: List[str], sentiment: str) -> float:
        """Heuristic actionability proxy in [0, 1]: higher = more urgent for CX teams."""
        score = 0.0
        if "Defect" in issues:
            score += 0.55
        if "Customer Support" in issues:
            score += 0.25
        if "Product Quality" in issues:
            score += 0.20
        if "Delivery" in issues or "Packaging" in issues:
            score += 0.15
        if "Price/Value" in issues:
            score += 0.05
        if sentiment == "Negative":
            score += 0.15
        if rating <= 2:
            score += 0.10
        noise = self.rng.normal(0, 0.03)
        return float(np.clip(score + noise, 0.0, 1.0))

    def _compose_review_text(self, rating: int, issues: List[str], sentiment: str) -> Tuple[str, str]:
        """Builds a (title, body) pair of realistic, noisy free text for one review."""
        if self.rng.random() < 0.06:  # ultra-short / noise reviews stress-test the sanitizer
            body = str(self.rng.choice(self._NOISE_SNIPPETS))
            return body[:20], body

        parts: List[str] = []
        if issues == ["None"] or sentiment == "Positive":
            parts.append(str(self.rng.choice(self._POSITIVE_SNIPPETS)))
        for issue in issues:
            if issue == "None":
                continue
            snippets = self._NEGATIVE_SNIPPETS.get(issue, [])
            if snippets:
                parts.append(str(self.rng.choice(snippets)))
        if not parts:
            parts.append(str(self.rng.choice(self._POSITIVE_SNIPPETS)))

        body = ". ".join(parts) + "."
        if self.rng.random() < 0.15:
            body = f"<br>{body}<br>"
        if self.rng.random() < 0.15:
            body += " " + str(self.rng.choice(["!!!", ":(", ":)", "ugh", "wow"]))
        if self.rng.random() < 0.10:
            body = body.upper()
        title = (parts[0][:30] + "...") if len(parts[0]) > 30 else parts[0]
        return title.capitalize(), body

    def generate(self) -> pd.DataFrame:
        """Generates the full synthetic review dataset.

        Returns:
            A DataFrame with columns: review_id, product_id, category, rating,
            review_title, review_body, issue_category (list[str]), sentiment,
            actionability_score.
        """
        rows = []
        n_products = max(20, self.n_samples // 8)
        product_ids = [f"PROD-{i:05d}" for i in range(n_products)]
        product_categories = {pid: str(self.rng.choice(self.categories)) for pid in product_ids}

        for i in range(self.n_samples):
            product_id = str(self.rng.choice(product_ids))
            category = product_categories[product_id]
            rating = self._sample_rating()
            issues = self._sample_issues(rating)
            sentiment = self._sample_sentiment(rating, issues)
            actionability = self._compute_actionability(rating, issues, sentiment)
            title, body = self._compose_review_text(rating, issues, sentiment)

            rows.append({
                "review_id": f"REV-{i:06d}",
                "product_id": product_id,
                "category": category,
                "rating": rating,
                "review_title": title,
                "review_body": body,
                "issue_category": issues,
                "sentiment": sentiment,
                "actionability_score": round(actionability, 4),
            })

        return pd.DataFrame(rows)
