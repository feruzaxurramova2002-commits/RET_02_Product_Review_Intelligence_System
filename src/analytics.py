"""Batch-level analytics and executive-summary aggregation.

Turns a list of `ReviewAnalysisResponse` objects into the aggregates product
and CX teams actually act on: top recurring issues, sentiment mix, routing
load, and a Net Sentiment Score.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from src.config import CFG
from src.inference_engine import ReviewAnalysisResponse


@dataclass
class ExecutiveSummary:
    """Aggregated, business-facing summary of a batch of analyzed reviews.

    Attributes:
        total_reviews: Number of reviews analyzed.
        auto_processed: Count routed to AUTO_PROCESSED.
        human_review_required: Count routed to HUMAN_REVIEW_REQUIRED.
        avg_actionability: Mean actionability score across the batch.
        urgent_count: Reviews above the urgent-actionability threshold.
        net_sentiment_score: % positive - % negative, in [-100, 100].
        sentiment_distribution: Count per sentiment label.
        issue_frequency: Count per issue label, sorted descending.
        top_issues: The top 5 (label, count) issue pairs.
    """

    total_reviews: int
    auto_processed: int
    human_review_required: int
    avg_actionability: float
    urgent_count: int
    net_sentiment_score: float
    sentiment_distribution: Dict[str, int]
    issue_frequency: Dict[str, int]
    top_issues: List[tuple]


def responses_to_dataframe(responses: List[ReviewAnalysisResponse]) -> pd.DataFrame:
    """Flattens a list of `ReviewAnalysisResponse` into a tabular DataFrame.

    One row per review; `issues` is a comma-joined string of predicted labels
    for easy CSV export.
    """
    rows = []
    for r in responses:
        rows.append({
            "review_id": r.review_id,
            "sentiment": r.sentiment,
            "sentiment_probability": round(r.sentiment_probability, 4),
            "predicted_issues": ", ".join(i.label for i in r.predicted_issues) or "None",
            "actionability_score": round(r.actionability_score, 4),
            "overall_confidence": round(r.overall_confidence, 4),
            "status": r.status,
            "edge_case_flag": r.edge_case_flag,
            "model_backbone_mode": r.model_backbone_mode,
        })
    return pd.DataFrame(rows)


def generate_executive_summary(responses: List[ReviewAnalysisResponse]) -> ExecutiveSummary:
    """Aggregates a batch of analyzed reviews into an `ExecutiveSummary`.

    Args:
        responses: Output of `ProductReviewIntelligenceEngine.batch_analyze`.

    Returns:
        An `ExecutiveSummary` dataclass. Returns a zeroed-out summary if
        `responses` is empty rather than raising.
    """
    total = len(responses)
    if total == 0:
        return ExecutiveSummary(0, 0, 0, 0.0, 0, 0.0, {}, {}, [])

    auto = sum(1 for r in responses if r.status == "AUTO_PROCESSED")
    human = total - auto
    avg_actionability = sum(r.actionability_score for r in responses) / total
    urgent = sum(1 for r in responses if r.actionability_score >= CFG.urgent_actionability_threshold)

    sentiment_counts = Counter(r.sentiment for r in responses)
    pos = sentiment_counts.get("Positive", 0)
    neg = sentiment_counts.get("Negative", 0)
    nss = ((pos - neg) / total) * 100.0

    issue_counter: Counter = Counter()
    for r in responses:
        for issue in r.predicted_issues:
            issue_counter[issue.label] += 1
    issue_frequency = dict(issue_counter.most_common())
    top_issues = issue_counter.most_common(5)

    return ExecutiveSummary(
        total_reviews=total,
        auto_processed=auto,
        human_review_required=human,
        avg_actionability=round(avg_actionability, 4),
        urgent_count=urgent,
        net_sentiment_score=round(nss, 2),
        sentiment_distribution=dict(sentiment_counts),
        issue_frequency=issue_frequency,
        top_issues=top_issues,
    )
