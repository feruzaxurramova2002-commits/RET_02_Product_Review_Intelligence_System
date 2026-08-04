"""Production inference engine with confidence-based human-review routing.

`ProductReviewIntelligenceEngine` wraps sanitization, model inference, and
routing behind a single `analyze()` / `batch_analyze()` call, and returns a
validated Pydantic schema on every call — including on internal errors, so a
single malformed review can never crash a batch or the UI.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field, field_validator

from src.config import CFG
from src.model_manager import ModelManager
from src.sanitizer import ReviewSanitizer

STATUS_AUTO = "AUTO_PROCESSED"
STATUS_HUMAN = "HUMAN_REVIEW_REQUIRED"


class IssuePrediction(BaseModel):
    """A single predicted issue/aspect label with its model probability."""

    label: str
    probability: float = Field(..., ge=0.0, le=1.0)


class ReviewAnalysisResponse(BaseModel):
    """Structured, validated output schema for one analyzed review."""

    review_id: Optional[str] = None
    predicted_issues: List[IssuePrediction]
    sentiment: str
    sentiment_probability: float = Field(..., ge=0.0, le=1.0)
    actionability_score: float = Field(..., ge=0.0, le=1.0)
    overall_confidence: float = Field(..., ge=0.0, le=1.0)
    status: str
    edge_case_flag: str
    model_backbone_mode: str
    explanation: List[str] = Field(default_factory=list)

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        allowed = {STATUS_AUTO, STATUS_HUMAN}
        if v not in allowed:
            raise ValueError(f"status must be one of {allowed}, got '{v}'")
        return v


class ProductReviewIntelligenceEngine:
    """Production-facing entry point for the RET-02 review-intelligence system.

    Attributes:
        model_manager: `ModelManager` exposing the active trained backbone.
        sanitizer: `ReviewSanitizer` instance for text normalization.
        confidence_threshold: Minimum overall confidence to auto-process a review.
    """

    def __init__(
        self,
        model_manager: ModelManager,
        sanitizer: ReviewSanitizer,
        confidence_threshold: float = CFG.confidence_threshold,
    ) -> None:
        self.model_manager = model_manager
        self.sanitizer = sanitizer
        self.confidence_threshold = confidence_threshold

    def analyze(
        self,
        review_text: str,
        rating: Optional[int] = None,
        review_id: Optional[str] = None,
    ) -> ReviewAnalysisResponse:
        """Analyzes a single raw review and returns a validated structured response.

        Args:
            review_text: Raw, unprocessed review body text.
            rating: Optional star rating (reserved for future feature fusion).
            review_id: Optional identifier echoed back in the response.

        Returns:
            A `ReviewAnalysisResponse` with predicted labels, confidence, and
            routing status. Never raises — internal errors are converted into
            a HUMAN_REVIEW_REQUIRED response with `edge_case_flag="error"`.
        """
        del rating  # reserved for future use; not consumed by the current model
        try:
            cleaned = self.sanitizer.clean_text(review_text)
            edge_case = self.sanitizer.classify_edge_case(cleaned)

            if edge_case in ("empty", "gibberish"):
                return ReviewAnalysisResponse(
                    review_id=review_id, predicted_issues=[], sentiment="Neutral",
                    sentiment_probability=0.0, actionability_score=0.0, overall_confidence=0.0,
                    status=STATUS_HUMAN, edge_case_flag=edge_case,
                    model_backbone_mode=self.model_manager.mode, explanation=[],
                )

            bundle = self.model_manager.predict([cleaned])
            issue_probs = bundle.issue_probs[0]
            sentiment_probs = bundle.sentiment_probs[0]
            actionability = float(bundle.actionability[0])

            issue_classes = self.model_manager.issue_classes
            sentiment_classes = self.model_manager.sentiment_classes

            predicted_issues = [
                IssuePrediction(label=issue_classes[i], probability=float(p))
                for i, p in enumerate(issue_probs)
                if p > 0.5 and issue_classes[i] != "None"
            ]
            sentiment_idx = int(np.argmax(sentiment_probs))
            sentiment_label = sentiment_classes[sentiment_idx]
            sentiment_confidence = float(sentiment_probs[sentiment_idx])

            # Overall confidence blends sentiment certainty with issue-label
            # decisiveness (distance from the 0.5 decision boundary).
            issue_decisiveness = float(np.mean(np.abs(issue_probs - 0.5) * 2))
            overall_confidence = 0.5 * sentiment_confidence + 0.5 * issue_decisiveness

            is_ambiguous = edge_case == "ultra_short"
            status = (
                STATUS_HUMAN
                if (overall_confidence < self.confidence_threshold or is_ambiguous)
                else STATUS_AUTO
            )

            explanation = self.model_manager.explain(cleaned)

            return ReviewAnalysisResponse(
                review_id=review_id,
                predicted_issues=predicted_issues,
                sentiment=sentiment_label,
                sentiment_probability=sentiment_confidence,
                actionability_score=actionability,
                overall_confidence=overall_confidence,
                status=status,
                edge_case_flag=edge_case,
                model_backbone_mode=self.model_manager.mode,
                explanation=explanation,
            )
        except Exception as exc:  # noqa: BLE001 - never let one bad review crash a batch
            return ReviewAnalysisResponse(
                review_id=review_id, predicted_issues=[], sentiment="Neutral",
                sentiment_probability=0.0, actionability_score=0.0, overall_confidence=0.0,
                status=STATUS_HUMAN, edge_case_flag=f"error:{type(exc).__name__}",
                model_backbone_mode=self.model_manager.mode, explanation=[],
            )

    def batch_analyze(self, reviews: List[Dict[str, Any]]) -> List[ReviewAnalysisResponse]:
        """Analyzes a batch of reviews.

        Args:
            reviews: List of dicts with keys `review_text`, optional `rating`,
                `review_id`.

        Returns:
            List of `ReviewAnalysisResponse`, one per input review, same order.
        """
        return [
            self.analyze(r.get("review_text", ""), r.get("rating"), r.get("review_id"))
            for r in reviews
        ]
