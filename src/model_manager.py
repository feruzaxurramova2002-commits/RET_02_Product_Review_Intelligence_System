"""Model training, persistence, and inference for the RET-02 pipeline.

Two backbone modes are supported end-to-end:

* ``baseline``  — TF-IDF features feeding a ``MultiOutputClassifier`` of
  ``LogisticRegression`` for multi-label issue tagging, a single-label
  ``LogisticRegression`` for sentiment, and a ``Ridge`` regressor for the
  actionability score. All three heads share the same TF-IDF feature space,
  which is a lightweight, CPU-only stand-in for a shared-encoder multi-task
  transformer — the "clean offline fallback" called for by the project brief.
  This is the default and recommended mode for Streamlit Community Cloud's
  free tier: training takes well under a second on the synthetic corpus and
  requires no GPU, no model download, and no PyTorch dependency.
* ``offline``   — A dependency-free rule/keyword-based scorer used only if
  scikit-learn training itself fails for any reason (e.g. a broken
  environment). It keeps the application usable even in a degraded state
  instead of crashing.

A third mode name, ``transformer``, is reserved for a future fine-tuned
multi-task neural backbone (see README "Architecture" section for the
extension point). It is intentionally not enabled by default, to keep cold
start fast and memory usage low on free-tier hosting.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

from src.config import CFG
from src.data_generator import SyntheticReviewDataGenerator
from src.sanitizer import ReviewSanitizer

BASELINE_MODE = "baseline"
OFFLINE_MODE = "offline"

# Keyword lexicon reused by both the "why this prediction" explainer and the
# dependency-free offline fallback scorer.
_ISSUE_KEYWORDS: Dict[str, List[str]] = {
    "Delivery": ["late", "delay", "delayed", "shipment", "shipping", "courier", "arrive", "arrived"],
    "Packaging": ["box", "crushed", "packaging", "padding", "torn", "package"],
    "Product Quality": ["material", "cheap", "durable", "quality", "stitching"],
    "Defect": ["broken", "defective", "stopped working", "cracked", "battery", "does not turn on"],
    "Price/Value": ["overpriced", "expensive", "worth", "price", "cheaper"],
    "Customer Support": ["support", "refund", "customer service", "complaint", "response"],
}
_POSITIVE_WORDS = {
    "perfect", "perfectly", "great", "love", "excellent", "recommend", "happy",
    "good", "fast", "solid", "exceeded", "best", "nice", "awesome",
}
_NEGATIVE_WORDS = {
    "broken", "defective", "bad", "cheap", "late", "delayed", "crushed",
    "overpriced", "rude", "unhelpful", "nightmare", "cracked", "worst", "terrible",
}
_STOPWORDS = {
    "the", "a", "an", "is", "was", "were", "are", "be", "been", "to", "of", "in",
    "on", "at", "and", "or", "but", "for", "with", "it", "its", "this", "that",
    "not", "no", "so", "as", "did", "does", "do", "i", "my", "me", "you", "your",
    "very", "just", "all", "than", "too", "out", "up", "if", "then",
}


@dataclass
class PredictionBundle:
    """Container for the raw numeric outputs of one prediction batch."""

    issue_probs: np.ndarray  # shape (n, n_issue_classes)
    sentiment_probs: np.ndarray  # shape (n, n_sentiment_classes)
    actionability: np.ndarray  # shape (n,)


class BaselineModel:
    """TF-IDF + linear-model multi-task pipeline (issues, sentiment, actionability).

    Attributes:
        vectorizer: Fitted `TfidfVectorizer`.
        issue_clf: Fitted `MultiOutputClassifier(LogisticRegression)`.
        sentiment_clf: Fitted `LogisticRegression`.
        actionability_reg: Fitted `Ridge` regressor.
        issue_classes: Ordered issue label vocabulary.
        sentiment_classes: Ordered sentiment label vocabulary.
    """

    def __init__(self) -> None:
        self.vectorizer: TfidfVectorizer | None = None
        self.issue_clf: MultiOutputClassifier | None = None
        self.sentiment_clf: LogisticRegression | None = None
        self.actionability_reg: Ridge | None = None
        self.issue_classes: List[str] = []
        self.sentiment_classes: List[str] = []

    def fit(self, df_clean: pd.DataFrame) -> "BaselineModel":
        """Fits all three heads on a sanitized training DataFrame.

        Args:
            df_clean: Output of `ReviewSanitizer.sanitize_dataframe`, must
                contain `review_body_clean`, `issue_category`, `sentiment`,
                and `actionability_score`.

        Returns:
            self, for chaining.
        """
        texts = df_clean["review_body_clean"]
        self.vectorizer = TfidfVectorizer(
            max_features=CFG.max_tfidf_features, ngram_range=(1, 2), min_df=2, sublinear_tf=True,
        )
        X = self.vectorizer.fit_transform(texts)

        mlb = MultiLabelBinarizer(classes=list(CFG.issue_categories))
        y_issues = mlb.fit_transform(df_clean["issue_category"])
        self.issue_classes = list(mlb.classes_)
        self.issue_clf = MultiOutputClassifier(
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=CFG.seed)
        )
        self.issue_clf.fit(X, y_issues)

        le = LabelEncoder().fit(list(CFG.sentiment_classes))
        y_sentiment = le.transform(df_clean["sentiment"])
        self.sentiment_classes = list(le.classes_)
        self.sentiment_clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=CFG.seed)
        self.sentiment_clf.fit(X, y_sentiment)

        self.actionability_reg = Ridge(alpha=1.0, random_state=CFG.seed)
        self.actionability_reg.fit(X, df_clean["actionability_score"])

        return self

    def predict(self, cleaned_texts: Sequence[str]) -> PredictionBundle:
        """Runs inference for a batch of already-cleaned texts."""
        X = self.vectorizer.transform(cleaned_texts)
        issue_probs = np.stack(
            [est.predict_proba(X)[:, 1] for est in self.issue_clf.estimators_], axis=1
        )
        sentiment_probs = self.sentiment_clf.predict_proba(X)
        actionability = np.clip(self.actionability_reg.predict(X), 0.0, 1.0)
        return PredictionBundle(issue_probs, sentiment_probs, actionability)

    def explain(self, cleaned_text: str, top_k: int = 4) -> List[str]:
        """Returns the top TF-IDF-weighted n-grams driving the sentiment prediction.

        Combines each present feature's TF-IDF weight with the fitted sentiment
        classifier's per-class coefficient to surface the words that most pushed
        the prediction toward its winning class.
        """
        X = self.vectorizer.transform([cleaned_text])
        row = X.tocoo()
        if row.nnz == 0:
            return []
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        sentiment_idx = int(np.argmax(self.sentiment_clf.predict_proba(X)[0]))
        coefs = self.sentiment_clf.coef_[sentiment_idx] if self.sentiment_clf.coef_.shape[0] > 1 else self.sentiment_clf.coef_[0]
        contributions = []
        for col, val in zip(row.col, row.data):
            term = feature_names[col]
            # Skip single-word stopword features; keep informative unigrams and
            # all bigrams (bigrams containing a stopword, e.g. "not working",
            # are still informative as a phrase).
            if " " not in term and term in _STOPWORDS:
                continue
            contributions.append((term, val * coefs[col]))
        contributions.sort(key=lambda t: t[1], reverse=True)
        top_terms = [term for term, score in contributions[:top_k] if score > 0]
        return top_terms

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "BaselineModel":
        return joblib.load(path)


class OfflineRuleModel:
    """Dependency-free keyword/rule-based scorer used only if `BaselineModel` fails.

    Not trained; deterministic given the shared keyword lexicon. Guarantees the
    application degrades gracefully rather than crashing if scikit-learn is
    unavailable or training raises for any reason.
    """

    issue_classes: List[str] = list(CFG.issue_categories)
    sentiment_classes: List[str] = list(CFG.sentiment_classes)

    def predict(self, cleaned_texts: Sequence[str]) -> PredictionBundle:
        n = len(cleaned_texts)
        issue_probs = np.zeros((n, len(self.issue_classes)))
        sentiment_probs = np.zeros((n, len(self.sentiment_classes)))
        actionability = np.zeros(n)

        for i, text in enumerate(cleaned_texts):
            tokens = set(text.split())
            pos_hits = len(tokens & _POSITIVE_WORDS)
            neg_hits = len(tokens & _NEGATIVE_WORDS)
            for j, issue in enumerate(self.issue_classes):
                if issue == "None":
                    continue
                kws = _ISSUE_KEYWORDS.get(issue, [])
                hits = sum(1 for kw in kws if kw in text)
                issue_probs[i, j] = min(1.0, 0.3 + 0.25 * hits) if hits else 0.05

            if neg_hits > pos_hits:
                s_idx = self.sentiment_classes.index("Negative")
                conf = min(0.95, 0.55 + 0.1 * neg_hits)
            elif pos_hits > neg_hits:
                s_idx = self.sentiment_classes.index("Positive")
                conf = min(0.95, 0.55 + 0.1 * pos_hits)
            else:
                s_idx = self.sentiment_classes.index("Neutral")
                conf = 0.4
            remaining = (1 - conf) / max(len(self.sentiment_classes) - 1, 1)
            sentiment_probs[i, :] = remaining
            sentiment_probs[i, s_idx] = conf

            actionability[i] = min(1.0, 0.2 + 0.15 * neg_hits + 0.05 * issue_probs[i].sum())

        return PredictionBundle(issue_probs, sentiment_probs, actionability)

    def explain(self, cleaned_text: str, top_k: int = 4) -> List[str]:
        tokens = cleaned_text.split()
        hits = [t for t in tokens if t in _POSITIVE_WORDS or t in _NEGATIVE_WORDS]
        return hits[:top_k]


class ModelManager:
    """Trains (or loads a cached copy of) the production model and exposes a
    single, backbone-agnostic prediction interface to the rest of the app.

    Attributes:
        mode: Currently active backbone mode string ("baseline" or "offline").
        model: The underlying `BaselineModel` or `OfflineRuleModel` instance.
        issue_classes: Ordered issue label vocabulary of the active model.
        sentiment_classes: Ordered sentiment label vocabulary of the active model.
    """

    def __init__(self) -> None:
        self.mode: str = BASELINE_MODE
        self.model = None
        self.issue_classes: List[str] = []
        self.sentiment_classes: List[str] = []
        self.training_rows: int = 0

    def build(self) -> "ModelManager":
        """Generates synthetic data, sanitizes it, and trains the production model.

        Falls back to the dependency-free `OfflineRuleModel` if training raises
        for any reason, so the application never crashes at startup.
        """
        try:
            generator = SyntheticReviewDataGenerator(n_samples=CFG.n_samples, seed=CFG.seed)
            df = generator.generate()
            sanitizer = ReviewSanitizer(CFG.min_alpha_ratio, CFG.min_token_count)
            df_clean = sanitizer.sanitize_dataframe(df, text_col="review_body")
            df_clean = df_clean[df_clean["review_body_edge_case"] == "ok"].reset_index(drop=True)

            model = BaselineModel().fit(df_clean)
            self.model = model
            self.mode = BASELINE_MODE
            self.issue_classes = model.issue_classes
            self.sentiment_classes = model.sentiment_classes
            self.training_rows = len(df_clean)
        except Exception:  # noqa: BLE001 - any failure degrades to the offline model
            self.model = OfflineRuleModel()
            self.mode = OFFLINE_MODE
            self.issue_classes = self.model.issue_classes
            self.sentiment_classes = self.model.sentiment_classes
            self.training_rows = 0
        return self

    def predict(self, cleaned_texts: Sequence[str]) -> PredictionBundle:
        """Delegates prediction to the active backbone model."""
        return self.model.predict(cleaned_texts)

    def explain(self, cleaned_text: str, top_k: int = 4) -> List[str]:
        """Delegates explanation extraction to the active backbone model."""
        return self.model.explain(cleaned_text, top_k=top_k)
