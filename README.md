# RET-02 — Product Review Intelligence System

An AI-powered review triage tool for product, CX, and e-commerce teams. It
cleans raw review text, predicts sentiment and issue categories, scores how
actionable each review is, and routes low-confidence or low-quality reviews
to a human reviewer instead of guessing. The interface is fully trilingual
(English / Oʻzbekcha / Русский) and adapts its visual theme to four user
profiles.

This is a standalone Streamlit application. It does not depend on any
notebook being open — the model trains itself from a bundled synthetic-data
generator the first time the app starts, and is cached for the life of the
server process.

---

## 1. How to run locally

```bash
# from the project root
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`. First load trains the baseline
model on ~900 synthetic reviews, which takes well under a second — there is
no download and no GPU requirement.

To run the verification notebook:

```bash
jupyter notebook RET_02_verification.ipynb
```

## 2. How to deploy to Streamlit Community Cloud

1. Push this folder to a GitHub repository (keep the folder structure as-is;
   `app.py` must sit at the repo root, or set the "Main file path" below
   accordingly).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in, and
   click **New app**.
3. Select the repository and branch, set **Main file path** to `app.py`.
4. Deploy. Streamlit Cloud installs `requirements.txt` automatically.
5. No secrets, API keys, or external services are required — the app is
   fully self-contained.
6. The `.streamlit/config.toml` file sets a sensible default dark theme and
   a 25 MB upload limit for batch files; adjust as needed.

## 3. Architecture

```
User (browser)
      │
      ▼
  app.py  ─── language selector (top) + profile/theme selector (sidebar)
      │
      ▼
ProductReviewIntelligenceEngine  (src/inference_engine.py)
      │
      ├── ReviewSanitizer          (src/sanitizer.py)
      │     clean_text() → normalized string
      │     classify_edge_case() → empty / ultra_short / gibberish / ok
      │
      ├── ModelManager             (src/model_manager.py)
      │     BaselineModel: TF-IDF → {issue MultiOutputClassifier(LogReg),
      │                              sentiment LogReg, actionability Ridge}
      │     OfflineRuleModel: dependency-free keyword fallback, used only
      │                       if BaselineModel training itself fails
      │
      └── confidence routing
            overall_confidence = 0.5 * sentiment_confidence
                               + 0.5 * issue_decisiveness
            status = HUMAN_REVIEW_REQUIRED if confidence < 0.65
                     or edge_case == "ultra_short" else AUTO_PROCESSED
      │
      ▼
src/analytics.py  → executive summary (NSS, top issues, routing load)
```

**Why a TF-IDF + linear-model baseline instead of a fine-tuned transformer
in production?** The project brief explicitly allows "a multi-task
transformer, or a clean offline fallback," and asks the deployed app to be
optimized for Streamlit Community Cloud's free tier. Downloading and running
a transformer checkpoint on every cold start is slow, memory-heavy, and
unnecessary for this workload: the three linear heads (`MultiOutputClassifier`
of `LogisticRegression` for issues, `LogisticRegression` for sentiment,
`Ridge` for actionability) share one TF-IDF feature space — the same
"shared encoder, multiple task heads" idea as a multi-task transformer, just
CPU-only and trainable in well under a second. `OfflineRuleModel` is a second,
fully dependency-free layer under that: if scikit-learn training itself ever
fails, the app still starts and serves keyword-based predictions rather than
crashing. `src/model_manager.py` reports whichever mode is actually active
(`baseline` or `offline`) — see the **Model Info** tab in the app — and is
structured so a fine-tuned transformer backbone could be dropped in as a
third mode without touching the sanitizer, routing logic, or UI.

**Confidence routing.** Every prediction carries an `overall_confidence` in
`[0, 1]`. Reviews below the `confidence_threshold` (0.65 by default,
`src/config.py`), or flagged `ultra_short` by the sanitizer, are routed to
`HUMAN_REVIEW_REQUIRED` instead of being auto-processed. `empty` and
`gibberish` reviews are short-circuited before they ever reach the model.

**Synthetic data.** No public dataset ships with a ready-made
`issue_category x sentiment x actionability` label set at this granularity,
so `src/data_generator.py` builds an internally-consistent synthetic corpus
(rating-conditioned issues, correlated sentiment with label noise, realistic
text noise) used to train the baseline model and to power the demo dataset
in the Executive Summary tab.

## 4. Folder structure

```
RET_02_streamlit_app/
├── app.py                     # Streamlit entry point
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml            # theme + server defaults
├── src/
│   ├── __init__.py
│   ├── config.py               # Config dataclass, profile color themes
│   ├── data_generator.py       # SyntheticReviewDataGenerator
│   ├── sanitizer.py            # ReviewSanitizer (edge-case detection)
│   ├── model_manager.py        # BaselineModel, OfflineRuleModel, ModelManager
│   ├── inference_engine.py     # ProductReviewIntelligenceEngine, confidence routing
│   ├── i18n.py                 # EN / UZ / RU translation catalog
│   └── analytics.py            # executive-summary aggregation
├── models/                     # (optional) cached joblib artifacts
└── RET_02_verification.ipynb   # imports src/ and runs sanity checks
```

## 5. Features

- **Trilingual UI** (English / Oʻzbekcha / Русский) — language selector at
  the top of the page, every string localized via `src/i18n.py`. Review text
  itself is never translated; the model is English-centric by design.
- **Adaptive design** — a profile selector in the sidebar (Product
  Manager/CX Lead, Data Analyst/ML Engineer, Executive/Business, Gen-Z/
  Student) swaps the entire color scheme and card styling via CSS
  variables in `src/config.py`.
- **Confidence-based routing visualization** — progress bar + Auto/Human
  Review badge on every single-review result.
- **Batch CSV / Excel upload** with a downloadable results CSV.
- **Executive summary** — Net Sentiment Score, top recurring issues, average
  actionability, urgent-item count, sourced from either your last batch
  upload or a built-in demo dataset.
- **"Why this prediction?"** — top TF-IDF-weighted keywords that drove the
  sentiment call for a single review.
- **Model Info tab** — shows the currently active backbone mode, confidence
  threshold, and label vocabularies.
- **Simple analytics** — sentiment-distribution and issue-frequency charts
  for any batch or the demo dataset.

## 6. Known limitations

- The sanitizer's gibberish heuristics and the baseline model's vocabulary
  are English-only; non-English review text will likely be misclassified as
  gibberish or score poorly. Only the interface is trilingual.
- The baseline model is trained on synthetic data and is a demonstration of
  the pipeline's mechanics, not a benchmark-grade classifier — treat its
  predictions as indicative, especially for the exact numeric confidence
  values.
- `OfflineRuleModel` intentionally trades accuracy for zero dependencies; it
  only activates if the primary training path fails.
