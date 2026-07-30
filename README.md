# RET-02 — Product Review Intelligence System

An end-to-end ML capstone project for RetailTech's marketplace: converts raw product
reviews into a structured signal (issue category, sentiment, actionability) that a
Product/CX team can act on, instead of a single 1–5 star number.

## 1. Business Problem

Star ratings alone don't explain *why* customers are happy or unhappy. This project
builds an ML pipeline that reads a review and returns:

- **Issue category** (multi-label): `Delivery`, `Packaging`, `Product Quality`, `Defect`,
  `Price/Value`, `Customer Support`, `None`
- **Sentiment** (3-class): `Positive`, `Neutral`, `Negative`
- **Actionability score** (0–1): how urgently the review should route to a human

## 2. Repository Contents

| File | Description |
|---|---|
| `RET_02_Product_Review_Intelligence_System.ipynb` | Full pipeline: data generation → sanitization → baseline → multi-task transformer → evaluation → inference engine → deployment stub |
| `RET-02.docx` | Original client/capstone brief this project implements |

## 3. Pipeline Overview

```
Raw Review --> Sanitizer/Guardrails --> [reject if empty/gibberish]
                                   |
                                   v
                         Feature Extraction (DeBERTa-v3 / TF-IDF)
                                   |
                                   v
                         Multi-Task Neural Head
                        /          |            \
                Issue Heads   Sentiment Head   Actionability Head
                        \          |            /
                                   v
                     Uncertainty / Confidence Scoring
                                   |
                    conf < 0.65 ?  yes -> HUMAN_REVIEW_REQUIRED
                                   no  -> AUTO_PROCESSED
                                   |
                                   v
                         Structured JSON Output
```

Every stage validates its input and can short-circuit before the expensive model
stages; every prediction carries a confidence signal that decides whether it's safe
to auto-process or needs a human reviewer.

## 4. Dataset

**⚠️ Important:** this notebook does **not** use a real public review dataset. It uses
a `SyntheticReviewDataGenerator` that fabricates 1,000 reviews from a small hand-written
vocabulary of positive/negative snippets, with issue/sentiment/actionability labels
assigned by rule rather than by a human annotator or real outcome. This was done because
no public dataset ships with ready-made issue-category labels at this granularity, and to
keep the notebook runnable offline.

**Consequence:** all metrics below describe how well the models recover the generator's
own rules, not real-world performance on genuine customer language. Before any real
deployment, the pipeline needs to be re-pointed at a real corpus (e.g. Amazon/Shopify
reviews) using the same column contract (`review_id, product_id, category, rating,
review_title, review_body, issue_category, sentiment, actionability_score`).

## 5. Models

| Model | Purpose |
|---|---|
| TF-IDF (1–2 gram) + `MultiOutputClassifier(LogisticRegression)` | Baseline for issue labels |
| TF-IDF + `LogisticRegression` | Baseline for sentiment |
| Shared-encoder multi-task transformer (`distilbert-base-uncased`, with an offline from-scratch fallback encoder) | Joint issue / sentiment / actionability head |

Split strategy: **Stratified GroupKFold by `product_id`** (5 folds; results below use
fold 0) so a product's reviews never leak across train/validation.

## 6. Results (validation set, synthetic data)

| Task | Baseline (TF-IDF+LR) | Multi-task Transformer |
|---|---|---|
| Issue labels — Macro-F1 | **0.83** | **0.14** |
| Issue labels — Micro-F1 | 0.94 | 0.82 |
| Sentiment — Macro-F1 | 0.53 | 0.61 |
| Sentiment — Accuracy | 0.73 | 0.85 |
| ECE (sentiment calibration) | — | 0.074 |

**⚠️ Known regression:** the transformer collapses to predicting `None` for every real
issue category (0.00 precision/recall on Delivery, Packaging, Product Quality, Defect,
Price/Value, Customer Support), even though per-label ROC-AUC is strong (0.84–0.99). This
is a **thresholding/class-imbalance problem, not a representation problem** — the loss
(`BCEWithLogitsLoss`) has no `pos_weight` for the ~80%-majority `None` class. The current
inference engine deploys this transformer as-is; fixing this (class-weighted or focal
loss, per-class threshold tuning) is the top priority before this model replaces the
baseline in production.

## 7. Requirements

```
python >= 3.9
numpy, pandas, scikit-learn
torch
transformers  (optional — falls back to a from-scratch encoder if unavailable/offline)
matplotlib, seaborn
pydantic
joblib
```

Install:
```bash
pip install numpy pandas scikit-learn torch transformers matplotlib seaborn pydantic joblib
```

## 8. How to Run

1. Open `RET_02_Product_Review_Intelligence_System.ipynb` in Jupyter.
2. Run all cells top to bottom (`SEED = 42`, fully deterministic, CPU-only compatible).
3. Section 7 auto-detects internet/Hugging Face Hub access:
   - **Online:** fine-tunes `distilbert-base-uncased`.
   - **Offline:** falls back to a from-scratch PyTorch Transformer encoder trained on
     the notebook's own vocabulary — same downstream architecture either way.
4. Section 11 emits a FastAPI service stub; Section 12 (last cell) emits a Streamlit
   `app.py` for a demo UI.

## 9. Inference Contract

**Input** (JSON):
```json
{"review_text": "...", "rating": 3, "product_id": "PROD-00001", "category": "Electronics"}
```

**Output** (JSON):
```json
{
  "predicted_issues": [{"label": "Delivery", "probability": 0.81}],
  "sentiment": "Negative",
  "sentiment_probability": 0.77,
  "actionability_score": 0.62,
  "overall_confidence": 0.71,
  "status": "AUTO_PROCESSED",
  "edge_case_flag": "none",
  "model_backbone_mode": "pretrained"
}
```

## 10. Known Limitations

- **Synthetic data only** — no real public dataset used yet; see §4.
- **Issue-head collapse** in the transformer model due to unweighted loss (§6).
- **Small validation set** — only 200 rows, 6–12 examples per minority issue class;
  metrics are noisy and only one of five GroupKFold folds is reported.
- **English-only** — sanitizer and offline tokenizer are not multilingual.
- **Actionability is a proxy label**, not a verified business outcome (return, refund,
  escalation); needs validation against real CX data before driving automated routing.
- **Cross-category domain shift** — a single shared model can under-perform on
  categories with distinctive vocabulary (Apparel vs. Electronics).

## 11. Roadmap

1. Replace synthetic corpus with a real, licensed review dataset (same column contract).
2. Fix issue-head class imbalance (class-weighted/focal loss + per-class threshold search)
   and re-evaluate before it replaces the baseline.
3. Report metrics averaged across all 5 GroupKFold folds (mean ± std).
4. Add drift monitoring, scheduled/triggered retraining, and experiment tracking
   (MLflow/W&B) once running against real data.
5. Containerize the FastAPI service for reproducible, student-scale deployment.

## 12. Scope

**In scope:** prototype ML solution, public/legally usable data, reproducible inference
workflow, student-scale deployment.
**Out of scope:** direct production integration, unauthorized private data,
enterprise-scale infrastructure, claims beyond what the model can validly support.
