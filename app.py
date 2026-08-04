"""RET-02 — Product Review Intelligence System.

Streamlit entry point. Run locally with:

    streamlit run app.py

No notebook needs to be open; the model trains itself from the bundled
synthetic-data generator on first load and is cached for the life of the
server process (see `get_engine`).
"""

from __future__ import annotations

import io
from typing import List

import pandas as pd
import plotly.express as px
import streamlit as st

from src.analytics import generate_executive_summary, responses_to_dataframe
from src.config import CFG, DEFAULT_PROFILE, PROFILE_THEMES
from src.data_generator import SyntheticReviewDataGenerator
from src.i18n import LANGUAGES, t
from src.inference_engine import ProductReviewIntelligenceEngine, ReviewAnalysisResponse
from src.model_manager import ModelManager
from src.sanitizer import ReviewSanitizer

st.set_page_config(
    page_title="Product Review Intelligence System",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ----------------------------------------------------------------------------
# Cached resources: model + engine build once per server process, not per rerun.
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_engine() -> ProductReviewIntelligenceEngine:
    model_manager = ModelManager().build()
    sanitizer = ReviewSanitizer(CFG.min_alpha_ratio, CFG.min_token_count)
    return ProductReviewIntelligenceEngine(model_manager, sanitizer, CFG.confidence_threshold)


@st.cache_data(show_spinner=False)
def get_demo_dataset() -> pd.DataFrame:
    return SyntheticReviewDataGenerator(n_samples=150, seed=CFG.seed + 1).generate()


def inject_theme_css(profile_key: str) -> None:
    """Applies the full color scheme, card styles, and visual mood for a profile."""
    p = PROFILE_THEMES[profile_key]
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {p['background']};
            color: {p['text']};
            font-family: {p['font']};
        }}
        section[data-testid="stSidebar"] {{
            background-color: {p['surface']};
            border-right: 1px solid {p['border']};
        }}
        h1, h2, h3, h4, h5, p, span, label, .stMarkdown {{
            color: {p['text']};
        }}
        .stTextInput input, .stTextArea textarea, .stNumberInput input,
        .stSelectbox div[data-baseweb="select"] > div {{
            background-color: {p['surface_alt']} !important;
            color: {p['text']} !important;
            border: 1px solid {p['border']} !important;
            border-radius: 8px !important;
        }}
        div.stButton > button, div.stDownloadButton > button {{
            background: linear-gradient(135deg, {p['accent']} 0%, {p['accent_2']} 100%);
            color: #FFFFFF !important;
            border: none;
            padding: 0.7rem 1.2rem;
            border-radius: 8px;
            font-weight: 600;
            letter-spacing: 0.2px;
            transition: all 0.2s ease-in-out;
            width: 100%;
            box-shadow: 0 4px 10px rgba(0,0,0,0.15);
        }}
        div.stButton > button:hover, div.stDownloadButton > button:hover {{
            filter: brightness(1.08);
            box-shadow: 0 6px 15px rgba(0,0,0,0.25);
        }}
        .ret02-card {{
            background: {p['surface_alt']};
            border: 1px solid {p['border']};
            border-radius: 12px;
            padding: 1.1rem 1.3rem;
            margin-bottom: 1rem;
        }}
        .ret02-warning-box {{
            background-color: {p['surface_alt']};
            color: {p['warning']};
            padding: 0.9rem;
            border-radius: 8px;
            text-align: center;
            font-weight: 600;
            border: 1px dashed {p['warning']};
            margin-top: 0.6rem;
        }}
        .ret02-info-box {{
            background-color: {p['surface_alt']};
            color: {p['text_muted']};
            padding: 0.9rem;
            border-radius: 8px;
            border: 1px dashed {p['border']};
            margin-top: 0.6rem;
        }}
        .ret02-badge-auto {{
            display: inline-block; padding: 0.35rem 0.9rem; border-radius: 999px;
            background: {p['success']}22; color: {p['success']}; font-weight: 700;
            border: 1px solid {p['success']};
        }}
        .ret02-badge-human {{
            display: inline-block; padding: 0.35rem 0.9rem; border-radius: 999px;
            background: {p['warning']}22; color: {p['warning']}; font-weight: 700;
            border: 1px solid {p['warning']};
        }}
        .ret02-chip {{
            display: inline-block; padding: 0.25rem 0.7rem; border-radius: 999px;
            background: {p['accent']}22; color: {p['accent_2']}; font-weight: 600;
            border: 1px solid {p['accent']}; margin: 0.15rem;
            font-size: 0.85rem;
        }}
        div[data-testid="stMetricValue"] {{ color: {p['accent_2']} !important; font-weight: 700 !important; }}
        div[data-testid="stMetric"] {{
            background: {p['surface_alt']}; border: 1px solid {p['border']};
            border-radius: 10px; padding: 0.7rem 0.9rem;
        }}
        .stProgress > div > div > div {{ background: linear-gradient(90deg, {p['accent']}, {p['accent_2']}); }}
        .ret02-footer {{ color: {p['text_muted']}; font-size: 0.8rem; margin-top: 2rem; text-align: center; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_result_card(resp: ReviewAnalysisResponse, lang: str) -> None:
    """Renders one ReviewAnalysisResponse as a styled result card."""
    status_html = (
        f'<span class="ret02-badge-auto">{t("status_auto", lang)}</span>'
        if resp.status == "AUTO_PROCESSED"
        else f'<span class="ret02-badge-human">{t("status_human", lang)}</span>'
    )
    st.markdown(f'<div class="ret02-card">{status_html}</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric(t("sentiment_label", lang), f"{resp.sentiment} ({resp.sentiment_probability:.0%})")
    col2.metric(t("actionability_label", lang), f"{resp.actionability_score:.2f}")
    col3.metric(t("edge_case_label", lang), resp.edge_case_flag)

    st.caption(t("confidence_label", lang))
    st.progress(min(max(resp.overall_confidence, 0.0), 1.0))
    st.caption(f"{resp.overall_confidence:.0%}  ·  {t('backbone_mode_label', lang)}: {resp.model_backbone_mode}")

    st.markdown(f"**{t('issues_label', lang)}**")
    if resp.predicted_issues:
        chips = "".join(
            f'<span class="ret02-chip">{i.label} · {i.probability:.0%}</span>' for i in resp.predicted_issues
        )
        st.markdown(chips, unsafe_allow_html=True)
    else:
        st.markdown(f"_{t('no_issues_detected', lang)}_")

    st.markdown(f"**{t('why_prediction_label', lang)}**")
    if resp.explanation:
        st.write(f"{t('why_prediction_intro', lang)} " + ", ".join(f"`{w}`" for w in resp.explanation))
    else:
        st.caption(t("why_prediction_empty", lang))


def build_summary_charts(summary, lang: str) -> None:
    """Renders the sentiment-distribution and issue-frequency charts."""
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**{t('sentiment_dist_label', lang)}**")
        if summary.sentiment_distribution:
            df_s = pd.DataFrame(
                {"sentiment": list(summary.sentiment_distribution.keys()),
                 "count": list(summary.sentiment_distribution.values())}
            )
            fig = px.pie(df_s, names="sentiment", values="count", hole=0.5)
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=280)
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown(f"**{t('issue_freq_label', lang)}**")
        if summary.issue_frequency:
            df_i = pd.DataFrame(
                {"issue": list(summary.issue_frequency.keys()),
                 "count": list(summary.issue_frequency.values())}
            ).sort_values("count", ascending=True)
            fig2 = px.bar(df_i, x="count", y="issue", orientation="h")
            fig2.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=280)
            st.plotly_chart(fig2, use_container_width=True)


# ----------------------------------------------------------------------------
# Top-of-page language selector (required to be the very first UI element)
# ----------------------------------------------------------------------------
lang_col, _ = st.columns([1, 3])
with lang_col:
    lang_display = st.selectbox(
        "Language / Til / Язык", options=list(LANGUAGES.values()), index=0, label_visibility="collapsed",
    )
lang = next(code for code, disp in LANGUAGES.items() if disp == lang_display)

# ----------------------------------------------------------------------------
# Sidebar: adaptive-design profile selector
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"### {t('profile_label', lang)}")
    profile_options = ["pm_cx", "data_ml", "exec_biz", "genz_student"]
    profile_labels = [t(PROFILE_THEMES[p]["label_key"], lang) for p in profile_options]
    chosen_label = st.radio(t("profile_label", lang), profile_labels, label_visibility="collapsed")
    profile_key = profile_options[profile_labels.index(chosen_label)]
    st.caption(t("sidebar_note", lang))

inject_theme_css(profile_key)

engine = get_engine()

st.title(t("app_title", lang))
st.caption(t("app_subtitle", lang))

tab_single, tab_batch, tab_summary, tab_model = st.tabs([
    t("tab_single", lang), t("tab_batch", lang), t("tab_summary", lang), t("tab_model", lang),
])

# ----------------------------------------------------------------------------
# Tab 1 — Single Review Analysis
# ----------------------------------------------------------------------------
with tab_single:
    st.markdown(f"#### {t('single_intro_title', lang)}")
    st.write(t("single_intro_body", lang))

    with st.form("single_review_form"):
        review_text = st.text_area(
            t("review_text_label", lang), height=140, placeholder=t("review_text_placeholder", lang),
        )
        c1, c2 = st.columns(2)
        rating = c1.selectbox(t("rating_label", lang), [None, 1, 2, 3, 4, 5], format_func=lambda x: "—" if x is None else str(x))
        review_id = c2.text_input(t("review_id_label", lang), value="")
        submitted = st.form_submit_button(t("analyze_btn", lang))

    if submitted:
        if not review_text or not review_text.strip():
            st.markdown(f'<div class="ret02-warning-box">{t("no_review_warning", lang)}</div>', unsafe_allow_html=True)
        else:
            result = engine.analyze(review_text, rating, review_id or None)
            st.markdown(f"#### {t('result_header', lang)}")
            render_result_card(result, lang)

# ----------------------------------------------------------------------------
# Tab 2 — Batch CSV / Excel Analysis
# ----------------------------------------------------------------------------
with tab_batch:
    st.markdown(f"#### {t('batch_intro_title', lang)}")
    st.write(t("batch_intro_body", lang))

    uploaded_file = st.file_uploader(t("upload_label", lang), type=["csv", "xlsx", "xls"])

    if uploaded_file is None:
        st.markdown(f'<div class="ret02-info-box">{t("no_file_info", lang)}</div>', unsafe_allow_html=True)
    else:
        try:
            if uploaded_file.name.lower().endswith(".csv"):
                df_upload = pd.read_csv(uploaded_file)
            else:
                df_upload = pd.read_excel(uploaded_file)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read file: {exc}")
            df_upload = None

        if df_upload is not None:
            if "review_text" not in df_upload.columns:
                st.markdown(f'<div class="ret02-warning-box">{t("missing_column_error", lang)}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f"**{t('file_preview_label', lang)}**")
                st.dataframe(df_upload.head(10), use_container_width=True)

                if st.button(t("process_batch_btn", lang)):
                    records = []
                    for _, row in df_upload.iterrows():
                        records.append({
                            "review_text": row.get("review_text", ""),
                            "rating": row.get("rating") if "rating" in df_upload.columns else None,
                            "review_id": str(row.get("review_id")) if "review_id" in df_upload.columns else None,
                        })
                    with st.spinner("..."):
                        responses: List[ReviewAnalysisResponse] = engine.batch_analyze(records)
                    st.session_state["last_batch_responses"] = responses
                    st.success(t("batch_done_msg", lang))

        if "last_batch_responses" in st.session_state:
            responses = st.session_state["last_batch_responses"]
            results_df = responses_to_dataframe(responses)
            st.markdown(f"**{t('results_table_label', lang)}**")
            st.dataframe(results_df, use_container_width=True)

            csv_bytes = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                t("download_results_btn", lang), data=csv_bytes,
                file_name="ret02_batch_results.csv", mime="text/csv",
            )

            summary = generate_executive_summary(responses)
            build_summary_charts(summary, lang)

# ----------------------------------------------------------------------------
# Tab 3 — Executive Summary
# ----------------------------------------------------------------------------
with tab_summary:
    source = st.radio(
        t("summary_source_label", lang),
        [t("summary_source_batch", lang), t("summary_source_demo", lang)],
        horizontal=True,
    )

    responses_for_summary: List[ReviewAnalysisResponse] = []
    if source == t("summary_source_batch", lang):
        responses_for_summary = st.session_state.get("last_batch_responses", [])
        if not responses_for_summary:
            st.markdown(f'<div class="ret02-info-box">{t("summary_no_data", lang)}</div>', unsafe_allow_html=True)
    else:
        demo_df = get_demo_dataset()
        records = [
            {"review_text": r.review_body, "rating": r.rating, "review_id": r.review_id}
            for r in demo_df.itertuples()
        ]
        responses_for_summary = engine.batch_analyze(records)

    if responses_for_summary:
        summary = generate_executive_summary(responses_for_summary)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric(t("metric_total_reviews", lang), summary.total_reviews)
        m2.metric(t("metric_auto_processed", lang), summary.auto_processed)
        m3.metric(t("metric_human_review", lang), summary.human_review_required)
        m4.metric(t("metric_avg_actionability", lang), f"{summary.avg_actionability:.2f}")
        m5.metric(t("metric_nss", lang), f"{summary.net_sentiment_score:+.1f}")

        st.markdown(f"**{t('top_issues_label', lang)}**")
        if summary.top_issues:
            chips = "".join(f'<span class="ret02-chip">{label} · {count}</span>' for label, count in summary.top_issues)
            st.markdown(chips, unsafe_allow_html=True)
        else:
            st.markdown(f"_{t('no_issues_detected', lang)}_")

        build_summary_charts(summary, lang)

        st.metric(t("metric_urgent", lang), summary.urgent_count)

# ----------------------------------------------------------------------------
# Tab 4 — Model Info
# ----------------------------------------------------------------------------
with tab_model:
    st.markdown(f"#### {t('model_info_title', lang)}")
    mode = engine.model_manager.mode
    st.markdown(f"**{t('model_mode_label', lang)}:** `{mode}`")
    if mode == "baseline":
        st.write(t("model_mode_baseline_desc", lang))
    else:
        st.warning(t("model_mode_offline_desc", lang))

    c1, c2 = st.columns(2)
    c1.metric(t("model_threshold_label", lang), f"{CFG.confidence_threshold:.2f}")
    c2.metric(t("model_training_rows_label", lang), engine.model_manager.training_rows)

    st.markdown(f"**{t('model_issue_classes_label', lang)}:** " + ", ".join(engine.model_manager.issue_classes))
    st.markdown(f"**{t('model_sentiment_classes_label', lang)}:** " + ", ".join(engine.model_manager.sentiment_classes))

st.markdown(f'<div class="ret02-footer">{t("footer_note", lang)}</div>', unsafe_allow_html=True)
