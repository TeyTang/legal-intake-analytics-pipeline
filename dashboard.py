from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from project1 import OUTPUT_DIR
from semantic_summary_search import DEFAULT_DEMO_QUERIES, build_search_artifacts, load_search_data, search_summaries


st.set_page_config(
    page_title="Legal Intake Analytics Dashboard",
    layout="wide",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top right, rgba(204, 177, 120, 0.22), transparent 28%),
                linear-gradient(180deg, #f7f2e8 0%, #fffdf8 100%);
            color: #1f1b18;
        }
        .block-container {
            max-width: 1200px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        h1, h2, h3 {
            font-family: Georgia, "Times New Roman", serif;
            letter-spacing: 0.01em;
        }
        .hero {
            background: linear-gradient(135deg, #203947 0%, #54736d 55%, #cfb27a 100%);
            border-radius: 22px;
            padding: 1.4rem 1.6rem;
            color: #fefbf5;
            box-shadow: 0 18px 40px rgba(32, 57, 71, 0.12);
            margin-bottom: 1rem;
        }
        .hero p {
            margin: 0.35rem 0 0 0;
            font-size: 1rem;
        }
        .badge {
            display: inline-block;
            margin-right: 0.5rem;
            margin-top: 0.35rem;
            padding: 0.2rem 0.55rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.16);
            border: 1px solid rgba(255,255,255,0.24);
            font-size: 0.82rem;
        }
        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.78);
            border: 1px solid #dfd3c0;
            border-radius: 16px;
            padding: 0.65rem;
            box-shadow: 0 8px 24px rgba(31, 27, 24, 0.05);
        }
        div[data-testid="stDataFrame"] {
            background: rgba(255, 255, 255, 0.7);
            border-radius: 14px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def read_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def read_csv(path):
    return pd.read_csv(path)


@st.cache_resource
def load_search_artifacts():
    search_df = load_search_data()
    return build_search_artifacts(search_df)


def load_dashboard_data() -> dict[str, object]:
    return {
        "executive_metrics": read_json(OUTPUT_DIR / "executive_metrics.json"),
        "delay_metrics": read_json(OUTPUT_DIR / "assignment_delay_model_metrics.json"),
        "text_metrics": read_json(OUTPUT_DIR / "practice_area_text_metrics.json"),
        "entity_metrics": read_json(OUTPUT_DIR / "legal_entity_metrics.json"),
        "cleaned_df": read_csv(OUTPUT_DIR / "cleaned_legal_intake.csv"),
        "practice_area_summary": read_csv(OUTPUT_DIR / "practice_area_summary.csv"),
        "status_summary": read_csv(OUTPUT_DIR / "status_summary.csv"),
        "delay_predictions": read_csv(OUTPUT_DIR / "assignment_delay_predictions.csv"),
        "text_predictions": read_csv(OUTPUT_DIR / "practice_area_text_predictions.csv"),
        "top_terms": read_csv(OUTPUT_DIR / "practice_area_top_terms.csv"),
        "entity_record_view": read_csv(OUTPUT_DIR / "legal_entity_record_view.csv"),
        "entity_summary": read_csv(OUTPUT_DIR / "legal_entity_summary.csv"),
        "search_demo": read_csv(OUTPUT_DIR / "summary_search_demo_results.csv"),
    }


def render_header() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>Legal Intake Analytics Dashboard</h1>
            <p>Portfolio view of analytics, machine learning, NLP routing, entity extraction, and summary search for a law-firm intake workflow.</p>
            <div>
                <span class="badge">Analytics Pipeline</span>
                <span class="badge">ML Baseline</span>
                <span class="badge">NLP Routing</span>
                <span class="badge">Entity Extraction</span>
                <span class="badge">Summary Search</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_overview(data: dict[str, object]) -> None:
    executive_metrics = data["executive_metrics"]
    practice_area_summary = data["practice_area_summary"]
    cleaned_df = data["cleaned_df"]
    status_summary = data["status_summary"]

    metric_columns = st.columns(4)
    metric_columns[0].metric("Total Matters", executive_metrics["total_matters"])
    metric_columns[1].metric("Priority Matters", executive_metrics["priority_matters"])
    metric_columns[2].metric("Open Matters", executive_metrics["open_matters"])
    metric_columns[3].metric("Avg Risk Score", executive_metrics["average_risk_score"])

    metric_columns = st.columns(4)
    metric_columns[0].metric("Avg Days to Assign", executive_metrics["average_days_to_assign"])
    metric_columns[1].metric("High-Value Matters", executive_metrics["high_value_matters"])
    metric_columns[2].metric("Delayed Assignments", executive_metrics["delayed_assignment_matters"])
    metric_columns[3].metric("Top Practice Area", executive_metrics["top_practice_area"])

    chart_columns = st.columns([1.2, 1])
    with chart_columns[0]:
        st.subheader("Practice Area Volume")
        practice_area_chart = practice_area_summary.set_index("practice_area")["matter_count"]
        st.bar_chart(practice_area_chart, use_container_width=True)
    with chart_columns[1]:
        st.subheader("Priority vs Status")
        status_chart = status_summary.copy()
        status_chart["label"] = status_chart["status"] + " | " + status_chart["priority_flag"]
        st.bar_chart(status_chart.set_index("label")["matter_count"], use_container_width=True)

    st.subheader("Matter Intake Table")
    st.dataframe(
        cleaned_df[
            [
                "matter_id",
                "practice_area",
                "status",
                "priority_flag",
                "assignment_delay_flag",
                "summary_topic",
                "estimated_value_usd",
                "risk_score",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )


def render_model_tab(data: dict[str, object]) -> None:
    delay_metrics = data["delay_metrics"]
    text_metrics = data["text_metrics"]
    delay_predictions = data["delay_predictions"]
    text_predictions = data["text_predictions"]
    top_terms = data["top_terms"]

    st.subheader("Delayed Assignment Model")
    cols = st.columns(4)
    cols[0].metric("Accuracy", delay_metrics["accuracy"])
    cols[1].metric("Precision", delay_metrics["precision_delayed"])
    cols[2].metric("Recall", delay_metrics["recall_delayed"])
    cols[3].metric("F1", delay_metrics["f1_delayed"])

    st.dataframe(delay_predictions, use_container_width=True, hide_index=True)

    st.subheader("Practice-Area NLP Classifier")
    cols = st.columns(4)
    cols[0].metric("Accuracy", text_metrics["accuracy"])
    cols[1].metric("Macro Precision", text_metrics["macro_precision"])
    cols[2].metric("Macro Recall", text_metrics["macro_recall"])
    cols[3].metric("Macro F1", text_metrics["macro_f1"])

    model_cols = st.columns([1.3, 1])
    with model_cols[0]:
        incorrect_rows = text_predictions[text_predictions["predicted_correctly"] == False]  # noqa: E712
        st.caption("Incorrect practice-area predictions")
        st.dataframe(incorrect_rows, use_container_width=True, hide_index=True)
    with model_cols[1]:
        selected_area = st.selectbox(
            "Top terms by practice area",
            options=sorted(top_terms["practice_area"].unique().tolist()),
        )
        area_terms = top_terms[top_terms["practice_area"] == selected_area]
        st.dataframe(area_terms, use_container_width=True, hide_index=True)


def render_entity_tab(data: dict[str, object]) -> None:
    entity_metrics = data["entity_metrics"]
    entity_summary = data["entity_summary"]
    entity_record_view = data["entity_record_view"]

    st.subheader("Entity Extraction Snapshot")
    cols = st.columns(4)
    cols[0].metric("Matters With Entities", entity_metrics["matters_with_entities"])
    cols[1].metric("Total Entity Mentions", entity_metrics["total_entity_mentions"])
    cols[2].metric("Unique Entity Values", entity_metrics["unique_entity_values"])
    cols[3].metric("Avg Entities / Matter", entity_metrics["average_entities_per_matter"])

    entity_cols = st.columns([1, 1.2])
    with entity_cols[0]:
        st.subheader("Entity Counts")
        entity_chart = entity_summary.groupby("entity_type")["mention_count"].sum()
        st.bar_chart(entity_chart, use_container_width=True)
    with entity_cols[1]:
        st.subheader("Entity Record View")
        st.dataframe(entity_record_view, use_container_width=True, hide_index=True)


def render_search_tab(data: dict[str, object]) -> None:
    search_artifacts = load_search_artifacts()
    search_demo = data["search_demo"]

    st.subheader("Summary Search")
    st.caption("This search uses latent semantic analysis over TF-IDF features as a lightweight retrieval baseline.")

    demo_query = st.selectbox("Example query", options=DEFAULT_DEMO_QUERIES)
    query = st.text_input("Search intake summaries", value=demo_query)
    top_k = st.slider("Results to show", min_value=1, max_value=5, value=3)

    if query:
        search_results = search_summaries(query, search_artifacts, top_k=top_k)
        st.dataframe(search_results, use_container_width=True, hide_index=True)

    st.subheader("Saved Demo Results")
    st.dataframe(search_demo, use_container_width=True, hide_index=True)


def main() -> None:
    inject_styles()
    render_header()
    data = load_dashboard_data()

    tabs = st.tabs(["Overview", "Models", "Entities", "Search"])
    with tabs[0]:
        render_overview(data)
    with tabs[1]:
        render_model_tab(data)
    with tabs[2]:
        render_entity_tab(data)
    with tabs[3]:
        render_search_tab(data)


if __name__ == "__main__":
    main()
