"""AI Job Landscape & Fit Analyzer — Streamlit entry point."""

from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="AI Job Landscape & Fit Analyzer",
    page_icon="📊",
    layout="wide",
)

from components import sidebar  # noqa: E402 - must follow set_page_config
from components import (  # noqa: E402
    ai_read_tab,
    demand_trends_tab,
    landscape_tab,
    resume_fit_tab,
    skills_tab,
)
from utils.data_loader import DatasetError, benchmark_subset  # noqa: E402
from utils.matching import build_benchmark  # noqa: E402

# Session state keys, defined once and used consistently across the app.
SESSION_DEFAULTS: dict[str, object] = {
    "resume_profile": None,      # structured profile extracted from the resume
    "fit_result": None,          # the computed score, valid only for scored_target
    "scored_target": None,       # (title, level) the score was computed against
    "chat_history": [],          # Tab 5 conversation turns
    "ai_summary": None,          # cached auto-summary, keyed by filter + score state
    "extraction_error": None,    # last resume-extraction failure message
    "target_changed_message": None,
}


def init_session_state() -> None:
    """Initialize every session state key the app relies on."""
    for key, default in SESSION_DEFAULTS.items():
        st.session_state.setdefault(key, default)


def invalidate_score_on_target_change(title: str, level: str) -> None:
    """Discard a fit score computed against a different benchmark, keeping the resume."""
    scored_target = st.session_state.get("scored_target")
    if st.session_state.get("fit_result") is None or scored_target is None:
        st.session_state["target_changed_message"] = None
        return
    if scored_target != (title, level):
        # Never show a score computed against a different benchmark. The extracted
        # resume profile is kept so the user does not have to re-upload.
        st.session_state["fit_result"] = None
        st.session_state["scored_target"] = None
        st.session_state["ai_summary"] = None
        st.session_state["target_changed_message"] = (
            f"Your target changed to **{title} — {level}**, so the previous fit score "
            f"no longer applies and has been cleared. Your resume is still loaded: "
            f"press **Score my fit** to re-score against the new benchmark."
        )
    else:
        st.session_state["target_changed_message"] = None


def main() -> None:
    """Compose the sidebar and the five tabs."""
    init_session_state()
    st.title("AI Job Landscape & Fit Analyzer")

    try:
        with st.spinner("Loading dataset…"):
            selections = sidebar.render_sidebar()
    except DatasetError as exc:
        st.error(f"The dataset could not be loaded — {exc}")
        st.stop()
        return

    dataset = selections["dataset"]
    df_filtered = selections["df_filtered"]
    category = selections["category"]
    title = selections["target_title"]
    level = selections["target_level"]

    invalidate_score_on_target_change(title, level)

    benchmark = build_benchmark(
        benchmark_subset(dataset.df_2026, title, level), title, level
    )
    target_category_values = dataset.df_2026.loc[
        dataset.df_2026["job_title"] == title, "job_category"
    ]
    target_category = (
        str(target_category_values.mode().iloc[0])
        if not target_category_values.empty
        else "AI"
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Landscape Overview",
            "Skills Landscape",
            "Demand & Trends",
            "Your Resume Fit",
            "AI Market & Fit Read",
        ]
    )

    with tab1:
        landscape_tab.render(df_filtered, category, dataset.rows_2026, dataset.level_order)
    with tab2:
        skills_tab.render(df_filtered, category)
    with tab3:
        demand_trends_tab.render(df_filtered, dataset.df_2026, dataset.df_full, category)
    with tab4:
        resume_fit_tab.render(
            benchmark,
            selections["resume_text"],
            target_category,
            dataset.education_ordinals,
            dataset.level_order,
        )
    with tab5:
        ai_read_tab.render(
            df_filtered,
            category,
            dataset.rows_2026,
            dataset.total_rows,
            dataset.level_order,
            benchmark=benchmark,
            fit_result=st.session_state.get("fit_result"),
        )


if __name__ == "__main__":
    main()
