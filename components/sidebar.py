"""Sidebar with two independent sections: landscape filters and resume-fit targets."""

from __future__ import annotations

from typing import Any

import streamlit as st

from utils.data_loader import (
    ALL_CATEGORIES,
    Dataset,
    DatasetError,
    category_options,
    filter_by_category,
    largest_benchmark_target,
    load_bundled_dataset,
    load_uploaded_dataset,
    target_counts,
    title_options,
)
from utils.resume_parser import ResumeParseError, extract_pdf_text, looks_unextractable


def _resolve_dataset() -> tuple[Dataset, str | None]:
    """Return the active dataset and an error message if an upload was rejected."""
    upload = st.file_uploader(
        "Upload a CSV (same schema)", type=["csv"], key="csv_upload"
    )
    if upload is None:
        return load_bundled_dataset(), None
    try:
        return load_uploaded_dataset(upload.getvalue(), upload.name), None
    except DatasetError as exc:
        # Validation failed: fall back to the bundled data rather than breaking the app.
        return load_bundled_dataset(), str(exc)


def _render_landscape_section() -> dict[str, Any]:
    """Render Section 1, which controls Tabs 1-3 only."""
    st.header("Explore the Landscape")
    st.caption("Controls Tabs 1–3.")

    dataset, upload_error = _resolve_dataset()
    if upload_error:
        st.error(f"Upload rejected — {upload_error}")
        st.info("Keeping the bundled dataset as the active data source.")
    st.success(f"Active data: {dataset.source_label}")

    st.caption(
        f"**2026 postings: {dataset.rows_2026:,} of {dataset.total_rows:,} total rows.**"
    )

    category = st.selectbox(
        "Job Category",
        options=category_options(dataset.df_2026),
        index=0,
        key="category_filter",
        help="The only landscape filter. It never affects the Resume Fit tab.",
    )
    filtered = filter_by_category(dataset.df_2026, category)
    st.caption(f"Showing {len(filtered):,} of {dataset.rows_2026:,} 2026 postings")

    return {"dataset": dataset, "category": category, "df_filtered": filtered}


def _render_resume_section(dataset: Dataset) -> dict[str, Any]:
    """Render Section 2, which controls Tab 4 only."""
    st.header("Resume Fit")
    st.caption(
        "Controls Tab 4 only. These targets define your benchmark and are "
        "**independent of the Job Category filter above** — changing the landscape "
        "filter never changes your benchmark."
    )

    pdf = st.file_uploader("Resume PDF", type=["pdf"], key="resume_pdf")
    pasted = st.text_area(
        "…or paste resume text", height=140, key="resume_paste",
        placeholder="Paste your resume here if you don't have a PDF, or if your PDF is a scan.",
    )

    resume_text, resume_note = "", None
    if pdf is not None:
        try:
            extracted = extract_pdf_text(pdf)
            if looks_unextractable(extracted):
                resume_note = (
                    "Almost no text could be extracted from that PDF — it is likely a "
                    "scan or image. Please paste your resume text in the box above."
                )
            else:
                resume_text = extracted
        except ResumeParseError as exc:
            resume_note = str(exc)
    if not resume_text and pasted.strip():
        resume_text = pasted.strip()

    if resume_note:
        st.warning(resume_note)

    counts = target_counts(dataset.df_2026)
    titles = title_options(dataset.df_2026)
    default_title, default_level = largest_benchmark_target(
        dataset.df_2026, dataset.level_order
    )

    title = st.selectbox(
        "Target Job Title",
        options=titles,
        index=titles.index(default_title) if default_title in titles else 0,
        key="target_title",
        format_func=lambda t: f"{t} ({counts_for_title(counts, t):,} postings)",
    )
    levels = dataset.level_order
    level = st.selectbox(
        "Target Experience Level",
        options=levels,
        index=levels.index(default_level) if default_level in levels else 0,
        key="target_level",
        format_func=lambda lv: f"{lv} — {counts.get((title, lv), 0):,} postings",
    )

    n_benchmark = counts.get((title, level), 0)
    st.caption(f"Benchmarking against {n_benchmark:,} postings")

    return {
        "resume_text": resume_text,
        "target_title": title,
        "target_level": level,
        "benchmark_count": n_benchmark,
    }


def counts_for_title(counts: dict[tuple[str, str], int], title: str) -> int:
    """Sum the 2026 postings across all experience levels for one job title."""
    return sum(n for (t, _), n in counts.items() if t == title)


def render_sidebar() -> dict[str, Any]:
    """Render both sidebar sections and return the combined selections."""
    with st.sidebar:
        st.title("AI Job Landscape & Fit Analyzer")
        landscape = _render_landscape_section()
        st.divider()
        resume = _render_resume_section(landscape["dataset"])
    return {**landscape, **resume}
