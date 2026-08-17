"""Tab 1: landscape overview — metric cards, salary tables, composition bars, geography."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components import (
    format_count,
    format_currency,
    format_percent,
    format_salary_range,
    horizontal_bar,
)
from utils.data_loader import MIN_ROWS_FOR_RELIABLE_STATS

TOP_TITLES_SHOWN = 10
TOP_INDUSTRIES_SHOWN = 10
TOP_GEOGRAPHY_SHOWN = 10
COMPOSITION_CHART_HEIGHT = 340


def _metric_cards(df: pd.DataFrame) -> None:
    """Render the five headline metric cards for the filtered postings."""
    columns = st.columns(5)
    columns[0].metric("Total postings", format_count(len(df)))
    columns[1].metric("Median salary", format_currency(df["annual_salary_usd"].median()))
    columns[2].metric("Avg demand score", f"{df['demand_score'].mean():.1f}")
    columns[3].metric("% remote-friendly", format_percent(df["is_remote_friendly"].mean() * 100))
    columns[4].metric("% LLM-related", format_percent(df["is_llm_role"].mean() * 100))


def _salary_by_level(df: pd.DataFrame, level_order: list[str]) -> None:
    """Render the salary-by-experience-level table in Entry to Lead order."""
    st.subheader("Salary by experience level")
    rows = []
    for level in level_order:
        subset = df[df["experience_level"] == level]
        if subset.empty:
            continue
        salary = subset["annual_salary_usd"]
        rows.append(
            {
                "Experience Level": level,
                "Postings": len(subset),
                "Median Salary": format_currency(salary.median()),
                "Typical Range": format_salary_range(
                    salary.quantile(0.25), salary.quantile(0.75)
                ),
            }
        )
    if not rows:
        st.info("No postings match the current filter.")
        return
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(
        f"Typical Range is the 25th–75th percentile of annual_salary_usd. "
        f"Based on {len(df):,} postings."
    )


def _top_titles(df: pd.DataFrame) -> None:
    """Render the ranked top job titles table with both median and average salary."""
    st.subheader(f"Top {TOP_TITLES_SHOWN} job titles by posting volume")
    grouped = (
        df.groupby("job_title")
        .agg(
            postings=("job_id", "count"),
            median_salary=("annual_salary_usd", "median"),
            avg_salary=("annual_salary_usd", "mean"),
            avg_demand=("demand_score", "mean"),
            pct_llm=("is_llm_role", "mean"),
        )
        .sort_values("postings", ascending=False)
        .head(TOP_TITLES_SHOWN)
        .reset_index()
    )
    if grouped.empty:
        st.info("No postings match the current filter.")
        return

    table = pd.DataFrame(
        {
            "Job Title": grouped["job_title"],
            "Postings": grouped["postings"],
            # Same statistic and source as the median metric card, so the two reconcile.
            "Median Salary": grouped["median_salary"].map(format_currency),
            "Avg Salary": grouped["avg_salary"].map(format_currency),
            "Avg Demand Score": grouped["avg_demand"].map(lambda v: f"{v:.1f}"),
            "% LLM-related": grouped["pct_llm"].map(lambda v: format_percent(v * 100)),
        }
    )
    st.dataframe(table, hide_index=True, width="stretch")
    st.caption(
        "Median and average are both shown so outlier-skewed titles are visible. "
        "The Median Salary column uses the same statistic and source (annual_salary_usd) "
        "as the median metric card above."
    )


def _composition_charts(df: pd.DataFrame, level_order: list[str]) -> None:
    """Render the three composition charts as horizontal bars, side by side."""
    st.subheader("Composition")
    left, middle, right = st.columns(3)

    level_counts = df["experience_level"].value_counts()
    ordered_levels = [lv for lv in level_order if lv in level_counts.index]
    level_counts = level_counts.reindex(ordered_levels).sort_values(ascending=False)
    with left:
        st.plotly_chart(
            horizontal_bar(
                level_counts.index.tolist(),
                level_counts.values.tolist(),
                "Experience level mix",
                height=COMPOSITION_CHART_HEIGHT,
            ),
            width="stretch",
        )

    size_counts = df["company_size"].value_counts().sort_values(ascending=False)
    with middle:
        st.plotly_chart(
            horizontal_bar(
                size_counts.index.tolist(),
                size_counts.values.tolist(),
                "Company size mix",
                height=COMPOSITION_CHART_HEIGHT,
            ),
            width="stretch",
        )

    industry_counts = df["industry"].value_counts().head(TOP_INDUSTRIES_SHOWN)
    with right:
        st.plotly_chart(
            horizontal_bar(
                industry_counts.index.tolist(),
                industry_counts.values.tolist(),
                f"Industry (top {TOP_INDUSTRIES_SHOWN})",
                height=COMPOSITION_CHART_HEIGHT,
            ),
            width="stretch",
        )

    st.caption(f"All three charts are counts across the same {len(df):,} filtered postings.")


def _geography(df: pd.DataFrame) -> None:
    """Render top countries and top cities as two side-by-side tables."""
    st.subheader("Geography")
    left, right = st.columns(2)

    countries = df["country"].value_counts().head(TOP_GEOGRAPHY_SHOWN).reset_index()
    countries.columns = ["Country", "Postings"]
    with left:
        st.markdown(f"**Top {TOP_GEOGRAPHY_SHOWN} countries**")
        st.dataframe(countries, hide_index=True, width="stretch")

    cities = df["city"].value_counts().head(TOP_GEOGRAPHY_SHOWN).reset_index()
    cities.columns = ["City", "Postings"]
    with right:
        st.markdown(f"**Top {TOP_GEOGRAPHY_SHOWN} cities**")
        st.dataframe(cities, hide_index=True, width="stretch")

    st.caption(f"Both tables count the same {len(df):,} filtered postings.")


def _render(df: pd.DataFrame, category: str, rows_2026: int, level_order: list[str]) -> None:
    """Render the full landscape tab."""
    st.caption(
        f"Scope: 2026 postings only. Category filter: **{category}** — "
        f"{len(df):,} of {rows_2026:,} 2026 postings."
    )
    if df.empty:
        st.info("No postings match the current filter. Try a different job category.")
        return
    if len(df) < MIN_ROWS_FOR_RELIABLE_STATS:
        st.warning(
            f"Only {len(df):,} postings match this filter — fewer than "
            f"{MIN_ROWS_FOR_RELIABLE_STATS}. Treat these figures as indicative only."
        )

    _metric_cards(df)
    st.divider()
    _salary_by_level(df, level_order)
    st.divider()
    _top_titles(df)
    st.divider()
    _composition_charts(df, level_order)
    st.divider()
    _geography(df)


def render(df: pd.DataFrame, category: str, rows_2026: int, level_order: list[str]) -> None:
    """Render the landscape tab, isolating any failure to this tab."""
    try:
        _render(df, category, rows_2026, level_order)
    except Exception as exc:  # noqa: BLE001 - keep the other tabs usable
        st.error(f"The landscape overview could not be rendered ({type(exc).__name__}).")
