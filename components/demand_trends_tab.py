"""Tab 3: demand and trends — filtered view, full 2026 landscape, and the 2025-2026 exception."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from components import horizontal_bar, series_color, style_figure
from utils.data_loader import ALL_CATEGORIES, MIN_ROWS_FOR_RELIABLE_STATS, filter_by_category

MAX_TITLES_FULL_LANDSCAPE = 20
MIN_YEAR_MONTHS_FOR_TREND = 3
TREND_CHART_HEIGHT = 460
METRIC_CHART_HEIGHT = 420

METRICS = (
    ("demand_score", "Avg demand score"),
    ("demand_growth_yoy_pct", "Avg demand growth YoY (%)"),
    ("ai_salary_premium_pct", "Avg AI salary premium (%)"),
)


def _metric_charts(df: pd.DataFrame, group_column: str, cap: int | None = None) -> None:
    """Render the three descending metric bar charts grouped by the given column."""
    volume = df.groupby(group_column)["job_id"].count()
    if cap is not None:
        volume = volume.sort_values(ascending=False).head(cap)
        df = df[df[group_column].isin(volume.index)]

    columns = st.columns(3)
    for column, (metric, label) in zip(columns, METRICS):
        grouped = (
            df.groupby(group_column)[metric].mean().sort_values(ascending=False)
        )
        figure = horizontal_bar(
            grouped.index.tolist(),
            grouped.values.tolist(),
            label,
            height=METRIC_CHART_HEIGHT,
        )
        # These are averages, not counts, so relabel with one decimal.
        figure.data[0].text = [f"{v:.1f}" for v in grouped.values][::-1]
        figure.data[0].hovertemplate = "%{y}: %{x:.1f}<extra></extra>"
        figure.update_xaxes(range=[0, float(grouped.max()) * 1.28] if len(grouped) else [0, 1])
        with column:
            st.plotly_chart(figure, width="stretch")


def _section_a(df: pd.DataFrame, category: str) -> None:
    """Render Section A, which respects the job category filter."""
    st.subheader("A. Your Filtered View")
    is_all = not category or category == ALL_CATEGORIES
    group_column = "job_category" if is_all else "job_title"
    st.caption(
        f"Category filter: **{category}** — {len(df):,} 2026 postings, "
        f"aggregated by **{'job category' if is_all else 'job title'}**."
    )
    if df.empty:
        st.info("No postings match the current filter.")
        return
    if len(df) < MIN_ROWS_FOR_RELIABLE_STATS:
        st.warning(
            f"Only {len(df):,} postings match this filter — fewer than "
            f"{MIN_ROWS_FOR_RELIABLE_STATS}. These averages are unreliable."
        )
    _metric_charts(df, group_column)


def _section_b(df_2026: pd.DataFrame) -> None:
    """Render Section B, which deliberately ignores the job category filter."""
    st.subheader("B. Full 2026 Landscape — ignores your category filter on purpose")
    st.caption(
        f"This section deliberately ignores the Job Category filter so you always have "
        f"the whole market for comparison. All {len(df_2026):,} 2026 postings, capped at "
        f"the top {MAX_TITLES_FULL_LANDSCAPE} job titles by posting volume."
    )
    if df_2026.empty:
        st.info("No 2026 postings available.")
        return
    _metric_charts(df_2026, "job_title", cap=MAX_TITLES_FULL_LANDSCAPE)


def _section_c(df_full: pd.DataFrame, category: str) -> None:
    """Render Section C, the only view in the app that uses 2025 data."""
    st.subheader("C. Over Time (2025–2026)")
    st.caption(
        "⚠️ **This is the only chart in the app that uses 2025 data.** Every other "
        "figure is 2026-only. It re-sources from the full dataset to show the "
        "2025→2026 change, and respects your category filter."
    )
    scoped = filter_by_category(df_full, category)
    if scoped.empty:
        st.info("No postings match the current filter.")
        return

    grouped = (
        scoped.groupby(["posting_year", "posting_month"])
        .agg(postings=("job_id", "count"), median_salary=("annual_salary_usd", "median"))
        .reset_index()
        .sort_values(["posting_year", "posting_month"])
    )
    if len(grouped) < MIN_YEAR_MONTHS_FOR_TREND:
        st.info(
            f"Only {len(grouped)} distinct year-month(s) in this selection — fewer than "
            f"{MIN_YEAR_MONTHS_FOR_TREND}. A trend line over so few points would imply a "
            f"pattern the data cannot support, so the chart is hidden."
        )
        return

    labels = [
        f"{int(year)}-{int(month):02d}"
        for year, month in zip(grouped["posting_year"], grouped["posting_month"])
    ]

    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Scatter(
            x=labels,
            y=grouped["postings"],
            name="Postings",
            mode="lines+markers",
            line=dict(color=series_color(0), width=2),
            marker=dict(size=8),
            hovertemplate="%{x}<br>%{y:,} postings<extra></extra>",
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=labels,
            y=grouped["median_salary"],
            name="Median salary",
            mode="lines+markers",
            # Dashed so the two lines stay distinguishable without relying on color.
            line=dict(color=series_color(1), width=2, dash="dash"),
            marker=dict(size=8, symbol="diamond"),
            hovertemplate="%{x}<br>Median salary $%{y:,.0f}<extra></extra>",
        ),
        secondary_y=True,
    )
    figure.update_layout(
        title=dict(text=f"Postings and median salary by month, 2025–2026 (category: {category})",
                   font=dict(size=14)),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified",
    )
    style_figure(figure, height=TREND_CHART_HEIGHT)
    figure.update_layout(showlegend=True)
    figure.update_yaxes(
        title_text="Postings (left axis, solid)", secondary_y=False, showgrid=False
    )
    figure.update_yaxes(
        title_text="Median salary USD (right axis, dashed)", secondary_y=True, showgrid=False
    )
    figure.update_xaxes(showgrid=False)
    st.plotly_chart(figure, width="stretch")
    st.caption(
        f"{len(scoped):,} postings across {len(grouped)} year-months (2025 and 2026 combined). "
        f"Both series are lines on **different axes** — posting count on the left (solid), "
        f"median salary on the right (dashed). Read each against its own labeled scale; "
        f"where the lines cross means nothing."
    )


def _render(df_filtered: pd.DataFrame, df_2026: pd.DataFrame, df_full: pd.DataFrame,
            category: str) -> None:
    """Render the full demand and trends tab."""
    _section_a(df_filtered, category)
    st.divider()
    _section_b(df_2026)
    st.divider()
    _section_c(df_full, category)


def render(df_filtered: pd.DataFrame, df_2026: pd.DataFrame, df_full: pd.DataFrame,
           category: str) -> None:
    """Render the demand and trends tab, isolating any failure to this tab."""
    try:
        _render(df_filtered, df_2026, df_full, category)
    except Exception as exc:  # noqa: BLE001 - keep the other tabs usable
        st.error(f"Demand & trends could not be rendered ({type(exc).__name__}).")
