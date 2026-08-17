"""Tab 2: skills landscape — exactly two charts, frequency bars and a salary/growth scatter."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components import format_currency, horizontal_bar, series_color, style_figure
from utils.skills_analysis import (
    TOP_SKILLS_FREQUENCY_CHART,
    aggregate_skills,
    split_by_threshold,
    unique_skill_count,
)

LABELLED_POINTS = 8
SCATTER_HEIGHT = 560
FREQUENCY_CHART_HEIGHT = 620


def _frequency_chart(skills: pd.DataFrame, n_rows: int) -> None:
    """Render chart 1: the top skills by posting count."""
    st.subheader(f"Skill frequency — top {TOP_SKILLS_FREQUENCY_CHART}")
    top = skills.head(TOP_SKILLS_FREQUENCY_CHART)
    figure = horizontal_bar(
        top["skill"].tolist(),
        top["postings"].tolist(),
        "Postings requiring each skill",
        height=FREQUENCY_CHART_HEIGHT,
    )
    # Restate count and share together so the bar length has a denominator.
    figure.data[0].text = [
        f"{int(c):,} ({c / n_rows:.1%})" for c in top["postings"].tolist()
    ][::-1]
    figure.data[0].hovertemplate = "%{y}<extra></extra>"
    st.plotly_chart(figure, width="stretch")
    st.caption(
        f"Counts and percentages are of the {n_rows:,} filtered 2026 postings. "
        f"This chart includes skills below the scatter threshold."
    )


def _scatter_chart(kept: pd.DataFrame, excluded: pd.DataFrame, threshold: int, n_rows: int) -> None:
    """Render chart 2: the skill by salary and growth scatter with quadrant lines."""
    st.subheader("Skill × salary and growth")

    if kept.empty:
        st.info(
            f"No skill appears in at least {threshold} of the {n_rows:,} filtered "
            f"postings, so the scatter has nothing to plot. Widen the job category filter."
        )
        return

    x = kept["median_salary"]
    y = kept["mean_growth"]
    x_median, y_median = float(x.median()), float(y.median())

    # Label only the highest-value points; a label on every dot is unreadable.
    labelled = set(
        kept.assign(score=(x > x_median).astype(int) + (y > y_median).astype(int))
        .sort_values(["score", "postings"], ascending=False)
        .head(LABELLED_POINTS)["skill"]
    )

    figure = go.Figure(
        go.Scatter(
            x=x,
            y=y,
            mode="markers+text",
            text=[s if s in labelled else "" for s in kept["skill"]],
            textposition="top center",
            textfont=dict(size=11),
            marker=dict(
                size=kept["postings"],
                sizemode="area",
                sizeref=2.0 * kept["postings"].max() / (46.0**2),
                sizemin=8,
                color=series_color(0),
                opacity=0.75,
                line=dict(width=2, color="rgba(255,255,255,0.55)"),
            ),
            customdata=kept[["skill", "postings"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>Median salary: $%{x:,.0f}"
                "<br>Avg role growth: %{y:.1f}%<br>%{customdata[1]:,} postings<extra></extra>"
            ),
        )
    )
    figure.add_vline(x=x_median, line_width=1, line_dash="dot", opacity=0.45)
    figure.add_hline(y=y_median, line_width=1, line_dash="dot", opacity=0.45)
    figure.update_layout(
        title=dict(text="Dot size = number of postings requiring the skill", font=dict(size=13)),
        xaxis_title="Median annual salary of postings requiring the skill (USD)",
        yaxis_title="Mean demand growth YoY of those roles (%)",
    )
    style_figure(figure, height=SCATTER_HEIGHT)
    figure.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    st.plotly_chart(figure, width="stretch")

    st.caption(
        f"Quadrant lines sit at the median of each axis "
        f"(salary {format_currency(x_median)}, growth {y_median:.1f}%). "
        f"**Top-right = high pay and high growth.**"
    )
    st.caption(
        f"Showing skills appearing in {threshold}+ postings. "
        f"{len(excluded)} rarer skill(s) excluded from the scatter"
        + (f": {', '.join(excluded['skill'].head(12))}." if len(excluded) else ".")
        + f" A skill in only 2–3 postings can top a median-salary chart purely by chance, "
        f"so rare skills are held out rather than presented as insight."
    )
    st.warning(
        "**Growth is a proxy, not a measurement.** `demand_growth_yoy_pct` is a "
        "per-posting attribute tied to the **role**, not to the skill. The Y-axis "
        "shows the average growth of roles that require a skill — it does not measure "
        "growth in demand for the skill itself."
    )


def _render(df: pd.DataFrame, category: str) -> None:
    """Render the full skills tab."""
    n_rows = len(df)
    st.caption(f"Scope: 2026 postings only. Category filter: **{category}**.")
    if df.empty:
        st.info("No postings match the current filter. Try a different job category.")
        return

    skills = aggregate_skills(df)
    st.caption(
        f"Found **{unique_skill_count(df)} unique skills** across **{n_rows:,} filtered "
        f"2026 postings**. Skills are split on '|', stripped, and deduplicated within "
        f"each posting before counting."
    )

    kept, excluded, threshold = split_by_threshold(skills, n_rows)
    _frequency_chart(skills, n_rows)
    st.divider()
    _scatter_chart(kept, excluded, threshold, n_rows)


def render(df: pd.DataFrame, category: str) -> None:
    """Render the skills tab, isolating any failure to this tab."""
    try:
        _render(df, category)
    except Exception as exc:  # noqa: BLE001 - keep the other tabs usable
        st.error(f"The skills landscape could not be rendered ({type(exc).__name__}).")
