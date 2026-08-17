"""Shared presentation helpers: formatting and Plotly chart styling."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
import streamlit as st

# Categorical slots 1 and 2 from the validated reference palette, stepped per mode.
SERIES_LIGHT = ("#2a78d6", "#eb6834", "#1baf7a")
SERIES_DARK = ("#3987e5", "#d95926", "#199e70")

GRID_LIGHT = "rgba(0,0,0,0.10)"
GRID_DARK = "rgba(255,255,255,0.14)"
TEXT_LIGHT = "#52514e"
TEXT_DARK = "#c3c2b7"

CHART_HEIGHT = 420


def theme_mode() -> str:
    """Return 'dark' or 'light' for the viewer's active Streamlit theme."""
    try:
        mode = st.context.theme.type
        if mode in {"light", "dark"}:
            return mode
    except Exception:  # noqa: BLE001 - older Streamlit or no runtime
        pass
    return "dark" if str(st.get_option("theme.base") or "").lower() == "dark" else "light"


def series_color(index: int = 0) -> str:
    """Return the categorical series color for a slot in the active theme."""
    palette = SERIES_DARK if theme_mode() == "dark" else SERIES_LIGHT
    return palette[index % len(palette)]


def style_figure(fig: go.Figure, height: int = CHART_HEIGHT, tight: bool = False) -> go.Figure:
    """Apply the shared chart styling: transparent surface, recessive grid, tight margins."""
    dark = theme_mode() == "dark"
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_DARK if dark else TEXT_LIGHT, size=12),
        margin=dict(l=8, r=16, t=32, b=8) if tight else dict(l=8, r=16, t=40, b=40),
        showlegend=False,
        hoverlabel=dict(font_size=12),
    )
    grid = GRID_DARK if dark else GRID_LIGHT
    fig.update_xaxes(showgrid=True, gridcolor=grid, gridwidth=1, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    return fig


# --- Value formatting --------------------------------------------------------


def format_currency(value: Any) -> str:
    """Format a USD amount with thousands separators, or 'N/A' when unavailable."""
    try:
        if value is None:
            return "N/A"
        return f"${float(value):,.0f}"
    except (TypeError, ValueError):
        return "N/A"


def format_percent(value: Any, decimals: int = 1) -> str:
    """Format an already-scaled percentage value to one decimal, or 'N/A'."""
    try:
        if value is None:
            return "N/A"
        return f"{float(value):.{decimals}f}%"
    except (TypeError, ValueError):
        return "N/A"


def format_fraction_as_percent(value: Any, decimals: int = 1) -> str:
    """Format a 0-1 fraction as a percentage, or 'N/A' when unavailable."""
    try:
        if value is None:
            return "N/A"
        return f"{float(value) * 100:.{decimals}f}%"
    except (TypeError, ValueError):
        return "N/A"


def format_salary_range(low: Any, high: Any) -> str:
    """Format a 25th-75th percentile salary band as '$150,000–$210,000'."""
    if low is None or high is None:
        return "N/A"
    return f"{format_currency(low)}–{format_currency(high)}"


def format_count(value: Any) -> str:
    """Format an integer count with thousands separators."""
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "N/A"


def horizontal_bar(
    labels: list[str],
    values: list[float],
    title: str,
    height: int = CHART_HEIGHT,
    color_index: int = 0,
) -> go.Figure:
    """Build a descending horizontal bar chart with a count labeled on each bar."""
    # Plotly draws the first category at the bottom, so reverse to read top-down.
    fig = go.Figure(
        go.Bar(
            x=list(values)[::-1],
            y=list(labels)[::-1],
            orientation="h",
            marker=dict(color=series_color(color_index), line=dict(width=0)),
            text=[f"{int(v):,}" for v in values][::-1],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x:,} postings<extra></extra>",
        )
    )
    fig.update_layout(title=dict(text=title, font=dict(size=14)))
    style_figure(fig, height=height, tight=True)
    # Headroom so the outside value labels are not clipped in a narrow column.
    fig.update_xaxes(showticklabels=False, showgrid=False, range=[0, max(values) * 1.28 if values else 1])
    fig.update_yaxes(automargin=True, ticklabelposition="outside")
    return fig
