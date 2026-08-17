"""Explode, threshold, and aggregate required_skills — every statistic carries its count."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from utils.data_loader import parse_skills

# --- Constants ---------------------------------------------------------------

MIN_POSTINGS_PER_SKILL_FLOOR = 10
MIN_POSTINGS_PER_SKILL_PCT = 0.02
TOP_SKILLS_FREQUENCY_CHART = 20
TOP_SKILLS_FOR_CONTEXT = 15


def min_postings_threshold(n_rows: int) -> int:
    """Return the minimum postings a skill needs to appear in the scatter."""
    return max(MIN_POSTINGS_PER_SKILL_FLOOR, int(round(MIN_POSTINGS_PER_SKILL_PCT * n_rows)))


@st.cache_data(show_spinner=False)
def explode_skills(df: pd.DataFrame) -> pd.DataFrame:
    """Explode postings to one row per (posting, deduplicated skill)."""
    if df.empty:
        return pd.DataFrame(columns=["job_id", "skill", "annual_salary_usd", "demand_growth_yoy_pct"])
    working = df[["job_id", "annual_salary_usd", "demand_growth_yoy_pct", "required_skills"]].copy()
    working["skill"] = working["required_skills"].map(parse_skills)
    exploded = working.explode("skill", ignore_index=True)
    exploded = exploded[exploded["skill"].notna() & (exploded["skill"] != "")]
    return exploded.drop(columns=["required_skills"])


@st.cache_data(show_spinner=False)
def aggregate_skills(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-skill posting count, share, median salary, and mean growth."""
    exploded = explode_skills(df)
    n_rows = len(df)
    if exploded.empty or n_rows == 0:
        return pd.DataFrame(
            columns=["skill", "postings", "pct_of_postings", "median_salary", "mean_growth"]
        )

    grouped = exploded.groupby("skill").agg(
        postings=("job_id", "nunique"),
        median_salary=("annual_salary_usd", "median"),
        mean_growth=("demand_growth_yoy_pct", "mean"),
    )
    grouped["pct_of_postings"] = grouped["postings"] / n_rows
    return (
        grouped.reset_index()
        .sort_values("postings", ascending=False, kind="mergesort")
        .reset_index(drop=True)
    )


def split_by_threshold(
    skills: pd.DataFrame, n_rows: int
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Split aggregated skills into kept and excluded sets by the posting threshold.

    A skill appearing in only 2-3 postings can top a median-salary chart purely by
    chance, so rare skills are held out of the scatter and reported instead.
    """
    threshold = min_postings_threshold(n_rows)
    if skills.empty:
        return skills, skills, threshold
    kept = skills[skills["postings"] >= threshold].reset_index(drop=True)
    excluded = skills[skills["postings"] < threshold].reset_index(drop=True)
    return kept, excluded, threshold


def unique_skill_count(df: pd.DataFrame) -> int:
    """Count the distinct skills appearing across a set of postings."""
    exploded = explode_skills(df)
    return int(exploded["skill"].nunique()) if not exploded.empty else 0


def skill_frequency_in(df: pd.DataFrame) -> pd.Series:
    """Return each skill's share of the given postings, as a 0-1 float indexed by skill."""
    exploded = explode_skills(df)
    n_rows = len(df)
    if exploded.empty or n_rows == 0:
        return pd.Series(dtype="float64")
    return exploded.groupby("skill")["job_id"].nunique() / n_rows
