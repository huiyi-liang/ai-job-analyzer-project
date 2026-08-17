"""Load, validate, and scope the AI job postings dataset to the target year."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# --- Constants ---------------------------------------------------------------

TARGET_YEAR = 2026
MIN_ROWS_FOR_RELIABLE_STATS = 20

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BUNDLED_CSV = DATA_DIR / "ai_jobs_market_2025_2026.csv"

ALL_CATEGORIES = "All categories"

REQUIRED_COLUMNS: tuple[str, ...] = (
    "job_id",
    "job_title",
    "job_category",
    "experience_level",
    "years_of_experience",
    "education_required",
    "annual_salary_usd",
    "salary_min_usd",
    "salary_max_usd",
    "city",
    "country",
    "remote_work",
    "company_size",
    "industry",
    "required_skills",
    "ai_salary_premium_pct",
    "demand_score",
    "demand_growth_yoy_pct",
    "benefits_score_10",
    "posting_year",
    "posting_month",
    "is_senior",
    "is_remote_friendly",
    "is_llm_role",
    "salary_tier",
)

# Canonical low-to-high education ladder. The ordinal map is built by filtering
# this list to the values actually present in the active dataset.
EDUCATION_LADDER: tuple[str, ...] = (
    "Bootcamp/Self-taught",
    "Associate's",
    "Bachelor's",
    "Master's",
    "PhD",
)

# Canonical experience display order, matched against real values by prefix
# (real values look like "Entry (0-2 yrs)").
EXPERIENCE_PREFIX_ORDER: tuple[str, ...] = ("Entry", "Mid", "Senior", "Lead")

SKILL_DELIMITER = "|"


class DatasetError(ValueError):
    """Raised when a CSV cannot be used as the app's dataset."""


@dataclass(frozen=True)
class Dataset:
    """A validated dataset exposing the 2026 scope and the full 2025-2026 frame."""

    df_full: pd.DataFrame
    df_2026: pd.DataFrame
    source_label: str
    education_ordinals: dict[str, int]
    level_order: list[str]

    @property
    def total_rows(self) -> int:
        """Row count of the full (all years) dataset."""
        return len(self.df_full)

    @property
    def rows_2026(self) -> int:
        """Row count of the 2026-scoped dataset that every tab but 3C uses."""
        return len(self.df_2026)


# --- Skill parsing -----------------------------------------------------------


def parse_skills(cell: Any) -> list[str]:
    """Split a pipe-delimited required_skills cell into stripped, deduplicated skills."""
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    seen: dict[str, None] = {}
    for raw in str(cell).split(SKILL_DELIMITER):
        skill = raw.strip()
        if skill:
            # dict preserves first-seen order while deduplicating; real rows repeat
            # skills within a single cell (119 rows in the bundled data).
            seen.setdefault(skill, None)
    return list(seen)


def normalize_skill(skill: str) -> str:
    """Return a case- and whitespace-normalized key for matching skills."""
    return " ".join(str(skill).split()).casefold()


# --- Ordering helpers --------------------------------------------------------


def build_education_ordinals(df: pd.DataFrame) -> dict[str, int]:
    """Build a low-to-high education ordinal map from the values present in the data."""
    present = {str(v) for v in df["education_required"].dropna().unique()}
    ordered = [level for level in EDUCATION_LADDER if level in present]
    # Any value outside the canonical ladder is appended above it, in sorted order,
    # so an uploaded CSV with novel values still yields a total ordering.
    ordered += sorted(present - set(EDUCATION_LADDER))
    return {level: index for index, level in enumerate(ordered)}


def build_level_order(df: pd.DataFrame) -> list[str]:
    """Order experience_level values Entry -> Mid -> Senior -> Lead using their prefixes."""
    present = sorted({str(v) for v in df["experience_level"].dropna().unique()})

    def sort_key(value: str) -> tuple[int, str]:
        """Rank a level by its canonical prefix, pushing unknown values to the end."""
        for index, prefix in enumerate(EXPERIENCE_PREFIX_ORDER):
            if value.startswith(prefix):
                return (index, value)
        return (len(EXPERIENCE_PREFIX_ORDER), value)

    return sorted(present, key=sort_key)


# --- Validation --------------------------------------------------------------


def validate_dataframe(df: pd.DataFrame) -> list[str]:
    """Return a list of specific, human-readable reasons the dataframe is unusable."""
    errors: list[str] = []

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        errors.append(f"Missing required column(s): {', '.join(missing)}.")
        # Column-specific checks below would raise KeyError; stop here.
        return errors

    if df.empty:
        errors.append("The file contains no data rows.")
        return errors

    salary = pd.to_numeric(df["annual_salary_usd"], errors="coerce")
    bad_salary = int(salary.isna().sum())
    if bad_salary:
        errors.append(
            f"Column 'annual_salary_usd' has {bad_salary:,} non-numeric value(s)."
        )

    year = pd.to_numeric(df["posting_year"], errors="coerce")
    bad_year = int(year.isna().sum())
    if bad_year:
        errors.append(f"Column 'posting_year' has {bad_year:,} non-numeric value(s).")

    unparseable = int(df["required_skills"].map(lambda c: len(parse_skills(c)) == 0).sum())
    if unparseable:
        errors.append(
            f"Column 'required_skills' yielded no skills for {unparseable:,} row(s); "
            f"values must be pipe-delimited, e.g. 'Python|SQL|Cloud'."
        )

    if not bad_year and int((year == TARGET_YEAR).sum()) == 0:
        errors.append(
            f"No rows with posting_year == {TARGET_YEAR}; the app is scoped to "
            f"{TARGET_YEAR} postings."
        )

    return errors


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce the numeric columns the app relies on, leaving labels untouched."""
    out = df.copy()
    numeric_columns = (
        "years_of_experience",
        "annual_salary_usd",
        "salary_min_usd",
        "salary_max_usd",
        "ai_salary_premium_pct",
        "demand_score",
        "demand_growth_yoy_pct",
        "posting_year",
        "posting_month",
        "is_senior",
        "is_remote_friendly",
        "is_llm_role",
    )
    for column in numeric_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def build_dataset(df: pd.DataFrame, source_label: str) -> Dataset:
    """Validate a raw dataframe and apply the 2026 scope filter exactly once."""
    errors = validate_dataframe(df)
    if errors:
        raise DatasetError(" ".join(errors))

    df_full = _coerce_types(df)
    # THE single 2026 filter for the whole app. Every tab consumes df_2026;
    # df_full is exposed separately and read only by Tab 3 Section C.
    df_2026 = df_full[df_full["posting_year"] == TARGET_YEAR].copy()

    return Dataset(
        df_full=df_full,
        df_2026=df_2026,
        source_label=source_label,
        education_ordinals=build_education_ordinals(df_2026),
        level_order=build_level_order(df_2026),
    )


# --- Loading -----------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_bundled_dataset() -> Dataset:
    """Load and validate the CSV bundled with the app."""
    if not BUNDLED_CSV.exists():
        raise DatasetError(f"Bundled dataset not found at {BUNDLED_CSV}.")
    df = pd.read_csv(BUNDLED_CSV)
    return build_dataset(df, source_label=f"Bundled: {BUNDLED_CSV.name}")


@st.cache_data(show_spinner=False)
def load_uploaded_dataset(file_bytes: bytes, filename: str) -> Dataset:
    """Load and validate an uploaded CSV supplied as raw bytes."""
    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as exc:  # noqa: BLE001 - surfaced as a friendly message
        raise DatasetError(f"Could not read '{filename}' as CSV ({type(exc).__name__}).") from exc
    return build_dataset(df, source_label=f"Uploaded: {filename}")


# --- Filtering ---------------------------------------------------------------


def category_options(df_2026: pd.DataFrame) -> list[str]:
    """Return the single-select job category options, led by the all-categories default."""
    return [ALL_CATEGORIES] + sorted(df_2026["job_category"].dropna().unique().tolist())


def filter_by_category(df: pd.DataFrame, category: str) -> pd.DataFrame:
    """Apply the landscape filter; the all-categories default returns the frame unchanged."""
    if not category or category == ALL_CATEGORIES:
        return df
    return df[df["job_category"] == category]


def benchmark_subset(df_2026: pd.DataFrame, title: str, level: str) -> pd.DataFrame:
    """Return the 2026 postings matching a target job title and experience level."""
    return df_2026[
        (df_2026["job_title"] == title) & (df_2026["experience_level"] == level)
    ]


def title_options(df_2026: pd.DataFrame) -> list[str]:
    """Return job titles ordered by 2026 posting volume, descending."""
    return df_2026["job_title"].value_counts().index.tolist()


def largest_benchmark_target(df_2026: pd.DataFrame, level_order: list[str]) -> tuple[str, str]:
    """Return the (title, level) pair with the most 2026 postings, for use as the default."""
    counts = df_2026.groupby(["job_title", "experience_level"]).size()
    if counts.empty:
        titles = title_options(df_2026)
        return (titles[0] if titles else "", level_order[0] if level_order else "")
    # idxmax breaks ties by first occurrence, which is stable for a given dataset.
    title, level = counts.idxmax()
    return str(title), str(level)


def target_counts(df_2026: pd.DataFrame) -> dict[tuple[str, str], int]:
    """Map each (job title, experience level) pair to its 2026 posting count."""
    counts = df_2026.groupby(["job_title", "experience_level"]).size()
    return {(str(t), str(level)): int(n) for (t, level), n in counts.items()}
