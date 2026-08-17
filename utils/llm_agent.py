"""Groq client wrapper, grounded context assembly, market summary, and chat."""

from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd

from utils.data_loader import ALL_CATEGORIES
from utils.skills_analysis import (
    TOP_SKILLS_FOR_CONTEXT,
    aggregate_skills,
    split_by_threshold,
)

# --- Constants ---------------------------------------------------------------

DEFAULT_MODEL = "llama-3.1-8b-instant"
SUMMARY_MAX_TOKENS = 700
CHAT_MAX_TOKENS = 900
EXTRACTION_MAX_TOKENS = 1400
TEMPERATURE = 0.2
HISTORY_TURNS = 8
TOP_TITLES_FOR_CONTEXT = 10

SETUP_MESSAGE = (
    "Set `GROQ_API_KEY` in a `.env` file at the project root to enable the AI "
    "features. Copy `.env.example` to `.env`, paste your key, and restart the app. "
    "A free key is available at https://console.groq.com/keys."
)


class LLMUnavailable(RuntimeError):
    """Raised when Groq cannot be used (missing key, failed call, or empty response)."""


# --- Client ------------------------------------------------------------------


def get_groq_model() -> str:
    """Return the configured Groq model, falling back to the default."""
    return (os.getenv("GROQ_MODEL") or "").strip() or DEFAULT_MODEL


def groq_api_key() -> str | None:
    """Return the Groq API key from the environment, or None when unset or blank."""
    return (os.getenv("GROQ_API_KEY") or "").strip() or None


def is_configured() -> bool:
    """Report whether a Groq API key is available, without making a network call."""
    return groq_api_key() is not None


def get_client() -> Any:
    """Construct a Groq client, raising LLMUnavailable when it cannot be built."""
    key = groq_api_key()
    if not key:
        raise LLMUnavailable("GROQ_API_KEY is not set.")
    try:
        # Imported lazily so this module stays importable without the SDK installed.
        from groq import Groq
    except ImportError as exc:
        raise LLMUnavailable("The 'groq' package is not installed.") from exc
    return Groq(api_key=key)


def _friendly_error(exc: Exception) -> str:
    """Translate an SDK exception into an actionable message that never echoes the key."""
    name = type(exc).__name__
    if name == "AuthenticationError":
        return (
            "Groq rejected the API key (401). Check that GROQ_API_KEY in your .env "
            "file is current and has no stray quotes or whitespace, then restart."
        )
    if name == "NotFoundError":
        return (
            f"Groq does not recognise the model '{get_groq_model()}'. Set GROQ_MODEL "
            f"in your .env file to a supported model (default: {DEFAULT_MODEL})."
        )
    if name == "RateLimitError":
        return "Groq rate limit reached. Wait a few seconds and try again."
    if name in {"APIConnectionError", "APITimeoutError"}:
        return "Could not reach Groq. Check your network connection and try again."
    return f"Groq request failed ({name})."


def complete(messages: list[dict[str, str]], max_tokens: int = CHAT_MAX_TOKENS) -> str:
    """Send a chat completion to Groq and return its text, or raise LLMUnavailable."""
    client = get_client()
    try:
        response = client.chat.completions.create(
            model=get_groq_model(),
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=max_tokens,
        )
    except Exception as exc:  # noqa: BLE001 - normalized into one exception type
        raise LLMUnavailable(_friendly_error(exc)) from exc

    if not response.choices:
        raise LLMUnavailable("Groq returned an empty response.")
    text = (response.choices[0].message.content or "").strip()
    if not text:
        raise LLMUnavailable("Groq returned an empty response.")
    return text


# --- Context assembly --------------------------------------------------------


def _round(value: Any, digits: int = 1) -> Any:
    """Round a numeric value defensively, passing non-numerics through untouched."""
    try:
        if value is None or pd.isna(value):
            return None
        return round(float(value), digits)
    except (TypeError, ValueError):
        return value


def build_landscape_context(
    df_filtered: pd.DataFrame,
    category: str,
    rows_2026: int,
    total_rows: int,
    level_order: list[str],
) -> dict[str, Any]:
    """Assemble aggregate-only landscape context; raw posting rows are never included."""
    n = len(df_filtered)
    context: dict[str, Any] = {
        "scope": (
            f"2026 postings only ({rows_2026} of {total_rows} rows in the dataset). "
            f"Tab 3 Section C is the only view that also uses 2025 data."
        ),
        "active_category_filter": category or ALL_CATEGORIES,
        "filtered_posting_count": n,
        "postings_2026_total": rows_2026,
    }

    if n == 0:
        context["note"] = "No postings match the active filter."
        return context

    context["headline_metrics"] = {
        "median_annual_salary_usd": _round(df_filtered["annual_salary_usd"].median(), 0),
        "avg_demand_score": _round(df_filtered["demand_score"].mean()),
        "pct_remote_friendly": _round(df_filtered["is_remote_friendly"].mean() * 100),
        "pct_llm_related": _round(df_filtered["is_llm_role"].mean() * 100),
        "based_on_postings": n,
    }

    by_level = []
    for level in level_order:
        subset = df_filtered[df_filtered["experience_level"] == level]
        if subset.empty:
            continue
        by_level.append(
            {
                "experience_level": level,
                "postings": len(subset),
                "median_salary_usd": _round(subset["annual_salary_usd"].median(), 0),
                "p25_salary_usd": _round(subset["annual_salary_usd"].quantile(0.25), 0),
                "p75_salary_usd": _round(subset["annual_salary_usd"].quantile(0.75), 0),
            }
        )
    context["salary_by_experience_level"] = by_level or "unavailable"

    titles = (
        df_filtered.groupby("job_title")
        .agg(
            postings=("job_id", "count"),
            median_salary_usd=("annual_salary_usd", "median"),
            avg_demand_score=("demand_score", "mean"),
            avg_growth_yoy_pct=("demand_growth_yoy_pct", "mean"),
        )
        .sort_values("postings", ascending=False)
        .head(TOP_TITLES_FOR_CONTEXT)
    )
    context["top_job_titles_by_volume"] = [
        {
            "job_title": str(title),
            "postings": int(row.postings),
            "median_salary_usd": _round(row.median_salary_usd, 0),
            "avg_demand_score": _round(row.avg_demand_score),
            "avg_growth_yoy_pct": _round(row.avg_growth_yoy_pct),
        }
        for title, row in titles.iterrows()
    ] or "unavailable"

    skills = aggregate_skills(df_filtered)
    kept, excluded, threshold = split_by_threshold(skills, n)
    context["skill_posting_threshold"] = {
        "threshold": threshold,
        "rule": "max(10, 2% of filtered postings)",
        "skills_excluded_from_salary_growth_view": int(len(excluded)),
        "excluded_skill_names": excluded["skill"].head(15).tolist() if not excluded.empty else [],
    }
    top_skills = kept.head(TOP_SKILLS_FOR_CONTEXT) if not kept.empty else skills.head(0)
    context["top_skills"] = [
        {
            "skill": str(row.skill),
            "postings": int(row.postings),
            "pct_of_filtered_postings": _round(row.pct_of_postings * 100),
            "median_salary_usd": _round(row.median_salary, 0),
            "avg_growth_yoy_pct_of_roles_requiring_it": _round(row.mean_growth),
        }
        for row in top_skills.itertuples()
    ] or "unavailable"
    context["skill_growth_caveat"] = (
        "demand_growth_yoy_pct is a per-posting attribute of the ROLE, not the skill. "
        "Skill growth figures are the average growth of roles requiring that skill — "
        "a proxy, not a measurement."
    )
    return context


def build_fit_context(
    benchmark: Any, fit_result: dict[str, Any] | None
) -> dict[str, Any] | str:
    """Assemble the scored-resume half of the context, or a marker when unscored."""
    if benchmark is None or fit_result is None:
        return "no resume scored yet"

    rows = fit_result.get("skill_rows", [])
    demonstrated = [r for r in rows if r["gap_type"] == "none"]
    listed_only = [r for r in rows if r["gap_type"] == "listed_only"]
    missing = [r for r in rows if r["gap_type"] == "missing"]

    def gap_rows(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Reduce skill rows to name, tier, and requirement frequency."""
        return [
            {"skill": r["skill"], "tier": r["tier"], "required_in_pct": _round(r["pct"] * 100)}
            for r in items[:12]
        ]

    return {
        "target_job_title": benchmark.title,
        "target_experience_level": benchmark.level,
        "benchmark_posting_count": benchmark.n_postings,
        "benchmark_requirements": {
            "median_years_of_experience": benchmark.years_median,
            "years_range": [benchmark.years_min, benchmark.years_max],
            "most_common_education": benchmark.education_mode,
            "median_salary_usd": _round(benchmark.salary_median, 0),
            "salary_p25_p75_usd": [
                _round(benchmark.salary_p25, 0),
                _round(benchmark.salary_p75, 0),
            ],
        },
        "candidate": {
            "total_years_experience": fit_result["total_years"],
            "related_years_experience": fit_result["related_years"],
            "education_level": fit_result["candidate_education"],
            "demonstrated_skill_count": len(demonstrated),
            "listed_only_skill_count": len(listed_only),
            "missing_skill_count": len(missing),
        },
        "fit_score_pct": _round(fit_result["final_score"] * 100),
        "skills_score_pct": _round(fit_result["skills_score"] * 100),
        "applied_modifier": _round(fit_result["modifier"], 3),
        "meets_bar": fit_result["meets_bar"],
        "meets_experience": fit_result["meets_experience"],
        "meets_education": fit_result["meets_education"],
        "gaps_have_it_but_not_demonstrated": gap_rows(listed_only),
        "gaps_missing_entirely": gap_rows(missing),
        "scoring_rule": (
            "Skill credit is weighted by how often each skill is required. "
            "demonstrated = 1.0. listed_only = 0.5 only if the candidate meets BOTH the "
            "experience and education bar, otherwise 0.0. Missing = 0.0."
        ),
    }


def build_context(
    df_filtered: pd.DataFrame,
    category: str,
    rows_2026: int,
    total_rows: int,
    level_order: list[str],
    benchmark: Any = None,
    fit_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the full grounded context passed to the model."""
    context = build_landscape_context(
        df_filtered, category, rows_2026, total_rows, level_order
    )
    context["resume_fit"] = build_fit_context(benchmark, fit_result)
    return context


def format_context(context: dict[str, Any]) -> str:
    """Serialize the context dict for inclusion in a system message."""
    return json.dumps(context, indent=2, default=str)


# --- Prompts -----------------------------------------------------------------

SYSTEM_PROMPT = """You are a careful analyst reading one snapshot of AI job postings.

STRICT GROUNDING RULES:
1. Use ONLY figures present in the CONTEXT block. Never invent numbers, companies,
   skills, job titles, or trends. If something is not in the context, say it is not
   available in this dataset rather than estimating.
2. Cite the posting count alongside any skill-level or role-level figure, e.g.
   "Python (560 postings)". Explicitly flag figures backed by few postings as
   low-confidence.
3. When discussing skill growth, repeat the proxy caveat: demand_growth_yoy_pct
   belongs to the ROLE, not the skill, so a skill's growth number is the average
   growth of roles requiring it — a proxy, not a measurement.
4. Always distinguish RELATED years of experience from TOTAL years. Only related
   years are used in scoring.
5. If the candidate is below the bar, explain that listed-only skills earn zero
   credit for them, and point to "have it, but not demonstrated" skills as the
   cheapest wins — those need evidence of use, not new learning.
6. All figures describe 2026 postings only, unless the context says otherwise.
7. Close with a one-line disclaimer: this is one dataset snapshot of 2026 postings,
   not comprehensive labor market data, and not career or hiring advice.

Write in plain prose. Be concrete and brief. Do not use headings unless asked."""

SUMMARY_INSTRUCTION = """Write a 3-5 sentence summary of what this data shows.

If a resume has been scored, cover both the market picture and the candidate's fit,
including their score, whether they meet the bar, and their most valuable gap.
If no resume has been scored, write a genuinely useful standalone read of the market:
what pays, what is growing, and what is most in demand. Do not mention the absence of
a resume more than once, and do not pad."""


def generate_summary(context: dict[str, Any]) -> str:
    """Generate the auto-summary shown when the AI read tab loads."""
    return complete(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": f"CONTEXT:\n{format_context(context)}"},
            {"role": "user", "content": SUMMARY_INSTRUCTION},
        ],
        max_tokens=SUMMARY_MAX_TOKENS,
    )


def chat(
    context: dict[str, Any], history: list[dict[str, str]], question: str
) -> str:
    """Answer a follow-up question grounded in the context and recent chat history."""
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"CONTEXT:\n{format_context(context)}"},
    ]
    for turn in history[-HISTORY_TURNS:]:
        role, content = turn.get("role"), turn.get("content")
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question})
    return complete(messages, max_tokens=CHAT_MAX_TOKENS)
