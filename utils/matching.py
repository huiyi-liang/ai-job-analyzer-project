"""Benchmark profiles and deterministic, frequency-weighted resume fit scoring.

Everything in this module is pure Python and pandas. No LLM call happens here and
no number in the fit result is ever produced or adjusted by a model — the same
resume scored against the same target always yields the same score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from utils.data_loader import normalize_skill, parse_skills

# --- Constants ---------------------------------------------------------------

SKILL_TIER_CORE_PCT = 0.50
SKILL_TIER_COMMON_PCT = 0.20
LISTED_ONLY_CREDIT_MEETS_BAR = 0.5
LISTED_ONLY_CREDIT_BELOW_BAR = 0.0
MODIFIER_MAX = 0.15
MIN_TARGET_POSTINGS_WARNING = 10

# Distances are measured in absolute units and normalized by these spans, never as
# a ratio to the benchmark — a candidate with 0 related years must not divide by zero.
EXPERIENCE_DISTANCE_CAP_YEARS = 5.0
EDUCATION_DISTANCE_CAP_LEVELS = 4.0

TIER_CORE = "Core"
TIER_COMMON = "Common"
TIER_OCCASIONAL = "Occasional"

EVIDENCE_DEMONSTRATED = "demonstrated"
EVIDENCE_LISTED_ONLY = "listed_only"
EVIDENCE_NONE = "none"
EVIDENCE_CHOICES = (EVIDENCE_DEMONSTRATED, EVIDENCE_LISTED_ONLY, EVIDENCE_NONE)

STATUS_DEMONSTRATED = "✅ Demonstrated"
STATUS_LISTED_ONLY = "➖ Listed only"
STATUS_MISSING = "❌ Missing"


@dataclass(frozen=True)
class Benchmark:
    """An aggregate requirements profile for one target job title and experience level."""

    title: str
    level: str
    n_postings: int
    years_median: float | None
    years_min: float | None
    years_max: float | None
    education_mode: str | None
    education_distribution: list[dict[str, Any]]
    salary_median: float | None
    salary_p25: float | None
    salary_p75: float | None
    skills: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """True when no postings match the target title and level."""
        return self.n_postings == 0

    @property
    def is_low_sample(self) -> bool:
        """True when too few postings back the benchmark to read it confidently."""
        return 0 < self.n_postings < MIN_TARGET_POSTINGS_WARNING


def tier_for_frequency(pct: float) -> str:
    """Classify a skill requirement frequency into Core, Common, or Occasional."""
    if pct >= SKILL_TIER_CORE_PCT:
        return TIER_CORE
    if pct >= SKILL_TIER_COMMON_PCT:
        return TIER_COMMON
    return TIER_OCCASIONAL


def build_benchmark(subset: pd.DataFrame, title: str, level: str) -> Benchmark:
    """Build the aggregate benchmark profile from postings matching a title and level."""
    n_postings = len(subset)
    if n_postings == 0:
        return Benchmark(
            title=title,
            level=level,
            n_postings=0,
            years_median=None,
            years_min=None,
            years_max=None,
            education_mode=None,
            education_distribution=[],
            salary_median=None,
            salary_p25=None,
            salary_p75=None,
            skills=[],
        )

    years = subset["years_of_experience"].dropna()
    salary = subset["annual_salary_usd"].dropna()
    education_counts = subset["education_required"].value_counts()

    education_distribution = [
        {"education": str(name), "postings": int(count), "pct": int(count) / n_postings}
        for name, count in education_counts.items()
    ]

    # Skill frequency across the benchmark, deduplicated within each posting.
    skill_counts: dict[str, int] = {}
    for cell in subset["required_skills"]:
        for skill in parse_skills(cell):
            skill_counts[skill] = skill_counts.get(skill, 0) + 1

    skills = [
        {
            "skill": skill,
            "postings": count,
            "pct": count / n_postings,
            "tier": tier_for_frequency(count / n_postings),
        }
        for skill, count in skill_counts.items()
    ]
    skills.sort(key=lambda row: (-row["pct"], row["skill"]))

    return Benchmark(
        title=title,
        level=level,
        n_postings=n_postings,
        years_median=float(years.median()) if not years.empty else None,
        years_min=float(years.min()) if not years.empty else None,
        years_max=float(years.max()) if not years.empty else None,
        education_mode=str(education_counts.index[0]) if not education_counts.empty else None,
        education_distribution=education_distribution,
        salary_median=float(salary.median()) if not salary.empty else None,
        salary_p25=float(salary.quantile(0.25)) if not salary.empty else None,
        salary_p75=float(salary.quantile(0.75)) if not salary.empty else None,
        skills=skills,
    )


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp a value into an inclusive range."""
    return max(low, min(high, value))


def _education_ordinal(level: str | None, ordinals: dict[str, int]) -> int | None:
    """Look up an education ordinal, returning None when the level is unknown."""
    if level is None:
        return None
    return ordinals.get(str(level))


def compute_modifier(
    related_years: float,
    benchmark_years: float | None,
    candidate_ordinal: int | None,
    benchmark_ordinal: int | None,
) -> dict[str, float]:
    """Compute the bounded ±MODIFIER_MAX adjustment from absolute distances above the bars."""
    if benchmark_years is None:
        experience_distance = 0.0
    else:
        # Absolute years, never a ratio: 0 related years is a valid input, not a divide.
        experience_distance = _clamp(
            (float(related_years) - float(benchmark_years)) / EXPERIENCE_DISTANCE_CAP_YEARS,
            -1.0,
            1.0,
        )

    if candidate_ordinal is None or benchmark_ordinal is None:
        education_distance = 0.0
    else:
        education_distance = _clamp(
            (candidate_ordinal - benchmark_ordinal) / EDUCATION_DISTANCE_CAP_LEVELS, -1.0, 1.0
        )

    # Averaging two values each bounded to [-1, 1] keeps the modifier within
    # ±MODIFIER_MAX by construction, so it can never reorder two skill profiles
    # that differ by more than MODIFIER_MAX.
    modifier = MODIFIER_MAX * (experience_distance + education_distance) / 2.0
    return {
        "modifier": modifier,
        "experience_distance": experience_distance,
        "education_distance": education_distance,
    }


def _why_text(evidence: str, meets_bar: bool, credit: float, pct: float) -> str:
    """Explain in one sentence why a skill earned the credit it did."""
    share = f"{pct:.1%} of benchmark postings"
    if evidence == EVIDENCE_DEMONSTRATED:
        return f"Demonstrated in your resume; full credit against {share}."
    if evidence == EVIDENCE_LISTED_ONLY:
        if meets_bar:
            return (
                f"Listed but not demonstrated; half credit "
                f"({LISTED_ONLY_CREDIT_MEETS_BAR}) against {share}."
            )
        return (
            f"Listed but not demonstrated, and you are below the experience or "
            f"education bar — this earns zero credit. Describing where you actually "
            f"used it is the cheapest way to gain points ({share})."
        )
    return f"Not found in your resume; no credit against {share}."


def score_fit(
    benchmark: Benchmark,
    profile: dict[str, Any],
    education_ordinals: dict[str, int],
) -> dict[str, Any]:
    """Score a resume profile against a benchmark using frequency-weighted skill credit."""
    total_years = float(profile.get("total_years_experience") or 0)
    related_years = float(profile.get("related_years_experience") or 0)
    candidate_education = profile.get("education_level")

    candidate_ordinal = _education_ordinal(candidate_education, education_ordinals)
    benchmark_ordinal = _education_ordinal(benchmark.education_mode, education_ordinals)

    # (a) meets_bar uses RELATED years, never total.
    meets_experience = (
        benchmark.years_median is None or related_years >= float(benchmark.years_median)
    )
    meets_education = benchmark_ordinal is None or (
        candidate_ordinal is not None and candidate_ordinal >= benchmark_ordinal
    )
    meets_bar = bool(meets_experience and meets_education)

    # Candidate evidence, keyed case- and whitespace-insensitively.
    evidence_by_skill: dict[str, str] = {}
    for entry in profile.get("skills") or []:
        skill = entry.get("skill")
        if not skill:
            continue
        evidence = str(entry.get("evidence") or EVIDENCE_NONE).strip().lower()
        if evidence not in EVIDENCE_CHOICES:
            evidence = EVIDENCE_NONE
        evidence_by_skill[normalize_skill(skill)] = evidence

    # (b) Frequency-weighted skill credit.
    listed_only_credit = (
        LISTED_ONLY_CREDIT_MEETS_BAR if meets_bar else LISTED_ONLY_CREDIT_BELOW_BAR
    )
    rows: list[dict[str, Any]] = []
    weighted_sum = 0.0
    weight_total = 0.0

    for skill_row in benchmark.skills:
        skill = skill_row["skill"]
        pct = float(skill_row["pct"])
        evidence = evidence_by_skill.get(normalize_skill(skill), EVIDENCE_NONE)

        if evidence == EVIDENCE_DEMONSTRATED:
            credit = 1.0
            status = STATUS_DEMONSTRATED
        elif evidence == EVIDENCE_LISTED_ONLY:
            credit = listed_only_credit
            status = STATUS_LISTED_ONLY
        else:
            credit = 0.0
            status = STATUS_MISSING

        weighted_sum += credit * pct
        weight_total += pct
        rows.append(
            {
                "skill": skill,
                "pct": pct,
                "tier": skill_row["tier"],
                "evidence": evidence,
                "status": status,
                "credit": credit,
                "is_gap": credit < 1.0,
                "gap_type": (
                    "none"
                    if evidence == EVIDENCE_DEMONSTRATED
                    else "listed_only"
                    if evidence == EVIDENCE_LISTED_ONLY
                    else "missing"
                ),
                "why": _why_text(evidence, meets_bar, credit, pct),
            }
        )

    skills_score = (weighted_sum / weight_total) if weight_total > 0 else 0.0

    # (c) Bounded modifier, then (d) clamp to [0, 1].
    modifier_parts = compute_modifier(
        related_years, benchmark.years_median, candidate_ordinal, benchmark_ordinal
    )
    modifier = modifier_parts["modifier"]
    final_score = _clamp(skills_score * (1.0 + modifier), 0.0, 1.0)

    return {
        "final_score": final_score,
        "skills_score": skills_score,
        "modifier": modifier,
        "experience_distance": modifier_parts["experience_distance"],
        "education_distance": modifier_parts["education_distance"],
        "meets_bar": meets_bar,
        "meets_experience": bool(meets_experience),
        "meets_education": bool(meets_education),
        "listed_only_credit": listed_only_credit,
        "total_years": total_years,
        "related_years": related_years,
        "candidate_education": candidate_education,
        "candidate_ordinal": candidate_ordinal,
        "benchmark_ordinal": benchmark_ordinal,
        "benchmark_years_median": benchmark.years_median,
        "benchmark_education_mode": benchmark.education_mode,
        "benchmark_postings": benchmark.n_postings,
        "skill_rows": rows,
    }
