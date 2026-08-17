"""Tab 4: resume fit — benchmark profile, LLM extraction, editable confirmation, scoring."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from components import (
    format_currency,
    format_fraction_as_percent,
    format_salary_range,
)
from utils.data_loader import EDUCATION_LADDER
from utils.llm_agent import SETUP_MESSAGE, LLMUnavailable, is_configured
from utils.matching import (
    Benchmark,
    EVIDENCE_CHOICES,
    EVIDENCE_LISTED_ONLY,
    LISTED_ONLY_CREDIT_MEETS_BAR,
    MIN_TARGET_POSTINGS_WARNING,
    score_fit,
)
from utils.resume_parser import ResumeParseError, extract_profile

FILTER_ALL = "All skills"
FILTER_GAPS = "Only my gaps"


# --- Step 0 ------------------------------------------------------------------


def _render_benchmark(benchmark: Benchmark, level_order: list[str]) -> None:
    """Render Step 0: the aggregate benchmark profile, which needs no resume and no LLM key."""
    st.subheader("Step 0 — Your benchmark")
    st.caption(
        f"Built from **all 2026 postings** matching **{benchmark.title}** at "
        f"**{benchmark.level}**. This is one aggregate profile, not a ranking of "
        f"individual postings."
    )

    if benchmark.is_empty:
        alternatives = ", ".join(lv for lv in level_order if lv != benchmark.level)
        st.error(
            f"No 2026 postings match **{benchmark.title}** at **{benchmark.level}**, "
            f"so there is nothing to benchmark against. Try another experience level "
            f"({alternatives}) or a different job title in the sidebar."
        )
        return

    if benchmark.is_low_sample:
        st.warning(
            f"⚠️ **Only {benchmark.n_postings} posting(s) back this benchmark** — fewer "
            f"than {MIN_TARGET_POSTINGS_WARNING}. Every figure below rests on a very "
            f"small sample and can swing on a single posting. Consider a different "
            f"experience level or job title in the sidebar."
        )

    columns = st.columns(4)
    columns[0].metric("Benchmark postings", f"{benchmark.n_postings:,}")
    columns[1].metric(
        "Median years required",
        f"{benchmark.years_median:.0f}" if benchmark.years_median is not None else "N/A",
        help=f"Range {benchmark.years_min:.0f}–{benchmark.years_max:.0f} years"
        if benchmark.years_min is not None
        else None,
    )
    columns[2].metric("Most common education", benchmark.education_mode or "N/A")
    columns[3].metric("Median salary", format_currency(benchmark.salary_median))
    st.caption(
        f"Years range: {benchmark.years_min:.0f}–{benchmark.years_max:.0f}. "
        f"Salary 25th–75th percentile: "
        f"{format_salary_range(benchmark.salary_p25, benchmark.salary_p75)}. "
        f"All figures from the same {benchmark.n_postings:,} postings."
        if benchmark.years_min is not None
        else f"All figures from the same {benchmark.n_postings:,} postings."
    )

    left, right = st.columns(2)
    with left:
        st.markdown("**Education required — full distribution**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Education": row["education"],
                        "Postings": row["postings"],
                        "% of benchmark": f"{row['pct']:.1%}",
                    }
                    for row in benchmark.education_distribution
                ]
            ),
            hide_index=True,
            width="stretch",
        )
    with right:
        st.markdown("**Skill requirements by tier**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Skill": row["skill"],
                        "Tier": row["tier"],
                        "% requiring it": f"{row['pct']:.1%}",
                    }
                    for row in benchmark.skills
                ]
            ),
            hide_index=True,
            width="stretch",
            height=320,
        )
    st.caption(
        "Tiers: **Core** = required in 50% or more of benchmark postings · "
        "**Common** = 20% up to but not including 50% · **Occasional** = under 20%. "
        f"Percentages are of the {benchmark.n_postings:,} benchmark postings."
    )


# --- Steps 1-2 ---------------------------------------------------------------


def _render_extraction(
    resume_text: str, benchmark: Benchmark, target_category: str
) -> None:
    """Render Steps 1-2: resume input status and the single LLM extraction call."""
    st.subheader("Steps 1–2 — Read your resume")
    st.caption(
        f"{len(resume_text):,} characters received. The AI is used **only here**, to turn "
        f"your resume into structured fields. It never produces or adjusts your score."
    )

    if not is_configured():
        st.info(f"🔑 **AI extraction unavailable.** {SETUP_MESSAGE}")
        return

    already = st.session_state.get("resume_profile") is not None
    label = "Re-read my resume" if already else "Read my resume"
    if st.button(label, type="primary" if not already else "secondary", key="extract_button"):
        with st.spinner("Reading your resume…"):
            try:
                profile = extract_profile(
                    resume_text,
                    benchmark.title,
                    target_category,
                    [row["skill"] for row in benchmark.skills],
                )
                st.session_state["resume_profile"] = profile
                st.session_state["extraction_error"] = None
                # A fresh read invalidates any score built on the previous read.
                st.session_state["fit_result"] = None
                st.session_state["scored_target"] = None
            except (LLMUnavailable, ResumeParseError) as exc:
                st.session_state["extraction_error"] = str(exc)
        st.rerun()

    error = st.session_state.get("extraction_error")
    if error:
        st.error(f"Could not read your resume — {error}")
        st.caption("Use the button above to retry.")


# --- Step 3 ------------------------------------------------------------------


def _render_confirmation(
    profile: dict[str, Any], benchmark: Benchmark, education_ordinals: dict[str, int]
) -> None:
    """Render Step 3: the editable confirmation form that must precede scoring."""
    st.subheader("Step 3 — Check and correct what was read")
    st.info(
        "**These classifications change your score.** A skill marked *demonstrated* "
        "earns full credit; *listed only* earns half credit, and **zero** if you are "
        "below the experience or education bar. Correct anything the model got wrong "
        "before scoring."
    )

    left, right = st.columns([1, 2])
    with left:
        total_years = st.number_input(
            "Total years of experience",
            min_value=0,
            max_value=60,
            value=int(profile.get("total_years_experience") or 0),
            key="edit_total_years",
            help="All professional experience. NOT used for scoring.",
        )
        related_years = st.number_input(
            "Related years of experience",
            min_value=0,
            max_value=60,
            value=int(profile.get("related_years_experience") or 0),
            key="edit_related_years",
            help="Only experience relevant to the target role. This IS what scoring uses.",
        )
        education_options = list(EDUCATION_LADDER)
        current = profile.get("education_level")
        education = st.selectbox(
            "Education level",
            options=education_options,
            index=education_options.index(current) if current in education_options else 0,
            key="edit_education",
        )
    with right:
        st.markdown("**Why the model counted those related years**")
        st.caption(
            profile.get("related_experience_reasoning")
            or "The model gave no reasoning. Check the related-years figure yourself."
        )
        st.caption(
            "Only **related** years are compared against the benchmark. Total years are "
            "shown for context and never scored."
        )

    st.markdown("**Your skills** — flip the evidence, delete rows, or add missing skills.")
    editor_rows = pd.DataFrame(
        [
            {
                "Skill": entry["skill"],
                "Evidence": entry.get("evidence") or EVIDENCE_LISTED_ONLY,
                "Where it appeared": entry.get("source") or "",
            }
            for entry in profile.get("skills", [])
        ]
    )
    if editor_rows.empty:
        editor_rows = pd.DataFrame(columns=["Skill", "Evidence", "Where it appeared"])
        st.warning(
            "No skills were identified in your resume. Add the ones you have below, or "
            "score as-is to see the full gap list."
        )

    edited = st.data_editor(
        editor_rows,
        key="skill_editor",
        num_rows="dynamic",
        width="stretch",
        hide_index=True,
        column_config={
            "Skill": st.column_config.TextColumn("Skill", required=True),
            "Evidence": st.column_config.SelectboxColumn(
                "Evidence", options=list(EVIDENCE_CHOICES), required=False,
                help="demonstrated = described as actual work · listed_only = named only · none = absent",
            ),
            "Where it appeared": st.column_config.TextColumn("Where it appeared"),
        },
    )

    if st.button("Score my fit", type="primary", key="score_button"):
        skills = [
            {
                "skill": str(row["Skill"]).strip(),
                # Skills the user adds default to listed_only.
                "evidence": (row["Evidence"] or EVIDENCE_LISTED_ONLY),
                "source": row["Where it appeared"] or None,
            }
            for _, row in edited.iterrows()
            if str(row.get("Skill") or "").strip()
        ]
        confirmed = {
            "total_years_experience": int(total_years),
            "related_years_experience": int(related_years),
            "related_experience_reasoning": profile.get("related_experience_reasoning", ""),
            "education_level": education,
            "skills": skills,
        }
        with st.spinner("Scoring…"):
            st.session_state["resume_profile"] = confirmed
            st.session_state["fit_result"] = score_fit(benchmark, confirmed, education_ordinals)
            st.session_state["scored_target"] = (benchmark.title, benchmark.level)
        st.rerun()


# --- Steps 4-5 ---------------------------------------------------------------


def _render_results(result: dict[str, Any], benchmark: Benchmark) -> None:
    """Render Steps 4-5: the headline score, comparison rows, and skill breakdown."""
    st.subheader("Steps 4–5 — Your fit")

    headline, badge = st.columns([1, 3])
    headline.metric("Fit score", format_fraction_as_percent(result["final_score"]))
    with badge:
        if result["meets_bar"]:
            st.success("**Meets experience & education** — listed-only skills earn half credit.")
        else:
            missed = []
            if not result["meets_experience"]:
                missed.append("related experience")
            if not result["meets_education"]:
                missed.append("education")
            st.warning(
                f"**Below bar — only demonstrated skills counted.** You are below the "
                f"benchmark on {' and '.join(missed)}, so merely listing a skill earns "
                f"**zero** credit; it must be demonstrated through project or work "
                f"evidence to compensate for a credentials gap."
            )

    st.markdown("**How you compare**")
    experience, education, salary = st.columns(3)
    with experience:
        st.markdown("**Experience**")
        st.markdown(
            f"Related: **{result['related_years']:.0f} yrs** vs benchmark median "
            f"**{result['benchmark_years_median']:.0f} yrs**"
            if result["benchmark_years_median"] is not None
            else f"Related: **{result['related_years']:.0f} yrs**"
        )
        st.caption(f"Total experience: {result['total_years']:.0f} yrs — NOT used for scoring.")
    with education:
        st.markdown("**Education**")
        st.markdown(
            f"Yours: **{result['candidate_education'] or 'N/A'}** vs most common "
            f"**{result['benchmark_education_mode'] or 'N/A'}**"
        )
        st.caption("Compared by ordinal rank, not by exact match.")
    with salary:
        st.markdown("**Salary context**")
        st.markdown(f"Median: **{format_currency(benchmark.salary_median)}**")
        st.caption(
            f"25th–75th: {format_salary_range(benchmark.salary_p25, benchmark.salary_p75)} "
            f"· {benchmark.n_postings:,} postings"
        )

    st.divider()
    st.markdown("**Skill breakdown**")
    choice = st.radio(
        "Show", [FILTER_ALL, FILTER_GAPS], horizontal=True, key="skill_filter",
        label_visibility="collapsed",
    )
    rows = result["skill_rows"]
    shown = [r for r in rows if r["is_gap"]] if choice == FILTER_GAPS else rows

    if not shown:
        st.success("No gaps — every benchmark skill is demonstrated in your resume.")
    else:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Skill": r["skill"],
                        "Required in": f"{r['pct']:.1%}",
                        "Tier": r["tier"],
                        "Your status": r["status"],
                        "Credit given": f"{r['credit']:.2f}",
                        "Why": r["why"],
                    }
                    for r in shown
                ]
            ),
            hide_index=True,
            width="stretch",
            height=430,
        )
    listed_only = [r for r in rows if r["gap_type"] == "listed_only"]
    missing = [r for r in rows if r["gap_type"] == "missing"]
    st.caption(
        f"Ordered by requirement frequency, descending, across "
        f"{benchmark.n_postings:,} benchmark postings. "
        f"**{len(listed_only)} skill(s) you have but have not demonstrated** — the cheaper "
        f"gap to close, since it needs evidence of use rather than new learning — versus "
        f"**{len(missing)} missing entirely**."
    )

    with st.expander("How this score was calculated"):
        st.markdown(
            f"- **Skills score:** {format_fraction_as_percent(result['skills_score'])} "
            f"— Σ(credit × requirement frequency) ÷ Σ(requirement frequency)\n"
            f"- **Applied modifier:** {result['modifier']:+.3f} "
            f"(bounded to ±0.15, from distance above/below the bars)\n"
            f"- **Final score:** skills score × (1 + modifier), clamped to 0–100% "
            f"= **{format_fraction_as_percent(result['final_score'])}**\n"
            f"- **Listed-only credit in effect:** {result['listed_only_credit']} "
            f"(={LISTED_ONLY_CREDIT_MEETS_BAR} only when you meet both bars)\n\n"
            f"Scoring is deterministic Python — the same inputs always give the same score. "
            f"No AI model produced or adjusted any number here."
        )


# --- Entry point -------------------------------------------------------------


def _render(
    benchmark: Benchmark,
    resume_text: str,
    target_category: str,
    education_ordinals: dict[str, int],
    level_order: list[str],
) -> None:
    """Render the full resume fit tab."""
    _render_benchmark(benchmark, level_order)
    st.divider()

    if benchmark.is_empty:
        return

    if not resume_text:
        st.info(
            "👈 **Add your resume to see how you compare.** Upload a PDF or paste your "
            "resume text in the **Resume Fit** section of the sidebar. The benchmark "
            "above is complete and needs no resume."
        )
        return

    if st.session_state.get("target_changed_message"):
        st.warning(st.session_state["target_changed_message"])

    _render_extraction(resume_text, benchmark, target_category)

    profile = st.session_state.get("resume_profile")
    if not profile:
        return

    st.divider()
    _render_confirmation(profile, benchmark, education_ordinals)

    result = st.session_state.get("fit_result")
    if result:
        st.divider()
        _render_results(result, benchmark)


def render(
    benchmark: Benchmark,
    resume_text: str,
    target_category: str,
    education_ordinals: dict[str, int],
    level_order: list[str],
) -> None:
    """Render the resume fit tab, isolating any failure to this tab."""
    try:
        _render(benchmark, resume_text, target_category, education_ordinals, level_order)
    except Exception as exc:  # noqa: BLE001 - keep the other tabs usable
        st.error(f"The resume fit tab could not be rendered ({type(exc).__name__}).")
