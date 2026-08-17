"""Resume text extraction (pdfplumber) and LLM-based structured profile extraction.

The model here only turns unstructured resume text into structured fields. It never
produces or adjusts a score — all scoring lives in utils/matching.py.
"""

from __future__ import annotations

import json
import re
from typing import Any, BinaryIO

from utils.data_loader import EDUCATION_LADDER
from utils.llm_agent import EXTRACTION_MAX_TOKENS, LLMUnavailable, complete
from utils.matching import EVIDENCE_CHOICES, EVIDENCE_LISTED_ONLY, EVIDENCE_NONE

# --- Constants ---------------------------------------------------------------

# Below this many characters a PDF has almost certainly been scanned as images.
MIN_USABLE_TEXT_CHARS = 200
MAX_RESUME_CHARS = 18_000

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


class ResumeParseError(ValueError):
    """Raised when resume text or model output cannot be turned into a profile."""


# --- PDF extraction ----------------------------------------------------------


def extract_pdf_text(file: BinaryIO) -> str:
    """Extract text from a PDF resume using pdfplumber."""
    try:
        import pdfplumber
    except ImportError as exc:
        raise ResumeParseError("The 'pdfplumber' package is not installed.") from exc

    try:
        with pdfplumber.open(file) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
    except Exception as exc:  # noqa: BLE001 - surfaced as a friendly message
        raise ResumeParseError(f"Could not read the PDF ({type(exc).__name__}).") from exc

    return "\n".join(pages).strip()


def looks_unextractable(text: str) -> bool:
    """Report whether extracted PDF text is too sparse to be a real resume."""
    return len(text.strip()) < MIN_USABLE_TEXT_CHARS


# --- LLM extraction ----------------------------------------------------------

EXTRACTION_SYSTEM_PROMPT = """You extract structured facts from a resume. You never
score, rank, or judge the candidate, and you never output numbers other than years of
experience. Return ONLY valid JSON — no preamble, no explanation, no markdown fences."""


def build_extraction_prompt(
    resume_text: str,
    target_title: str,
    target_category: str,
    benchmark_skills: list[str],
) -> str:
    """Build the extraction prompt, giving the model the target role as relevance context."""
    skill_list = ", ".join(benchmark_skills) if benchmark_skills else "(none provided)"
    return f"""TARGET ROLE CONTEXT (use this to judge what counts as related):
- Target job title: {target_title}
- Job category: {target_category}
- Skills required by postings for this role: {skill_list}

EXPERIENCE RULES:
- "total_years_experience" = ALL professional experience, in whole years.
- "related_years_experience" = ONLY experience relevant to the target role above.
  This is always less than or equal to total_years_experience. A career changer may
  legitimately have 0 related years.
- "related_experience_reasoning" = ONE sentence stating what you counted as related
  and what you excluded, so the reader can judge and correct you.

SKILL RULES:
- Consider ONLY the skills in the target skill list above. Do not add skills that are
  not on that list, however impressive they are.
- Return one entry for every skill on that list.

EVIDENCE RULES:
- "demonstrated" = the skill appears in a bullet or sentence describing actual work —
  a project, responsibility, or accomplishment.
  Example: "Built a RAG pipeline using vector databases and Python."
- "listed_only" = the skill appears ONLY in a skills section, or is named with no
  description of how it was used.
- "none" = the skill is neither named nor demonstrated anywhere in the resume. Set
  "source" to null for these.

EDUCATION:
- "education_level" must be exactly one of: {" | ".join(EDUCATION_LADDER)}
- Choose the HIGHEST level the resume evidences.

Return JSON in exactly this shape:
{{
  "total_years_experience": 0,
  "related_years_experience": 0,
  "related_experience_reasoning": "",
  "education_level": "Bachelor's",
  "skills": [
    {{"skill": "Python", "evidence": "demonstrated", "source": "brief paraphrase"}},
    {{"skill": "Tableau", "evidence": "listed_only", "source": "Skills section"}},
    {{"skill": "Spark", "evidence": "none", "source": null}}
  ]
}}

RESUME TEXT:
\"\"\"
{resume_text[:MAX_RESUME_CHARS]}
\"\"\"
"""


def strip_code_fences(text: str) -> str:
    """Remove markdown code fences and any prose around a JSON object."""
    cleaned = _FENCE_RE.sub("", text.strip())
    # Fall back to the outermost brace pair when the model adds a preamble anyway.
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start : end + 1]
    return cleaned


def _coerce_years(value: Any) -> int:
    """Coerce a years value to a non-negative integer, defaulting to 0."""
    try:
        return max(0, int(round(float(value))))
    except (TypeError, ValueError):
        return 0


def parse_extraction_json(raw: str, allowed_skills: list[str] | None = None) -> dict[str, Any]:
    """Parse and normalize the model's JSON output, raising ResumeParseError on failure."""
    payload = strip_code_fences(raw)
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ResumeParseError(
            f"The model did not return valid JSON ({exc.msg})."
        ) from exc

    if not isinstance(data, dict):
        raise ResumeParseError("The model returned JSON that was not an object.")

    total = _coerce_years(data.get("total_years_experience"))
    related = _coerce_years(data.get("related_years_experience"))
    # Related experience is a subset of total by definition; enforce it rather than
    # trusting the model's arithmetic.
    related = min(related, total)

    education = data.get("education_level")
    if education not in EDUCATION_LADDER:
        education = None

    allowed = {s.casefold(): s for s in (allowed_skills or [])}
    skills: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in data.get("skills") or []:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("skill") or "").strip()
        if not name:
            continue
        # Snap to the benchmark's spelling and drop skills outside the target list.
        if allowed:
            canonical = allowed.get(name.casefold())
            if canonical is None:
                continue
            name = canonical
        if name.casefold() in seen:
            continue
        seen.add(name.casefold())

        evidence = str(entry.get("evidence") or EVIDENCE_NONE).strip().lower()
        if evidence not in EVIDENCE_CHOICES:
            evidence = EVIDENCE_NONE
        source = entry.get("source")
        skills.append(
            {
                "skill": name,
                "evidence": evidence,
                "source": None if evidence == EVIDENCE_NONE else (source or None),
            }
        )

    # Any benchmark skill the model omitted is treated as absent, so the editable
    # step always shows the full target skill list.
    for skill in allowed_skills or []:
        if skill.casefold() not in seen:
            skills.append({"skill": skill, "evidence": EVIDENCE_NONE, "source": None})

    return {
        "total_years_experience": total,
        "related_years_experience": related,
        "related_experience_reasoning": str(
            data.get("related_experience_reasoning") or ""
        ).strip(),
        "education_level": education,
        "skills": skills,
    }


def extract_profile(
    resume_text: str,
    target_title: str,
    target_category: str,
    benchmark_skills: list[str],
) -> dict[str, Any]:
    """Extract a structured resume profile via Groq, raising on LLM or parse failure."""
    if not resume_text or not resume_text.strip():
        raise ResumeParseError("The resume text is empty.")

    prompt = build_extraction_prompt(
        resume_text, target_title, target_category, benchmark_skills
    )
    raw = complete(
        [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=EXTRACTION_MAX_TOKENS,
    )
    return parse_extraction_json(raw, allowed_skills=benchmark_skills)


def new_skill_entry(skill: str) -> dict[str, Any]:
    """Build a skill entry for a user-added skill, defaulting to listed_only."""
    return {"skill": skill.strip(), "evidence": EVIDENCE_LISTED_ONLY, "source": "Added by you"}


__all__ = [
    "LLMUnavailable",
    "ResumeParseError",
    "extract_pdf_text",
    "extract_profile",
    "looks_unextractable",
    "new_skill_entry",
    "parse_extraction_json",
    "strip_code_fences",
]
