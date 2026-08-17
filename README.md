# AI Job Landscape & Fit Analyzer

A Streamlit app for exploring the 2026 AI job market from a CSV dataset, and — optionally —
benchmarking a resume against one target role.

The app is built around a single discipline: **every number tells you what it rests on.** Each
metric, chart, and table shows the posting count behind it, rare skills are excluded from
salary charts rather than presented as insight, and the resume score is deterministic Python
that a language model never touches.

**Live app:** [AI-job-landscape-fit-analyzer.streamlit.app](https://huiyi-liang-ai-job-analyzer-project-app-4xaklh.streamlit.app/)

---

## Features

| Tab | What it does |
|---|---|
| **1. Landscape Overview** | Five headline metrics, salary by experience level with typical ranges, top-10 job titles (median *and* average salary), side-by-side company size and industry composition charts, and geography tables. |
| **2. Skills Landscape** | Exactly two charts: skill frequency, and a skill × salary/growth scatter with quadrant lines and a minimum-posting threshold. |
| **3. Demand & Trends** | Your filtered view, the always-unfiltered full 2026 landscape for comparison, and the one chart that spans 2025–2026. |
| **4. Your Resume Fit** | An aggregate benchmark for your target role, AI-assisted resume extraction, an editable confirmation step, and a deterministic fit score with a full skill breakdown. |
| **5. AI Market & Fit Read** | An adaptive summary and grounded chat — useful with or without a scored resume. |

---

## Setup (uv only)

```bash
uv sync                      # install dependencies from uv.lock
cp .env.example .env         # then paste your key into .env
uv run streamlit run app.py
```

To recreate the environment from scratch:

```bash
uv init
uv add streamlit pandas numpy plotly groq python-dotenv pdfplumber
```

### Getting a free Groq API key

1. Sign up at <https://console.groq.com>.
2. Open **API Keys** → **Create API Key** (<https://console.groq.com/keys>).
3. Paste it into `.env` as `GROQ_API_KEY=...` and restart the app.

`GROQ_MODEL` is optional and defaults to `llama-3.1-8b-instant`.

**Without a key the app still works:** Tabs 1–3 are fully functional, Tab 4 renders its
complete Step 0 benchmark, and Tab 4's resume steps and Tab 5 show a setup message. No feature
fails with an error just because a key is missing.

---

## Folder structure

```
AI-job-analyzer/
├── app.py                  # page config, session state, sidebar, tabs
├── pyproject.toml          # requires-python = ">=3.11"
├── uv.lock
├── README.md
├── .env.example
├── .gitignore
├── data/
│   └── ai_jobs_market_2025_2026.csv
├── utils/                  # all data logic
│   ├── data_loader.py      # load, validate, apply the 2026 filter once, build ordinals
│   ├── skills_analysis.py  # explode, threshold, per-skill aggregation with counts
│   ├── resume_parser.py    # pdfplumber, LLM extraction, defensive JSON parsing
│   ├── matching.py         # benchmark, tiering, meets_bar, scoring — pure Python
│   └── llm_agent.py        # Groq client, context assembly, summary, chat
└── components/             # UI only, no data logic
    ├── sidebar.py
    ├── landscape_tab.py
    ├── skills_tab.py
    ├── demand_trends_tab.py
    ├── resume_fit_tab.py
    └── ai_read_tab.py
```

---

## Dataset

**1,500 job postings** spanning 2025–2026 — 25 AI/ML roles, 14 countries, 12 industries, and
5 company sizes. Of these, **876 rows are 2026 postings**, which is the app's working scope.

### Schema

`job_id`, `job_title`, `job_category`, `experience_level`, `years_of_experience`,
`education_required`, `annual_salary_usd`, `salary_min_usd`, `salary_max_usd`, `city`,
`country`, `remote_work`, `company_size`, `industry`, `required_skills`,
`ai_salary_premium_pct`, `demand_score`, `demand_growth_yoy_pct`, `benefits_score_10`,
`posting_year`, `posting_month`, `is_senior`, `is_remote_friendly`, `is_llm_role`,
`salary_tier`

`benefits_score_10` and `salary_tier` are intentionally unused; no feature is built on them.

### Uploading your own CSV

The uploader accepts the same schema and validates that all columns are present,
`required_skills` is parseable, `annual_salary_usd` is numeric, `posting_year` is present, and
at least one 2026 row exists. **Failures name exactly what went wrong and the app keeps the
bundled dataset**, telling you it did so. The active dataset is always shown in the sidebar.

Sample row:

```csv
job_id,job_title,job_category,experience_level,years_of_experience,education_required,annual_salary_usd,salary_min_usd,salary_max_usd,city,country,remote_work,company_size,industry,required_skills,ai_salary_premium_pct,demand_score,demand_growth_yoy_pct,benefits_score_10,posting_year,posting_month,is_senior,is_remote_friendly,is_llm_role,salary_tier
AIJOB0001,AI Agent Developer,AI Engineering,Senior (6-9 yrs),7,Master's,239000.0,155000,290000,Boston,USA,On-site,Startup (1-50),Finance,APIs|Planning Systems|Python|Cloud|SQL|Leadership,13.1,96,16.9,6.8,2026,3,1,0,1,Senior ($200-300k)
```

---

## Design decisions, and why

### The app-wide 2026 scope

Every tab operates on `posting_year == 2026`, filtered **once at load** immediately after
validation. `demand_score` and `demand_growth_yoy_pct` are point-in-time measures: averaging a
2025 posting's demand score with a 2026 posting's produces a figure describing neither year. A
single consistent scope also means every number in the app reconciles with every other one.

The sidebar states this permanently: *"2026 postings: 876 of 1500 total rows."*

**The one exception** is Tab 3 Section C, the "Over Time" chart, which re-sources from the full
dataset to show the 2025→2026 change. It is labeled 2025–2026 and carries a warning that it is
the only view in the app using 2025 data.

### Why the two sidebar sections are independent

**Section 1 — Explore the Landscape** (job category) controls Tabs 1–3.
**Section 2 — Resume Fit** (target title, target level) controls Tab 4 only.

They are never cross-applied. Your benchmark should not silently change because you were
browsing a different slice of the market, and the market view should not narrow to your target
role because you uploaded a resume. Filtering the landscape to "Data Science" while
benchmarking against an LLM Engineer role is a legitimate thing to want.

### The minimum-posting threshold

`MIN_POSTINGS_PER_SKILL = max(10, 2% of filtered rows)` — 18 postings at the default
"All categories" view. Skills below it are excluded from the salary/growth scatter, though they
still appear in the frequency chart.

A skill appearing in two or three postings can produce a median salary that tops the chart
purely by chance, presenting noise as insight. The caption always states the threshold and
names the excluded skills, so you know exclusion happened rather than wondering whether a
missing skill is a bug. (At "All categories" exactly three skills are excluded: CI/CD,
Enterprise Architecture, and Risk Assessment. Narrow the category filter and the threshold
starts excluding much more.)

### The skill-growth proxy caveat

On the scatter's Y-axis, `demand_growth_yoy_pct` is a per-posting attribute tied to the
**role**, not the skill. The axis shows the average growth of roles that require a skill — a
**proxy, not a measurement**. It does not measure growth in demand for the skill itself.

### Why Tab 4 benchmarks against one aggregate profile

Tab 4 builds **one** aggregate profile from all 2026 postings matching your target job title
and experience level, rather than ranking you against individual postings. Individual postings
are noisy — one unusual listing should not become "the bar" — and a ranked list of postings
would invite reading a synthetic dataset as though it contained real openings to apply to. One
aggregate profile answers the actual question: what does this role typically ask for?

If fewer than 10 postings back the benchmark, the app warns prominently. This is common: **61
of the 100 title × level combinations in the bundled data have fewer than 10 postings.** The
sidebar therefore shows the posting count next to every option and defaults to the
best-supported combination.

### Scoring is deterministic Python

**The LLM only extracts structured data from resume text. It never produces or adjusts a
number.** Benchmark construction, comparison, and scoring are pure pandas and arithmetic in
`utils/matching.py`, which imports no LLM code at all. The same resume against the same target
always produces the same score.

The scoring pipeline:

1. **`meets_bar`** = `meets_experience` **and** `meets_education`, where experience uses
   **related** years against the benchmark's median `years_of_experience`, and education
   compares ordinal ranks.
2. **Frequency-weighted skill credit** — a skill required in 90% of postings matters more than
   one required in 25%:
   `skills_score = Σ(credit × frequency) ÷ Σ(frequency)`
   - `demonstrated` → 1.0
   - `listed_only` → 0.5 if `meets_bar`, otherwise **0.0**
   - missing → 0.0
3. **A bounded modifier** — `final_score = skills_score × (1 + modifier)`, where the modifier is
   clamped to ±0.15:
   ```
   exp_d    = clamp((related_years − benchmark_median_years) / 5, −1, 1)
   edu_d    = clamp((candidate_ordinal − benchmark_ordinal) / 4, −1, 1)
   modifier = 0.15 × (exp_d + edu_d) / 2
   ```
   Experience distance is measured in **absolute years, never as a ratio**, so a career changer
   with 0 related years cannot cause a division error. Because the modifier is bounded, it can
   never flip the ranking of two skill profiles that differ by more than 0.15.
4. **Clamped to 0–100%.**

Skill matching is case-insensitive and whitespace-normalized.

### Demonstrated vs. listed-only, and why below-bar candidates are scored more strictly

- **`demonstrated`** — the skill appears in a bullet or sentence describing actual work: a
  project, a responsibility, an accomplishment. *"Built a RAG pipeline using vector databases
  and Python."*
- **`listed_only`** — the skill appears only in a skills section, or is named with no
  description of how it was used.
- **`none`** — the skill is neither named nor demonstrated.

If you are below the experience or education bar, listed-only skills earn **zero** credit. A
candidate who already clears the bar has independent evidence of competence, so a skills-section
mention is reasonable corroboration. A candidate below the bar does not, so a bare list is not
enough to compensate for a credentials gap — the skill must be **demonstrated**.

The skill breakdown table distinguishes *"have it, but not demonstrated"* from *"missing
entirely"*, because the first is the far cheaper gap to close: it needs evidence of use written
down, not a new skill learned.

### Related vs. total years

Extraction returns **both** total and related years plus one sentence of reasoning explaining
what was counted and what was excluded, so you can judge and correct it. **Only related years
are scored.** Total years are displayed alongside, explicitly labeled as not used for scoring.
Twenty years in a different field should not read as twenty years of relevant preparation.

### Two dataset contradictions, and how the app resolves them

- **`years_of_experience` can contradict `experience_level`** — a real row is labeled
  "Senior (6-9 yrs)" with 2 years. The app treats `years_of_experience` as the authoritative
  numeric value and `experience_level` purely as a grouping label.
- **`annual_salary_usd` can fall outside `salary_min_usd`–`salary_max_usd`** — 288 rows do.
  The app uses `annual_salary_usd` as the single salary figure everywhere, so all salary
  statistics reconcile.

Duplicate skills within one `required_skills` value (119 rows have them) are deduplicated
per posting before any counting.

---

## Smoke test

1. **Load with "All categories."** All tabs render. Tab 1 shows five metric cards, the
   salary-by-experience-level table with typical ranges, two side-by-side horizontal bar
   charts (company size, industry), and a top-10 title table whose median column reconciles
   with the median metric card.
2. **Select "AI Engineering."** Tabs 1–3 update; Tab 3 Section B still shows the full
   unfiltered 2026 landscape.
3. **Tab 2.** Exactly two charts, with both the threshold caption and the skill-growth-proxy
   caveat visible.
4. **Tab 3 Section C.** The dual-axis chart renders two lines — posting count (left axis,
   solid) and median salary (right axis, dashed) — labeled 2025–2026.
5. **Tab 4 with no resume.** The Step 0 benchmark renders in full, including the tiered skill
   table.
6. **Paste a resume** with one demonstrated skill ("Built a pipeline in Python") and one
   listed-only skill (a bare "SQL"). The related-vs-total years split and the evidence
   classification appear in the editable step, and flipping a skill to *demonstrated* changes
   the fit score.
7. **Score the same resume twice unchanged.** Identical score, confirming determinism.
8. **Change the target experience level after scoring.** The old score disappears and a
   re-score prompt appears, with your resume still loaded.

---

## Disclaimer

This app reads **one dataset snapshot of 2026 job postings**. It is not comprehensive labor
market data, and it is not career or hiring advice. The fit score is a structured comparison
against one aggregate profile in this dataset — not an assessment of your employability, and
not a prediction of any hiring outcome.
