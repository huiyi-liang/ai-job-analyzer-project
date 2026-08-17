"""Tab 5: adaptive AI read of the market, and the fit when a resume has been scored."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from utils.llm_agent import (
    SETUP_MESSAGE,
    LLMUnavailable,
    build_context,
    chat,
    generate_summary,
    is_configured,
)

MARKET_QUESTIONS = (
    "Which skills pay the most right now?",
    "Which roles are growing fastest?",
    "Which job titles have the highest demand?",
)
FIT_QUESTIONS = (
    "What should I learn next?",
    "Which of my skills should I demonstrate better?",
    "Am I realistically competitive for this role?",
)


def _summary_key(category: str, scored: bool) -> str:
    """Build a cache key so the summary refreshes when the filter or score changes."""
    return f"{category}|{'scored' if scored else 'unscored'}"


def _render_summary(context: dict[str, Any], key: str) -> None:
    """Render the auto-summary, generating it once per filter and score state."""
    cached = st.session_state.get("ai_summary")
    if not cached or cached.get("key") != key:
        with st.spinner("Reading the market…"):
            try:
                text = generate_summary(context)
                st.session_state["ai_summary"] = {"key": key, "text": text, "error": None}
            except LLMUnavailable as exc:
                st.session_state["ai_summary"] = {"key": key, "text": None, "error": str(exc)}
        cached = st.session_state["ai_summary"]

    if cached.get("error"):
        st.error(f"Could not generate the summary — {cached['error']}")
        if st.button("Retry summary", key="retry_summary"):
            st.session_state.pop("ai_summary", None)
            st.rerun()
        return
    st.markdown(cached["text"])


def _ask(context: dict[str, Any], question: str) -> None:
    """Send a question to the model and append both turns to the chat history."""
    history = st.session_state.get("chat_history", [])
    with st.spinner("Thinking…"):
        try:
            answer = chat(context, history, question)
        except LLMUnavailable as exc:
            answer = f"⚠️ {exc}"
    st.session_state["chat_history"] = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]


def _render(
    df_filtered: pd.DataFrame,
    category: str,
    rows_2026: int,
    total_rows: int,
    level_order: list[str],
    benchmark: Any,
    fit_result: dict[str, Any] | None,
) -> None:
    """Render the full AI read tab."""
    scored = fit_result is not None
    st.caption(
        f"Scope: 2026 postings only · category **{category}** · {len(df_filtered):,} postings"
        + (
            f" · fit scored against {benchmark.title} ({benchmark.level})"
            if scored
            else " · no resume scored yet"
        )
    )

    if not is_configured():
        st.info(f"🔑 **AI read unavailable.** {SETUP_MESSAGE}")
        st.caption("Tabs 1–3 and the Step 0 benchmark in Tab 4 work fully without a key.")
        return

    context = build_context(
        df_filtered, category, rows_2026, total_rows, level_order,
        benchmark=benchmark if scored else None,
        fit_result=fit_result,
    )

    st.subheader("The read")
    _render_summary(context, _summary_key(category, scored))
    if not scored:
        st.caption(
            "Score a resume in **Your Resume Fit** to add a personalized read of how you "
            "compare against a target role."
        )

    st.divider()
    st.subheader("Ask a question")

    st.caption("**About the market**")
    for index, question in enumerate(MARKET_QUESTIONS):
        if st.button(question, key=f"market_q_{index}", width="stretch"):
            _ask(context, question)
            st.rerun()

    if scored:
        st.caption("**About my fit**")
        for index, question in enumerate(FIT_QUESTIONS):
            if st.button(question, key=f"fit_q_{index}", width="stretch"):
                _ask(context, question)
                st.rerun()

    for turn in st.session_state.get("chat_history", []):
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    typed = st.chat_input("Ask about this data…")
    if typed:
        _ask(context, typed)
        st.rerun()

    if st.session_state.get("chat_history"):
        if st.button("Clear conversation", key="clear_chat"):
            st.session_state["chat_history"] = []
            st.rerun()

    st.caption(
        "Answers are grounded in aggregate figures from this dataset only — no raw "
        "postings and no resume text are sent. One snapshot of 2026 postings; not "
        "comprehensive labor market data, and not career or hiring advice."
    )


def render(
    df_filtered: pd.DataFrame,
    category: str,
    rows_2026: int,
    total_rows: int,
    level_order: list[str],
    benchmark: Any = None,
    fit_result: dict[str, Any] | None = None,
) -> None:
    """Render the AI read tab, isolating any failure to this tab."""
    try:
        _render(
            df_filtered, category, rows_2026, total_rows, level_order, benchmark, fit_result
        )
    except Exception as exc:  # noqa: BLE001 - keep the other tabs usable
        st.error(f"The AI read could not be rendered ({type(exc).__name__}).")
