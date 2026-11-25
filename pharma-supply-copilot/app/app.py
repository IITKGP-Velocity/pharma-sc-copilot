"""Streamlit front end with a per-session rate limit and answer caching.

  python -m streamlit run app/app.py
"""
import json
import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from copilot.config import load_settings  # noqa: E402
from copilot.service import build_engine  # noqa: E402

RATE_LIMIT_PER_MIN = int(load_settings().get("rate_limit_per_min", 10))


@st.cache_resource
def engine():
    return build_engine()


@st.cache_data(show_spinner=False)
def cached_answer(question: str) -> dict:
    return engine().answer(question).model_dump()


def allowed_now() -> bool:
    """Sliding window: at most RATE_LIMIT_PER_MIN questions per session per minute."""
    now = time.time()
    stamps = [t for t in st.session_state.get("asks", []) if now - t < 60]
    if len(stamps) >= RATE_LIMIT_PER_MIN:
        st.session_state["asks"] = stamps
        return False
    stamps.append(now)
    st.session_state["asks"] = stamps
    return True


st.set_page_config(page_title="Pharma Supply Chain Copilot", layout="wide")
st.title("Pharmaceutical Supply Chain Copilot")
st.caption("Answers cite sources. Release, disposition, and approval decisions are escalated to a human.")
st.info(f"Mode: {engine().mode}")

question = st.text_area("Your question", placeholder="e.g. Logger shows 12 h at 9 °C. Can we release the batch?")
if st.button("Ask", type="primary"):
    if not question.strip():
        st.warning("Type a question first.")
    elif not allowed_now():
        st.error(f"Limit reached: {RATE_LIMIT_PER_MIN} questions per minute. Wait a moment and try again.")
    else:
        result = cached_answer(question.strip())
        st.subheader(f"Decision: {result['decision']}")
        if result.get("escalate_to"):
            st.warning(f"Route to: {result['escalate_to']}")
        if result.get("missing_evidence"):
            st.info(result["missing_evidence"])
        st.write(result.get("answer") or "_No answer text._")
        with st.expander("Details (citations, tool results, raw output)"):
            st.json(result)
