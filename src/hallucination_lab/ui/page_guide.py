"""Getting started: what the tool does and an example workflow."""

from __future__ import annotations

import streamlit as st

from hallucination_lab.ui import common

EXAMPLE = (
    "Pi is approximately 3.14159. The Eiffel Tower was completed in 1899. "
    "An unknown research group published a revolutionary energy discovery in 2040."
)


@common.safe_page
def render() -> None:
    st.title("Getting started")
    st.markdown(
        """
This lab checks **individual factual claims** in AI-generated text against **trusted
reference documents**, and records **human** verdicts with cited evidence.

| Step | Page | What happens | Who decides |
|---|---|---|---|
| 1 | Analyze response | Rule-based extraction proposes candidate claims | Automated, editable |
| 2 | Analyze response | BM25 finds passages **sharing words** with claims | Automated, not truth |
| 3 | Human verification | Label passages, choose a verdict, cite evidence, explain | **You** |
| 4 | Reports | Download a traceable JSON / CSV / Markdown report | Generated from saved records |

**Important**
- Retrieved passages are *candidates*. A high BM25 score means word overlap, not support.
- Missing evidence leads to **INSUFFICIENT_EVIDENCE**, never CONTRADICTED.
- Saved verdicts are never edited. Corrections become new, linked revision records.
- Everything runs locally and offline. Data is stored as JSON in the folder shown in the
  sidebar.
"""
    )
    st.subheader("Try the example")
    st.markdown(
        "Open **Analyze response**, paste the text below, tick *illustrative example*, "
        "extract claims, retrieve from the `demo` collection, then verify each claim."
    )
    st.code(EXAMPLE, language=None)
    st.markdown(
        "Full decision criteria: `docs/HUMAN_VERIFICATION_GUIDE.md`. "
        "Retrieval details: `docs/RETRIEVAL.md`."
    )
