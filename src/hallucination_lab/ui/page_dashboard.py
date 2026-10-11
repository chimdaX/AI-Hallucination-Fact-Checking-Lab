"""Dashboard: totals computed from saved session files."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from hallucination_lab.services.dashboard import PROVENANCE_LABELS, compute_dashboard
from hallucination_lab.ui import common


@common.safe_page
def render() -> None:
    st.title("Dashboard")
    st.caption(
        "All numbers are computed from the session files in "
        f"`{common.workspace().settings.sessions_dir}`. Active verdicts only: "
        "superseded revisions are kept for audit but not counted."
    )
    stats = compute_dashboard(common.workspace().store)
    for problem in stats.unreadable:
        st.warning(f"Unreadable session file skipped: {problem}")
    if stats.empty:
        st.info(
            "No sessions yet. Open **Analyze response** to paste an AI response, or run "
            "`hallab demo --save` to load the illustrative fixtures. **Getting started** "
            "in the sidebar explains the workflow."
        )
        return

    t = stats.totals
    row1 = st.columns(5)
    row1[0].metric("Sessions", t.sessions)
    row1[1].metric("Candidate claims", t.candidate_claims)
    row1[2].metric("Flagged for verification", t.flagged)
    row1[3].metric("Human-verified claims", t.verified)
    row1[4].metric("Pending review", t.pending)
    row2 = st.columns(5)
    row2[0].metric("Supported", t.verdicts["SUPPORTED"])
    row2[1].metric("Contradicted", t.verdicts["CONTRADICTED"])
    row2[2].metric("Insufficient evidence", t.verdicts["INSUFFICIENT_EVIDENCE"])
    row2[3].metric("Not checkable", t.verdicts["NOT_CHECKABLE"])
    row2[4].metric(
        "Revisions",
        t.revisions,
        help="Correction records. The claims they "
        "correct are counted once, under their latest verdict.",
    )
    st.caption(
        "These counts describe the claims reviewed here. They are not an accuracy "
        "metric or an LLM hallucination rate: the responses are not a representative "
        "sample, and verdicts depend on the reference documents used."
    )

    st.subheader("Where the data comes from")
    rows = [
        {
            "Data": PROVENANCE_LABELS[key],
            "Sessions": tot.sessions,
            "Candidate claims": tot.candidate_claims,
            "Human-verified": tot.verified,
            "Pending": tot.pending,
        }
        for key, tot in sorted(stats.by_provenance.items())
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if "synthetic_fixture" in stats.by_provenance:
        st.caption(
            "Illustrative fixtures are hand-written examples, not real model outputs. "
            "Their verdicts are practice judgements on demonstration data."
        )
    if t.legacy_records:
        st.caption(
            f"{t.legacy_records} active verdict(s) are legacy records created before "
            "per-passage evidence assessments existed."
        )

    left, right = st.columns(2)
    with left:
        common.bar_chart(
            {k.replace("_", " ").title(): v for k, v in t.verdicts.items()}, "Active human verdicts"
        )
    with right:
        common.bar_chart(
            {
                "Human-verified": t.verified,
                "Pending review": t.pending,
                "Not flagged": t.candidate_claims - t.flagged,
            },
            "Claims by status",
        )

    st.subheader("Sessions")
    st.dataframe(pd.DataFrame(stats.sessions), hide_index=True, width="stretch")
