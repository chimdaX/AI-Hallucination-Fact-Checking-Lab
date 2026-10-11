"""Reports: session findings, charts, and JSON / CSV / Markdown downloads."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from hallucination_lab.services.dashboard import compute_dashboard
from hallucination_lab.services.reports import (
    MEDIA_TYPES,
    REVIEW_STATUS_TEXT,
    build_report,
    export,
    report_filename,
)
from hallucination_lab.ui import common


def _session_report(ws, loaded) -> None:
    session = loaded.session
    report = build_report(session, ws.evidence_checks(session))
    s = report["summary"]
    st.markdown(
        f"**{session.title}** · {common.provenance_badge(session)} · `{session.session_id}`"
    )
    if s["complete"]:
        st.success("Complete: every claim flagged for verification has an active human verdict.")
    else:
        st.warning(
            f"Incomplete: {s['awaiting_human_review']} flagged claim(s) still await "
            "human review. The report lists them as unresolved."
        )
    cols = st.columns(4)
    cols[0].metric("Human-verified", s["verified_by_human"])
    cols[1].metric("Awaiting review", s["awaiting_human_review"])
    cols[2].metric("Revisions", s["revisions"])
    cols[3].metric("Integrity warnings", len(report["integrity_warnings"]))
    st.caption(s["note"])

    left, right = st.columns(2)
    with left:
        common.bar_chart(
            {k.replace("_", " ").title(): v for k, v in s["active_verdicts"].items()},
            "Active verdicts in this session",
        )
    with right:
        statuses: dict[str, int] = {}
        for c in report["claims"]:
            label = REVIEW_STATUS_TEXT[c["review_status"]]
            statuses[label] = statuses.get(label, 0) + 1
        common.bar_chart(statuses, "Claims by review status")
    candidates = sum(len(c["retrieval"]["candidates"]) for c in report["claims"])
    assessed = sum(
        sum(x["reviewer_assessment"] != "not assessed" for x in c["retrieval"]["candidates"])
        for c in report["claims"]
    )
    cited = sum(
        len(c["human_verdict"]["cited_evidence"]) for c in report["claims"] if c["human_verdict"]
    )
    common.bar_chart(
        {
            "Retrieved (candidates)": candidates,
            "Assessed by a reviewer": assessed,
            "Cited in an active verdict": cited,
        },
        "Evidence usage (latest retrieval per claim)",
        "Passages",
    )
    st.caption(
        "Retrieved passages are lexical matches. Only assessed and cited passages "
        "reflect a reviewer's judgement."
    )

    st.subheader("Download")
    st.caption(
        f"Content fingerprint `{report['content_sha256'][:16]}…` is identical for "
        "repeated exports of unchanged data (it excludes the generation time)."
    )
    cols = st.columns(3)
    for col, fmt in zip(cols, ("json", "csv", "md"), strict=True):
        col.download_button(
            f"Download {fmt.upper()}",
            export(report, fmt),
            file_name=report_filename(report, fmt),
            mime=MEDIA_TYPES[fmt],
            key=f"dl_{fmt}",
            width="stretch",
        )

    if report["unresolved_claims"]:
        st.subheader("Unresolved claims")
        st.dataframe(pd.DataFrame(report["unresolved_claims"]), hide_index=True, width="stretch")
    if report["integrity_warnings"]:
        st.subheader("Document integrity warnings")
        st.dataframe(pd.DataFrame(report["integrity_warnings"]), hide_index=True, width="stretch")
    with st.expander("Preview the Markdown report"):
        st.markdown(export(report, "md"))
    with st.expander("Methodological limitations"):
        for item in report["limitations"]:
            st.markdown(f"- {item}")


@common.safe_page
def render() -> None:
    st.title("Reports")
    ws = common.workspace()
    loaded = common.load_current(ws)
    if loaded is None:
        st.info(
            "Open a session on **Analyze response** to see its findings and download a "
            "report. Saved sessions are summarised below."
        )
    else:
        _session_report(ws, loaded)
        common.mark_seen(loaded)

    st.divider()
    st.subheader("All sessions")
    stats = compute_dashboard(ws.store)
    if stats.empty:
        st.caption("No saved sessions.")
        return
    st.dataframe(pd.DataFrame(stats.sessions), hide_index=True, width="stretch")
    st.caption(
        "Per-session counts of active human verdicts. They are not comparable as rates: "
        "sessions differ in topic, claims and reference documents."
    )
