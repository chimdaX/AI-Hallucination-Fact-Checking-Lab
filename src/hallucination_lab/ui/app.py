"""Streamlit entry point: page setup, sidebar and navigation."""

from __future__ import annotations

import streamlit as st

from hallucination_lab import __version__
from hallucination_lab.logging_utils import configure_logging
from hallucination_lab.ui import (
    common,
    page_analyze,
    page_benchmark,
    page_collections,
    page_dashboard,
    page_guide,
    page_reports,
    page_verify,
)


def _sidebar() -> None:
    ws = common.workspace()
    st.sidebar.markdown("### Fact-Checking Lab")
    st.sidebar.caption(f"v{__version__} · local and offline")
    st.sidebar.text_input(
        "Reviewer ID",
        key=common.REVIEWER_KEY,
        help="Recorded on every edit and verdict. Self-declared, not authenticated.",
        placeholder="e.g. reviewer_01",
    )
    summaries, problems = ws.list_sessions()
    current = common.current_session_id()
    ids = [s.session_id for s in summaries]
    labels = {s.session_id: s.title[:40] for s in summaries}
    if current and current not in ids:
        common.open_session(None)
        current = None
    choice = st.sidebar.selectbox(
        "Current session",
        [None, *ids],
        index=([None, *ids].index(current)),
        format_func=lambda sid: "No session open" if sid is None else f"{labels[sid]} ({sid})",
    )
    if choice != current:
        common.open_session(choice)
        st.rerun()
    if problems:
        st.sidebar.warning(
            f"{len(problems)} unreadable session file(s) skipped; see the dashboard."
        )
    st.sidebar.caption(
        f"Data folder: `{ws.settings.data_dir.resolve()}` (set HALLAB_DATA_DIR to change it)"
    )


def main() -> None:
    st.set_page_config(
        page_title="Fact-Checking Lab", page_icon=":material/fact_check:", layout="wide"
    )
    configure_logging()
    pages = [
        st.Page(
            page_dashboard.render,
            title="Dashboard",
            icon=":material/dashboard:",
            url_path="dashboard",
            default=True,
        ),
        st.Page(
            page_analyze.render,
            title="Analyze response",
            icon=":material/edit_note:",
            url_path="analyze",
        ),
        st.Page(
            page_verify.render,
            title="Human verification",
            icon=":material/fact_check:",
            url_path="verify",
        ),
        st.Page(
            page_reports.render, title="Reports", icon=":material/description:", url_path="reports"
        ),
        st.Page(
            page_collections.render,
            title="Reference collections",
            icon=":material/library_books:",
            url_path="collections",
        ),
        st.Page(
            page_benchmark.render, title="Benchmark", icon=":material/rule:", url_path="benchmark"
        ),
        st.Page(
            page_guide.render, title="Getting started", icon=":material/help:", url_path="start"
        ),
    ]
    navigation = st.navigation(pages)
    _sidebar()
    navigation.run()
