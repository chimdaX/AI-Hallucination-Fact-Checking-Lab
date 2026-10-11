"""Shared helpers for the Streamlit pages: state, error handling, components.

State rules (Streamlit reruns the script on every interaction):

* **Saved data lives only in session files.** Each run reloads the current
  session from disk, so pages never show stale data. ``st.session_state``
  holds only interface state: the selected session ID, the reviewer ID, the
  file version the user last *saw*, and a pending verification awaiting
  confirmation.
* **Saving needs a fresh view.** Before any write, :func:`require_fresh`
  compares the file version now on disk with the one the user saw on the
  previous run. If another tab or the CLI changed the session in between, or
  a double click replays an action, the write is refused and the page reloads.
  The store's own ``expected_version`` check and the schema's one-active-verdict
  rule are further layers underneath.
"""

from __future__ import annotations

import functools
import logging
import re
from collections.abc import Callable

import altair as alt
import pandas as pd
import streamlit as st
from pydantic import ValidationError

from hallucination_lab.errors import HallucinationLabError
from hallucination_lab.models import Evidence, ResponseProvenance, VerificationSession
from hallucination_lab.services.dashboard import PROVENANCE_LABELS
from hallucination_lab.services.workspace import LoadedSession, Workspace
from hallucination_lab.storage import summarize_validation_error
from hallucination_lab.verification.integrity import EvidenceCheck

logger = logging.getLogger(__name__)

SERIES_COLOR = "#2a78d6"  # single-series bars (validated default palette, slot 1)
REVIEWER_RE = re.compile(r"^[\w.@-]{1,64}$")

SESSION_KEY = "current_session_id"
REVIEWER_KEY = "reviewer_id"
SEEN_KEY = "seen_versions"


class StaleViewError(HallucinationLabError):
    """The session changed after the page was rendered; the action was not applied."""


# --------------------------------------------------------------------- workspace


def workspace() -> Workspace:
    return Workspace.from_env()


def current_session_id() -> str | None:
    return st.session_state.get(SESSION_KEY)


def open_session(session_id: str | None) -> None:
    st.session_state[SESSION_KEY] = session_id
    st.session_state.pop("pending_verification", None)


def load_current(ws: Workspace) -> LoadedSession | None:
    """Reload the selected session from disk (fresh on every run)."""
    sid = current_session_id()
    if not sid:
        return None
    try:
        loaded = ws.load(sid)
    except HallucinationLabError as exc:
        st.error(f"Could not open session {sid}: {exc}")
        open_session(None)
        return None
    return loaded


def mark_seen(loaded: LoadedSession) -> None:
    """Call after rendering: records which file version the user is looking at."""
    st.session_state.setdefault(SEEN_KEY, {})[loaded.session_id] = loaded.version


def require_fresh(loaded: LoadedSession) -> None:
    seen = st.session_state.get(SEEN_KEY, {}).get(loaded.session_id)
    if seen is not None and seen != loaded.version:
        mark_seen(loaded)
        raise StaleViewError(
            "This session changed since the page was displayed (another tab, the CLI, or a "
            "repeated click). Nothing was saved. Review the refreshed page and try again."
        )


def after_save(loaded: LoadedSession, message: str) -> None:
    mark_seen(loaded)
    st.session_state["flash"] = message
    st.rerun()


def show_flash() -> None:
    message = st.session_state.pop("flash", None)
    if message:
        st.success(message)


def reviewer_id(required_for: str | None = None) -> str | None:
    value = (st.session_state.get(REVIEWER_KEY) or "").strip()
    if value and REVIEWER_RE.fullmatch(value):
        return value
    if required_for:
        st.warning(
            f"Enter a reviewer ID in the sidebar to {required_for} "
            "(letters, digits, . _ - @; no spaces)."
        )
    return None


# ------------------------------------------------------------------ error handling


def safe_page(render: Callable[[], None]) -> Callable[[], None]:
    """Show actionable messages instead of stack traces."""

    @functools.wraps(render)
    def wrapper() -> None:
        try:
            render()
        except StaleViewError as exc:
            st.warning(str(exc))
        except HallucinationLabError as exc:
            st.error(str(exc))
        except ValidationError as exc:
            st.error(f"Invalid input: {summarize_validation_error(exc)}")
        except Exception as exc:  # last resort: no traceback in the browser
            if type(exc).__module__.startswith("streamlit"):
                raise  # st.rerun / st.stop use exceptions internally
            logger.exception("unexpected error in page %s", render.__name__)
            st.error(
                "Something unexpected went wrong and the action was not completed. "
                "Your saved data is unchanged. Details were written to the terminal log."
            )
            with st.expander("Technical summary"):
                st.code(f"{type(exc).__name__}: {exc}")

    return wrapper


def run_action(fn: Callable[[], None]) -> None:
    """Run a button handler, converting expected failures into messages."""
    try:
        fn()
    except StaleViewError as exc:
        st.warning(str(exc))
    except HallucinationLabError as exc:
        st.error(str(exc))
    except ValidationError as exc:
        st.error(f"Invalid input: {summarize_validation_error(exc)}")


# --------------------------------------------------------------------- components


def provenance_badge(session: VerificationSession) -> str:
    label = PROVENANCE_LABELS[session.response.provenance.value]
    if session.response.provenance is ResponseProvenance.SYNTHETIC_FIXTURE:
        return f":orange-badge[{label}: not a real model output]"
    return f":blue-badge[{label}]"


def session_label(summary) -> str:
    return f"{summary.title[:60]}  ·  {summary.session_id}"


def bar_chart(data: dict[str, int], title: str, value_title: str = "Claims") -> None:
    """Single-series horizontal bar chart with direct labels, tooltip and a table view."""
    frame = pd.DataFrame({"category": list(data), "value": list(data.values())})
    st.markdown(f"**{title}**")
    if frame["value"].sum() == 0:
        st.caption("Nothing to show yet.")
        return
    base = alt.Chart(frame).encode(
        y=alt.Y("category:N", sort=None, title=None, axis=alt.Axis(labelLimit=220)),
        x=alt.X("value:Q", title=value_title, axis=alt.Axis(tickMinStep=1, grid=False)),
        tooltip=[alt.Tooltip("category:N", title=""), alt.Tooltip("value:Q", title=value_title)],
    )
    bars = base.mark_bar(color=SERIES_COLOR, cornerRadiusEnd=4, height={"band": 0.6})
    labels = base.mark_text(align="left", dx=4).encode(text="value:Q")
    st.altair_chart((bars + labels).properties(height=40 * len(frame) + 30), width="stretch")
    with st.expander("Show as table"):
        st.dataframe(
            frame.rename(columns={"category": "", "value": value_title}),
            hide_index=True,
            width="stretch",
        )


def evidence_card(ev: Evidence, check: EvidenceCheck | None, rank: int | None = None) -> None:
    """Source details for one retrieved passage. The score is labelled relevance-only."""
    with st.container(border=True):
        head = f"**{ev.source_title}**"
        if rank:
            head = f"#{rank} · " + head
        st.markdown(head)
        score = f"{ev.retrieval_score:.3f}" if ev.retrieval_score is not None else "n/a"
        st.caption(
            f"BM25 relevance score {score}: lexical overlap only, not evidence of "
            f"support · matched terms: {', '.join(ev.matched_terms) or '-'}"
        )
        details = []
        if ev.source_url:
            details.append(f"[Source link]({ev.source_url})")
        if ev.document_reference:
            details.append(f"`{ev.document_reference}`")
        loc = ev.passage_location
        if loc and loc.section:
            details.append(f"section: {loc.section}")
        if ev.publication_date:
            details.append(f"published {ev.publication_date}")
        if check and check.last_updated:
            details.append(f"last updated {check.last_updated}")
        if ev.document_version:
            details.append(ev.document_version)
        if check and check.license:
            details.append(f"licence: {check.license}")
        if ev.source_origin:
            details.append(ev.source_origin.value)
        st.caption(" · ".join(details))
        if check is not None:
            text = f"Source check: {check.status.value} ({check.detail}). {check.credibility_note}."
            if check.blocks_citation:
                st.error(text + " Cannot be cited until retrieval is re-run.")
            elif check.status.value != "verified":
                st.warning(text)
            else:
                st.caption(text)
        st.text(ev.quoted_passage)


def claim_context(session: VerificationSession, claim, radius: int = 220) -> str:
    text = session.response.text
    if claim.span is None:
        return "_This claim has no position in the response (reviewer wording or merged)._"
    s, e = claim.span.start, claim.span.end
    before = text[max(0, s - radius) : s].replace("*", r"\*")
    after = text[e : e + radius].replace("*", r"\*")
    target = text[s:e].replace("*", r"\*")
    return (
        f"{'…' if s > radius else ''}{before}**:blue-background[{target}]**{after}"
        f"{'…' if e + radius < len(text) else ''}"
    )
