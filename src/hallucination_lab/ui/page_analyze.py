"""Analyze response: paste text, extract claims, edit them, retrieve evidence."""

from __future__ import annotations

import hashlib

import streamlit as st

from hallucination_lab.errors import HallucinationLabError
from hallucination_lab.models import RetrievalOutcome
from hallucination_lab.retrieval.search import RetrievalConfig
from hallucination_lab.services.workspace import LoadedSession, Workspace
from hallucination_lab.ui import common

EXTRACTION_WARNING = (
    "Claim extraction is **rule-based**: it splits sentences and applies transparent rules. "
    "It can miss claims, split them wrongly, or misclassify opinions and facts. Review the "
    "list below and edit it before verifying."
)
RECENT_KEY = "recent_extractions"  # text hash -> session id, to stop accidental duplicates


def _new_analysis(ws: Workspace) -> None:
    with st.form("new_analysis"):
        text = st.text_area(
            "AI-generated response",
            height=200,
            placeholder="Paste the response text here. It is stored exactly as entered.",
        )
        title = st.text_input("Title (optional)")
        synthetic = st.checkbox("This text is an illustrative example (not a real model output)")
        another = st.checkbox("Create a new session even if I already analysed this exact text")
        submitted = st.form_submit_button("Extract claims", type="primary")
    if not submitted:
        return
    if not text.strip():
        st.error("The response is empty. Paste some text first.")
        return
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    recent = st.session_state.setdefault(RECENT_KEY, {})
    if digest in recent and not another:
        st.warning(
            f"You already created session `{recent[digest]}` from this exact text in "
            "this browser session; it is now open. Tick the box above to create another."
        )
        common.open_session(recent[digest])
        return

    def create() -> None:
        loaded = ws.create_session(text, title=title.strip() or None, synthetic=synthetic)
        recent[digest] = loaded.session_id
        common.open_session(loaded.session_id)
        n = len(loaded.session.claims)
        common.after_save(
            loaded,
            f"Saved session {loaded.session_id} with {n} candidate claim(s)."
            if n
            else f"Saved session {loaded.session_id}, but no candidate claims were "
            "found. Add claims manually below.",
        )

    common.run_action(create)


def _open_saved(ws: Workspace) -> None:
    summaries, problems = ws.list_sessions()
    for problem in problems:
        st.warning(f"Skipped unreadable session file: {problem}")
    if summaries:
        options = {common.session_label(s): s.session_id for s in summaries}
        choice = st.selectbox(
            "Saved sessions (newest first)",
            list(options),
            index=None,
            placeholder="Choose a session",
        )
        if choice and st.button("Open selected session"):
            common.open_session(options[choice])
            st.rerun()
    else:
        st.caption("No saved sessions yet.")
    manual = st.text_input("…or open by session ID", placeholder="ses_0123456789abcdef")
    if manual and st.button("Open by ID"):
        try:
            ws.load(manual)
        except HallucinationLabError as exc:
            st.error(f"Cannot open {manual.strip()!r}: {exc}")
        else:
            common.open_session(manual.strip())
            st.rerun()


def _claims_section(ws: Workspace, loaded: LoadedSession) -> None:
    session = loaded.session
    st.subheader("Candidate claims")
    st.info(EXTRACTION_WARNING)
    reviewer = common.reviewer_id()
    if not session.claims:
        st.warning(
            "No candidate claims were extracted. Add the factual claims you want to check manually."
        )
    for n, claim in enumerate(session.claims, 1):
        verdict = session.human_verdict(claim.claim_id) if claim.is_active else None
        if not claim.is_active:
            tag = ":gray-badge[inactive]"
        elif verdict:
            tag = f":violet-badge[verified: {verdict.verdict.value}]"
        elif claim.requires_verification:
            tag = ":blue-badge[to verify]"
        else:
            reason = claim.non_checkable_reason.value if claim.non_checkable_reason else "reviewer"
            tag = f":gray-badge[skipped: {reason}]"
        with st.expander(f"{n}. {claim.normalized_claim}", expanded=False):
            st.markdown(tag)
            if claim.original_text != claim.normalized_claim:
                st.caption(f"Original wording: “{claim.original_text}”")
            st.caption("Why: " + "; ".join(claim.extraction_notes or ["added by a reviewer"]))
            f = claim.features
            facts = [
                f"{k}: {', '.join(v)}"
                for k, v in [
                    ("numbers", f.numbers),
                    ("dates", f.dates),
                    ("entities", f.named_entities),
                    ("citations", f.citations),
                    ("quotes", f.quotations),
                    ("time-sensitive", f.temporal_markers),
                ]
                if v
            ]
            if facts:
                st.caption(" · ".join(facts))
            if claim.reviewer_edits:
                st.caption(
                    f"{len(claim.reviewer_edits)} reviewer edit(s); last by "
                    f"{claim.reviewer_edits[-1].reviewer_id}"
                )
            if not reviewer:
                st.caption("Enter a reviewer ID in the sidebar to edit claims.")
                continue
            _claim_actions(ws, loaded, claim, reviewer)

    if reviewer:
        with st.form("add_claim", clear_on_submit=True):
            text = st.text_input("Add a claim the extractor missed")
            if st.form_submit_button("Add claim") and text.strip():

                def add() -> None:
                    common.require_fresh(loaded)
                    common.after_save(ws.add_claim(loaded, text, reviewer), "Claim added.")

                common.run_action(add)


def _claim_actions(ws: Workspace, loaded: LoadedSession, claim, reviewer: str) -> None:
    cid = claim.claim_id

    def act(fn, message: str) -> None:
        def run() -> None:
            common.require_fresh(loaded)
            common.after_save(fn(), message)

        common.run_action(run)

    if not claim.is_active:
        if st.button("Restore claim", key=f"restore_{cid}"):
            act(lambda: ws.restore_claim(loaded, cid, reviewer), "Claim restored.")
        return
    cols = st.columns(3)
    flagged = claim.requires_verification
    if cols[0].button(
        "Skip verification" if flagged else "Mark for verification", key=f"flag_{cid}"
    ):
        act(lambda: ws.set_flag(loaded, cid, not flagged, reviewer), "Verification flag updated.")
    if cols[1].button("Remove claim", key=f"remove_{cid}"):
        st.session_state[f"confirm_remove_{cid}"] = True
    if st.session_state.get(f"confirm_remove_{cid}"):
        st.warning(
            "Remove this claim? It is kept in the file as inactive (audit trail) and "
            "can be restored."
        )
        if st.button("Yes, remove it", key=f"remove_yes_{cid}"):
            st.session_state.pop(f"confirm_remove_{cid}")
            act(lambda: ws.remove_claim(loaded, cid, reviewer), "Claim removed.")
    with st.form(f"edit_{cid}"):
        new_text = st.text_input(
            "Edit claim wording (the original wording is kept)", value=claim.normalized_claim
        )
        parts = st.text_area("…or split into several claims (one per line)", height=80)
        c1, c2 = st.columns(2)
        save_edit = c1.form_submit_button("Save wording")
        do_split = c2.form_submit_button("Split")
    if save_edit:
        act(lambda: ws.edit_claim(loaded, cid, new_text, reviewer), "Claim wording saved.")
    if do_split:
        lines = [p for p in parts.splitlines() if p.strip()]
        act(lambda: ws.split_claim(loaded, cid, lines, reviewer), f"Split into {len(lines)}.")


def _retrieval_section(ws: Workspace, loaded: LoadedSession) -> None:
    st.subheader("Evidence retrieval")
    st.caption(
        "BM25 finds passages that share words with each claim. These are candidates "
        "for you to judge on the verification page, not verified evidence."
    )
    collections = ws.catalog.list_collections()
    if not collections:
        st.error(
            "No reference collections found. The packaged `demo` collection should be in "
            "`data/references/demo`; see the Reference collections page."
        )
        return
    names = [c.name for c in collections]
    with st.form("retrieve"):
        cols = st.columns(3)
        collection = cols[0].selectbox(
            "Reference collection", names, index=names.index("demo") if "demo" in names else 0
        )
        top_k = cols[1].number_input("Passages per claim", 1, 20, 5)
        per_source = cols[2].number_input("Max passages per document", 1, 20, 3)
        include = st.checkbox("Also search claims marked as skipped")
        go = st.form_submit_button("Retrieve evidence", type="primary")
    if go:

        def retrieve() -> None:
            common.require_fresh(loaded)
            with st.spinner("Searching…"):
                updated, run, rebuilt = ws.retrieve(
                    loaded,
                    collection,
                    RetrievalConfig(top_k=int(top_k), max_per_source=int(per_source)),
                    include_non_checkable=include,
                )
            found = sum(r.outcome is RetrievalOutcome.CANDIDATES_FOUND for r in run.results)
            note = " (index rebuilt first)" if rebuilt else ""
            common.after_save(
                updated,
                f"Searched {len(run.results)} claim(s){note}; {found} have candidate passages.",
            )

        common.run_action(retrieve)

    session = loaded.session
    if not session.retrieval_runs:
        st.caption("No retrieval has been run for this session yet.")
        return
    checks = ws.evidence_checks(session)
    evidence = {e.evidence_id: e for e in session.evidence}
    for claim in session.active_claims:
        latest = session.latest_retrieval(claim.claim_id)
        if latest is None:
            continue
        label = {
            "candidates_found": f"{len(latest.evidence_ids)} candidate(s)",
            "no_candidates": "no candidates",
            "no_searchable_terms": "no searchable terms",
        }[latest.outcome.value]
        with st.expander(f"{claim.normalized_claim}  ·  {label}"):
            if latest.outcome is not RetrievalOutcome.CANDIDATES_FOUND:
                st.info(
                    "Nothing found. That only means no passage shares words with the "
                    "claim; it is not evidence that the claim is false."
                )
            for rank, eid in enumerate(latest.evidence_ids, 1):
                common.evidence_card(evidence[eid], checks.get(eid), rank)


@common.safe_page
def render() -> None:
    st.title("Analyze response")
    common.show_flash()
    ws = common.workspace()
    new_tab, open_tab = st.tabs(["New analysis", "Open saved session"])
    with new_tab:
        _new_analysis(ws)
    with open_tab:
        _open_saved(ws)

    loaded = common.load_current(ws)
    if loaded is None:
        return
    session = loaded.session
    st.divider()
    st.header(session.title)
    st.markdown(
        f"{common.provenance_badge(session)} · session `{session.session_id}` · "
        f"response `{session.response.response_id}`"
    )
    with st.expander("Original response (stored exactly as submitted; never modified)"):
        st.text(session.response.text)
    _claims_section(ws, loaded)
    _retrieval_section(ws, loaded)
    st.info("Next: open **Human verification** in the sidebar to judge each claim.")
    common.mark_seen(loaded)
