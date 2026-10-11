"""Benchmark: browse candidates, review blind, adjudicate, approve, export.

Benchmark records live in ``data/benchmark/`` (or ``HALLAB_BENCHMARK_DIR``) and
are separate from analysis sessions. All rules come from
``hallucination_lab.benchmark.workflow``; this page only collects input.
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from hallucination_lab.benchmark import workflow as wf
from hallucination_lab.benchmark.models import (
    Difficulty,
    Domain,
    EvidenceJudgement,
    Partition,
    Reliability,
    ReviewKind,
    ReviewStatus,
)
from hallucination_lab.benchmark.store import BenchmarkError, BenchmarkStore
from hallucination_lab.models import EvidenceLabel, Verdict
from hallucination_lab.ui import common

NOTICE = (
    "**Under development.** Candidate claims and any proposed labels were written by an "
    "AI assistant and are provisional. Only items a human has **APPROVED** are ground "
    "truth."
)
REVEAL_KEY = "bm_revealed"  # item ids whose proposed label was revealed in this browser


def _store() -> BenchmarkStore:
    return BenchmarkStore()


def _overview(store: BenchmarkStore, items) -> None:
    summary = wf.status_summary(store)
    approved = sum(summary["approved_labels"].values())
    cols = st.columns(5)
    cols[0].metric("Candidates", summary["total"])
    cols[1].metric("Ready for review", summary["by_status"]["READY_FOR_REVIEW"])
    cols[2].metric("Disputed", summary["by_status"]["DISPUTED"])
    cols[3].metric("Research needed", summary["by_status"]["RESEARCH_NEEDED"])
    cols[4].metric("Human-approved", approved)
    left, right = st.columns(2)
    with left:
        common.bar_chart(
            {
                s.value.replace("_", " ").title(): summary["by_status"][s.value]
                for s in ReviewStatus
            },
            "Items by review status",
            "Items",
        )
    with right:
        common.bar_chart(
            {
                v.value.replace("_", " ").title(): summary["approved_labels"].get(v.value, 0)
                for v in Verdict
            },
            "Human-approved labels",
            "Items",
        )
        if not approved:
            st.caption(
                "No approved labels yet. Evaluation metrics cannot be computed until "
                "items are approved and exported."
            )
    if summary["disputed"]:
        st.warning("Disagreements awaiting adjudication: " + ", ".join(summary["disputed"]))
    for problem in summary["unreadable"]:
        st.error(problem)


def _filters(items):
    cols = st.columns(4)
    status = cols[0].multiselect("Status", [s.value for s in ReviewStatus])
    domain = cols[1].multiselect("Domain", [d.value for d in Domain])
    difficulty = cols[2].multiselect("Difficulty", [d.value for d in Difficulty])
    partition = cols[3].multiselect("Partition", [p.value for p in Partition])
    return [
        i
        for i in items
        if (not status or i.status.value in status)
        and (not domain or i.domain.value in domain)
        and (not difficulty or i.difficulty.value in difficulty)
        and (not partition or i.partition.value in partition)
    ]


def _evidence_view(store, item) -> None:
    refs = store.references()
    if not item.evidence:
        st.caption("No evidence passages attached.")
    for n, e in enumerate(item.evidence, 1):
        ref = refs.get(e.reference_id)
        with st.container(border=True):
            st.markdown(
                f"**Evidence {n}**: {ref.title if ref else e.reference_id} `{e.evidence_id}`"
            )
            if ref:
                bits = [
                    ref.author_or_organization or "",
                    ref.source_type.value,
                    f"accessed {ref.accessed_on}" if ref.accessed_on else "",
                    ref.document_version or "",
                    f"licence: {ref.license or 'unknown'}",
                ]
                st.caption(" · ".join(b for b in bits if b))
                if ref.url:
                    st.markdown(f"[Open source]({ref.url})")
                if ref.reliability_notes:
                    st.caption(f"Reliability notes: {ref.reliability_notes}")
            st.caption(f"Locator: {e.locator}")
            if e.passage_text:
                st.text(e.passage_text)
            else:
                st.info(
                    "Passage text not stored (licence). Open the source and use the "
                    f"locator. Stored hash: `{(e.passage_sha256 or '')[:16]}…`"
                )
    for s in item.research_log:
        st.caption(
            f"Search · {s.service} · {s.performed_at:%Y-%m-%d} · '{s.query}': {s.result_summary}"
        )
    for t in item.research_tasks:
        st.caption(f"Research task ({'done' if t.resolved else 'open'}): {t.task}. {t.reason}")


def _history(store, item) -> None:
    annotations = store.annotations(item.item_id)
    adjudications = store.adjudications(item.item_id)
    current = {a.annotation_id for a in wf.current_annotations(item, annotations)}
    if annotations:
        st.markdown("**Human reviews** (all kept; corrections are new records)")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "review": a.annotation_id,
                        "reviewer": a.reviewer_id,
                        "kind": a.review_kind.value,
                        "verdict": a.verdict.value,
                        "current": a.annotation_id in current,
                        "revises": a.revision_of or "",
                        "saw proposed label": a.proposed_label_seen_before_review,
                        "time (UTC)": a.created_at.strftime("%Y-%m-%d %H:%M"),
                        "rationale": a.rationale,
                    }
                    for a in annotations
                ]
            ),
            hide_index=True,
            width="stretch",
        )
    for d in adjudications:
        st.caption(
            f"Adjudication {d.adjudication_id} by {d.adjudicator_id}: "
            f"{d.final_verdict.value}: {d.rationale}"
        )
    for a in item.approvals:
        st.caption(
            f"Approval {a.approval_id}: {a.final_verdict.value} ({a.review_level.value}) "
            f"by {a.approved_by} at {a.approved_at:%Y-%m-%d %H:%M}"
        )
    st.caption(
        "Status history: "
        + " → ".join(f"{e.to_status.value} ({e.actor})" for e in item.status_events)
    )


def _review_form(store, item, reviewer: str) -> None:
    annotations = store.annotations(item.item_id)
    current = wf.current_annotations(item, annotations)
    mine = [a for a in current if a.reviewer_id == reviewer]
    revise = None
    if mine:
        options = ["new review", *[a.annotation_id for a in mine]]
        choice = st.radio(
            "Record",
            options,
            horizontal=True,
            key=f"bm_mode_{item.item_id}",
            format_func=lambda o: o if o == "new review" else f"revise {o}",
        )
        revise = None if choice == "new review" else choice
    if revise:
        kind = None
    elif not current:
        kind = ReviewKind.FIRST_REVIEW
    elif mine:
        kind = ReviewKind.REPEAT_REVIEW_SAME_REVIEWER
    else:
        kind = ReviewKind.INDEPENDENT_SECOND_REVIEW
    st.caption(
        f"Review type: **{kind.value if kind else 'revision of ' + revise}**"
        + (
            " (a delayed self-review, not independent agreement)"
            if kind is ReviewKind.REPEAT_REVIEW_SAME_REVIEWER
            else ""
        )
    )
    seen = item.item_id in st.session_state.get(REVEAL_KEY, set())
    with st.form(f"bm_review_{item.item_id}_{revise or kind.value}"):
        judgements = []
        for n, e in enumerate(item.evidence, 1):
            c1, c2, c3 = st.columns([2, 1, 2])
            relation = c1.radio(
                f"Evidence {n} shows",
                [x.value for x in EvidenceLabel],
                horizontal=True,
                index=None,
                key=f"bm_rel_{item.item_id}_{n}",
            )
            reliability = c2.selectbox(
                f"Evidence {n} reliability",
                [r.value for r in Reliability],
                index=3,
                key=f"bm_relia_{item.item_id}_{n}",
            )
            note = c3.text_input(f"Evidence {n} note", key=f"bm_note_{item.item_id}_{n}")
            judgements.append((e, relation, reliability, note))
        verdict = st.selectbox(
            "Your verdict",
            [None, *[v.value for v in Verdict]],
            format_func=lambda v: "Choose a verdict" if v is None else v,
            key=f"bm_verdict_{item.item_id}",
        )
        cited = st.multiselect(
            "Evidence cited for SUPPORTED / CONTRADICTED",
            [e.evidence_id for e in item.evidence],
            format_func=lambda eid: (
                f"Evidence {[e.evidence_id for e in item.evidence].index(eid) + 1}"
            ),
            key=f"bm_cited_{item.item_id}",
        )
        rationale = st.text_area(
            "Rationale: what the evidence shows and why", key=f"bm_rat_{item.item_id}"
        )
        uncertainty = st.text_input("Uncertainty note (optional)", key=f"bm_unc_{item.item_id}")
        research = st.checkbox(
            "More research is needed before this can be labelled", key=f"bm_res_{item.item_id}"
        )
        reason = (
            st.text_input(
                "Reason for revising (required for a revision)", key=f"bm_reason_{item.item_id}"
            )
            if revise
            else None
        )
        confirm = st.checkbox(
            "I confirm this judgement is my own assessment of the evidence.",
            key=f"bm_confirm_{item.item_id}",
        )
        submitted = st.form_submit_button("Save review", type="primary")
    if not submitted:
        return
    if verdict is None:
        st.error("Choose a verdict.")
        return
    unjudged = [n for n, (_, rel, _, _) in enumerate(judgements, 1) if rel is None]
    if unjudged:
        st.error(f"Judge every evidence passage (missing: {unjudged}).")
        return
    try:
        annotation, updated = wf.record_annotation(
            store,
            item.item_id,
            reviewer_id=reviewer,
            review_kind=kind or ReviewKind.FIRST_REVIEW,
            verdict=Verdict(verdict),
            rationale=rationale,
            evidence_judgements=[
                EvidenceJudgement(
                    evidence_id=e.evidence_id,
                    relation=EvidenceLabel(rel),
                    reliability=Reliability(rl),
                    note=nt or None,
                )
                for e, rel, rl, nt in judgements
            ],
            cited_evidence_ids=cited,
            confirmed_own_judgement=confirm,
            uncertainty_note=uncertainty or None,
            research_needed=research,
            proposed_label_seen_before_review=seen,
            revision_of=revise,
            revision_reason=reason,
        )
    except (BenchmarkError, ValueError) as exc:
        st.error(f"Not saved: {common.summarize_error(exc)}")
        return
    st.session_state["flash"] = (
        f"Saved review {annotation.annotation_id}; item is now {updated.status.value}."
    )
    st.rerun()


def _adjudication_form(store, item, reviewer: str) -> None:
    with st.form(f"bm_adj_{item.item_id}"):
        verdict = st.selectbox("Final verdict", [v.value for v in Verdict], index=None)
        rationale = st.text_area("Adjudication rationale")
        confirm = st.checkbox("I confirm this adjudication is my own assessment.")
        go = st.form_submit_button("Save adjudication", type="primary")
    if go:
        try:
            wf.adjudicate(
                store,
                item.item_id,
                adjudicator_id=reviewer,
                final_verdict=Verdict(verdict) if verdict else None,
                rationale=rationale,
                confirmed_own_judgement=confirm,
            )
        except (BenchmarkError, ValueError) as exc:
            st.error(f"Not saved: {common.summarize_error(exc)}")
            return
        st.session_state["flash"] = "Adjudication saved."
        st.rerun()


def _approval(store, item, reviewer: str) -> None:
    decision = wf.decide_label(
        item, store.annotations(item.item_id), store.adjudications(item.item_id)
    )
    if decision.verdict is None:
        st.caption(f"Not approvable yet: {decision.problem}.")
        return
    st.markdown(
        f"Current human label: **{decision.verdict.value}** ({decision.review_level.value})"
    )
    confirm = st.checkbox(
        "I checked the reviews and evidence myself and approve this label.",
        key=f"bm_apr_c_{item.item_id}",
    )
    if st.button("Approve label", disabled=not confirm, key=f"bm_apr_{item.item_id}"):
        try:
            wf.approve(store, item.item_id, approver_id=reviewer, confirmed_own_judgement=True)
        except BenchmarkError as exc:
            st.error(f"Not approved: {exc}")
            return
        st.session_state["flash"] = f"Approved {item.item_id} as {decision.verdict.value}."
        st.rerun()


def _item_detail(store, item) -> None:
    st.markdown(f"### {item.claim_text}")
    st.caption(
        f"`{item.item_id}` · {item.status.value} · {item.domain.value} · "
        f"{item.difficulty.value} · partition {item.partition.value} · types: "
        + ", ".join(t.value for t in item.claim_types)
    )
    if item.temporal_scope:
        st.caption(f"Temporal scope: {item.temporal_scope}")
    if item.original_context:
        with st.expander("Original response context (illustrative fixture)"):
            st.text(item.original_context)
    if item.split_recommendation and item.split_recommendation.should_split:
        st.warning(
            "Compound claim, split recommended: "
            + " / ".join(item.split_recommendation.proposed_parts)
        )
    st.caption(
        f"Candidate written by {item.candidate_provenance.author} "
        f"({item.candidate_provenance.author_type.value})."
    )
    _evidence_view(store, item)
    if item.proposed_label:
        revealed = st.session_state.setdefault(REVEAL_KEY, set())
        if item.item_id in revealed:
            p = item.proposed_label
            st.info(f"AI-proposed label (NOT ground truth): {p.verdict.value}: {p.rationale}")
        elif st.button(
            "Reveal AI-proposed label (may bias your review; recorded if you review afterwards)",
            key=f"bm_reveal_{item.item_id}",
        ):
            revealed.add(item.item_id)
            st.rerun()
    _history(store, item)

    reviewer = common.reviewer_id()
    if not reviewer:
        st.info("Enter a reviewer ID in the sidebar to review, adjudicate or approve.")
        return
    if wf.is_automated_actor(reviewer):
        st.error("This reviewer ID is reserved for automated authors and cannot review.")
        return
    if item.status in wf.REVIEWABLE:
        st.markdown("#### Your review")
        _review_form(store, item, reviewer)
    elif item.status is ReviewStatus.RESEARCH_NEEDED:
        st.info("Research needed: add evidence or mark it ready (CLI `hallab benchmark mark`).")
    if item.status is ReviewStatus.DISPUTED:
        st.markdown("#### Adjudication")
        _adjudication_form(store, item, reviewer)
    if item.status in {ReviewStatus.REVIEWED, ReviewStatus.DISPUTED}:
        st.markdown("#### Approval")
        _approval(store, item, reviewer)


def _exports(store) -> None:
    st.subheader("Versioned exports")
    exports = store.exports()
    for path in exports:
        export = store.load_export(path)
        st.download_button(
            f"Download {path.name} ({len(export.items)} approved items)",
            path.read_text(encoding="utf-8"),
            file_name=path.name,
            mime="application/json",
            key=f"bm_dl_{path.name}",
        )
    reviewer = common.reviewer_id()
    with st.form("bm_export"):
        version = st.text_input("New version (MAJOR.MINOR.PATCH)", placeholder="1.0.0")
        go = st.form_submit_button("Export approved items as an immutable version")
    if go:
        if not reviewer:
            st.error("Enter a reviewer ID in the sidebar first.")
        else:
            try:
                export, path = wf.export_approved(store, version.strip(), reviewer)
            except (BenchmarkError, ValueError) as exc:
                st.error(f"Not exported: {common.summarize_error(exc)}")
            else:
                st.success(
                    f"Exported {len(export.items)} items to {path.name} "
                    f"(dataset_sha256 {export.dataset_sha256[:16]}…)."
                )
    st.download_button(
        "Download provisional candidates (NOT ground truth)",
        json.dumps(wf.provisional_export(store), indent=2, ensure_ascii=False),
        file_name="benchmark-provisional-candidates.json",
        mime="application/json",
        key="bm_dl_prov",
    )


@common.safe_page
def render() -> None:
    st.title("Benchmark")
    common.show_flash()
    st.warning(NOTICE)
    store = _store()
    if not store.exists():
        st.info(
            f"No benchmark found at `{store.root}`. Run "
            "`python scripts/build_benchmark_candidates.py --download DIR` to create the "
            "candidate set."
        )
        return
    st.caption(f"Benchmark data: `{store.root}` (separate from analysis sessions).")
    items, _ = store.load_items()
    _overview(store, items)
    st.subheader("Candidates")
    shown = _filters(items)
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "item": i.item_id,
                    "status": i.status.value,
                    "domain": i.domain.value,
                    "difficulty": i.difficulty.value,
                    "partition": i.partition.value,
                    "approved label": i.current_approval.final_verdict.value
                    if i.current_approval
                    else "",
                    "claim": i.claim_text,
                }
                for i in shown
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    if shown:
        choice = st.selectbox(
            "Open item",
            [i.item_id for i in shown],
            format_func=lambda iid: next(
                f"{i.claim_text[:80]} [{i.status.value}]" for i in shown if i.item_id == iid
            ),
            key="bm_open",
        )
        _item_detail(store, store.load_item(choice))
    _exports(store)
