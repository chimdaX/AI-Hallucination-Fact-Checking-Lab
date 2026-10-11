"""Human verification: assess passages, choose a verdict, confirm, save; revise safely.

Everything is validated by the same workflow functions the CLI uses
(``build_verification``, ``record_verification``, drafts, revision checks), so
the web interface cannot bypass any rule. Form inputs are temporary until you
click *Save draft* or confirm a verification.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from hallucination_lab.errors import VerificationError
from hallucination_lab.models import (
    EvidenceAssessment,
    EvidenceLabel,
    Verdict,
    Verification,
    VerificationDraft,
    utc_now,
)
from hallucination_lab.services.workspace import LoadedSession, Workspace
from hallucination_lab.ui import common
from hallucination_lab.verification.workflow import (
    VerificationInput,
    check_revision_allowed,
    compute_progress,
    ordered_evidence_ids,
)

NOT_ASSESSED = "Not assessed"
LABEL_OPTIONS = [NOT_ASSESSED, *(label.value for label in EvidenceLabel)]
VERDICT_HELP = {
    None: "Choose a verdict",
    Verdict.SUPPORTED.value: "SUPPORTED: reliable evidence you marked SUPPORTING establishes it",
    Verdict.CONTRADICTED.value: "CONTRADICTED: reliable evidence you marked CONTRADICTING shows "
    "it is false",
    Verdict.INSUFFICIENT_EVIDENCE.value: "INSUFFICIENT_EVIDENCE: none, weak, ambiguous or "
    "unresolved conflicting evidence",
    Verdict.NOT_CHECKABLE.value: "NOT_CHECKABLE: opinion, subjective, or too vague to be objective",
}
PENDING_KEY = "pending_verification"


def _status_label(session, claim) -> str:
    v = session.human_verdict(claim.claim_id)
    if v:
        return f"✔ {v.verdict.value}"
    if not claim.requires_verification:
        return "not flagged"
    if any(d.claim_id == claim.claim_id for d in session.verification_drafts):
        return "draft"
    return "pending" if session.latest_retrieval(claim.claim_id) else "pending (not searched)"


def _show_claim(session, claim) -> None:
    st.markdown(f"### {claim.normalized_claim}")
    if claim.original_text != claim.normalized_claim:
        st.caption(f"Original wording: “{claim.original_text}”")
    st.markdown("**Response context**")
    st.markdown(common.claim_context(session, claim))
    info = [f"extraction: {claim.extraction_method.value}"]
    if claim.temporal_context:
        info.append(f"time context: {claim.temporal_context} (check evidence dates)")
    latest = session.latest_retrieval(claim.claim_id)
    if latest is None:
        info.append("retrieval: NOT SEARCHED (run it on Analyze response)")
    else:
        run = next(r for r in session.retrieval_runs if latest in r.results)
        p = run.parameters
        info.append(
            f"retrieval: {latest.outcome.value} in '{run.collection_name}' "
            f"(bm25 k1={p.get('k1')} b={p.get('b')} top_k={p.get('top_k')}, "
            f"{run.started_at:%Y-%m-%d %H:%M} UTC)"
        )
    st.caption(" · ".join(info))


def _show_active(session, verdict: Verification) -> None:
    evidence = {e.evidence_id: e for e in session.evidence}
    with st.container(border=True):
        st.markdown(
            f"**Active human verdict: {verdict.verdict.value}** by "
            f"`{verdict.reviewer_id}` at {verdict.verification_timestamp:%Y-%m-%d %H:%M} "
            f"UTC (`{verdict.verification_id}`)"
        )
        st.markdown(f"**Rationale:** {verdict.reasoning}")
        for role, ids in (
            ("supporting", verdict.supporting_evidence_ids),
            ("contradicting", verdict.contradicting_evidence_ids),
        ):
            for eid in ids:
                ev = evidence[eid]
                st.caption(f"Cited ({role}): {ev.source_title} · `{ev.document_reference}`")
        for note in verdict.noted_limitations:
            st.caption(f"Limitation: {note}")
        if verdict.session_id is None:
            st.caption("Legacy record (created before per-passage assessments existed).")
    chain = session.revision_chain(verdict.verification_id)
    if len(chain) > 1:
        st.markdown("**Revision history** (all records kept; only the latest is active)")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "record": v.verification_id,
                        "status": "active" if v is chain[-1] else "superseded",
                        "verdict": v.verdict.value,
                        "reviewer": v.reviewer_id,
                        "time (UTC)": v.verification_timestamp.strftime("%Y-%m-%d %H:%M"),
                        "correction reason": v.revision_reason or "",
                    }
                    for v in chain
                ]
            ),
            hide_index=True,
            width="stretch",
        )


def _form(
    ws: Workspace, loaded: LoadedSession, claim, reviewer: str, previous: Verification | None
) -> None:
    session = loaded.session
    cid = claim.claim_id
    prev_id = previous.verification_id if previous else None
    key = f"{cid}:{reviewer}:{prev_id or 'new'}"
    draft = session.get_draft(cid, reviewer)
    if draft and draft.revision_of != prev_id:
        st.warning(
            "You have a saved draft for this claim from a different task "
            f"(revision_of={draft.revision_of})."
        )
        if st.button("Discard that draft", key=f"discard_{key}"):

            def discard() -> None:
                common.require_fresh(loaded)
                common.after_save(ws.discard_draft(loaded, cid, reviewer), "Draft discarded.")

            common.run_action(discard)
        return
    if draft:
        st.info(f"Resuming your saved draft from {draft.updated_at:%Y-%m-%d %H:%M} UTC.")
    prefill = {
        a.evidence_id: a
        for a in (
            draft.evidence_assessments
            if draft
            else previous.evidence_assessments
            if previous
            else []
        )
    }
    checks = ws.evidence_checks(session, cid)
    blocked = [c for c in checks.values() if c.blocks_citation]
    if blocked:
        st.error(
            f"{len(blocked)} passage(s) have sources that changed or disappeared after "
            "retrieval and cannot be cited. Re-run retrieval on Analyze response to "
            "revalidate."
        )
    evidence = {e.evidence_id: e for e in session.evidence}
    ids = ordered_evidence_ids(session, cid)
    st.caption(
        "Form inputs are temporary until you click **Save draft** or confirm the "
        "verification. Saved drafts survive closing the browser."
    )

    with st.form(f"form_{key}"):
        reason = authorization = None
        if previous:
            reason = st.text_area(
                "Correction reason (required)", value=(draft.revision_reason if draft else "") or ""
            )
            if previous.reviewer_id != reviewer:
                authorization = st.text_input(
                    f"Authorization to revise {previous.reviewer_id}'s verdict (required): who "
                    "approved it and why"
                )
        st.markdown("**1. Assess each retrieved passage**")
        if not ids:
            st.caption(
                "No retrieved passages for this claim. Only INSUFFICIENT_EVIDENCE or "
                "NOT_CHECKABLE can be justified without evidence."
            )
        labels, notes = {}, {}
        for rank, eid in enumerate(ids, 1):
            common.evidence_card(evidence[eid], checks.get(eid), rank)
            prior = prefill.get(eid)
            labels[eid] = st.radio(
                "Your assessment",
                LABEL_OPTIONS,
                horizontal=True,
                key=f"lbl_{key}_{eid}",
                index=LABEL_OPTIONS.index(prior.label.value) if prior else 0,
            )
            notes[eid] = st.text_input(
                "Note (optional)",
                key=f"note_{key}_{eid}",
                value=(prior.explanation if prior else "") or "",
            )
        st.markdown("**2. Verdict and cited evidence**")
        verdict_options = [None, *(v.value for v in Verdict)]
        verdict = st.selectbox(
            "Verdict",
            verdict_options,
            format_func=lambda v: VERDICT_HELP[v],
            index=verdict_options.index(draft.verdict.value) if draft and draft.verdict else 0,
            key=f"verdict_{key}",
        )
        cited_default = [
            e
            for e in (
                (draft.supporting_evidence_ids + draft.contradicting_evidence_ids) if draft else []
            )
            if e in ids
        ]
        cited = st.multiselect(
            "Passages cited as justification (SUPPORTED: passages you marked SUPPORTING; "
            "CONTRADICTED: passages you marked CONTRADICTING)",
            ids,
            default=cited_default,
            format_func=lambda e: f"#{ids.index(e) + 1} {evidence[e].source_title}",
            key=f"cited_{key}",
        )
        rationale = st.text_area(
            "3. Rationale: what the evidence says and how it relates to the claim",
            value=(draft.rationale if draft else "") or "",
            key=f"rat_{key}",
        )
        c1, c2 = st.columns(2)
        save_clicked = c1.form_submit_button("Save draft")
        review_clicked = c2.form_submit_button("Check and review", type="primary")

    if not (save_clicked or review_clicked):
        return
    assessments = [
        EvidenceAssessment(
            evidence_id=eid, label=EvidenceLabel(labels[eid]), explanation=notes[eid] or None
        )
        for eid in ids
        if labels[eid] != NOT_ASSESSED
    ]
    supporting = cited if verdict == Verdict.SUPPORTED.value else []
    contradicting = cited if verdict == Verdict.CONTRADICTED.value else []
    now = utc_now()
    new_draft = VerificationDraft(
        claim_id=cid,
        reviewer_id=reviewer,
        started_at=draft.started_at if draft else now,
        updated_at=now,
        revision_of=prev_id,
        revision_reason=reason or None,
        evidence_assessments=assessments,
        verdict=Verdict(verdict) if verdict else None,
        rationale=rationale or None,
        supporting_evidence_ids=supporting,
        contradicting_evidence_ids=contradicting,
    )

    def save() -> None:
        common.require_fresh(loaded)
        updated = ws.save_draft(loaded, new_draft)
        if not review_clicked:
            common.after_save(updated, "Draft saved. It is not a verification yet.")
            return
        if verdict is None:
            common.after_save(updated, "Draft saved. Choose a verdict to continue.")
            return
        if cited and verdict not in {Verdict.SUPPORTED.value, Verdict.CONTRADICTED.value}:
            common.mark_seen(updated)
            st.error(
                "Cited passages are only used for SUPPORTED or CONTRADICTED. Clear them or "
                "change the verdict (your draft was saved)."
            )
            return
        inp = VerificationInput(
            claim_id=cid,
            verdict=Verdict(verdict),
            rationale=rationale,
            assessments=assessments,
            supporting_evidence_ids=supporting,
            contradicting_evidence_ids=contradicting,
            revision_of=prev_id,
            revision_reason=reason or None,
            revision_authorization=authorization or None,
        )
        try:
            prepared, warnings = ws.prepare_verification(updated, reviewer, inp)
        except VerificationError as exc:
            common.mark_seen(updated)
            st.error(f"Cannot record this verification yet: {exc}. Your draft was saved.")
            return
        st.session_state[PENDING_KEY] = {
            "key": key,
            "input": inp,
            "warnings": warnings,
            "preview": prepared.model_dump(mode="json"),
        }
        common.after_save(
            updated, "Draft saved. Review the summary below and confirm to save the verification."
        )

    common.run_action(save)


def _confirmation(ws: Workspace, loaded: LoadedSession, key: str, reviewer: str) -> None:
    pending = st.session_state.get(PENDING_KEY)
    if not pending or pending["key"] != key:
        return
    preview, warnings = pending["preview"], pending["warnings"]
    evidence = {e.evidence_id: e for e in loaded.session.evidence}
    with st.container(border=True):
        st.markdown("#### Review before saving (not saved yet)")
        st.markdown(f"**Verdict:** {preview['verdict']}")
        for role in ("supporting", "contradicting"):
            for eid in preview[f"{role}_evidence_ids"]:
                st.markdown(
                    f"- Cited ({role}): {evidence[eid].source_title} · "
                    f"`{evidence[eid].document_reference}`"
                )
        counts = {
            lbl.value: sum(a["label"] == lbl.value for a in preview["evidence_assessments"])
            for lbl in EvidenceLabel
        }
        summary = ", ".join(f"{n} {label.lower()}" for label, n in counts.items())
        st.markdown(f"**Passages assessed:** {summary}")
        st.markdown(f"**Rationale:** {preview['reasoning']}")
        if preview.get("revision_of"):
            st.markdown(f"**Revises** `{preview['revision_of']}`: {preview['revision_reason']}")
        for note in preview["noted_limitations"]:
            st.caption(f"Limitation: {note}")
        for w in warnings:
            st.warning(f"Rationale warning: {w}")
        confirm = st.checkbox(
            "I confirm this verdict is my own judgement of the evidence and "
            "want to save it permanently (saved verdicts can only be "
            "corrected by a new revision).",
            key=f"confirm_{key}",
        )
        ack = True
        if warnings:
            ack = st.checkbox(
                "I have read the rationale warnings and want to keep this rationale.",
                key=f"ack_{key}",
            )
        c1, c2 = st.columns(2)
        if c1.button(
            "Save verification", type="primary", disabled=not (confirm and ack), key=f"commit_{key}"
        ):

            def commit() -> None:
                common.require_fresh(loaded)
                verification, _ = ws.prepare_verification(loaded, reviewer, pending["input"])
                updated = ws.commit_verification(loaded, verification)
                st.session_state.pop(PENDING_KEY, None)
                st.session_state.pop(f"revise_{key.split(':')[0]}", None)  # correction done
                common.after_save(updated, f"Saved verification {verification.verification_id}.")

            common.run_action(commit)
        if c2.button("Back to editing", key=f"back_{key}"):
            st.session_state.pop(PENDING_KEY, None)
            st.rerun()


@common.safe_page
def render() -> None:
    st.title("Human verification")
    common.show_flash()
    ws = common.workspace()
    loaded = common.load_current(ws)
    if loaded is None:
        st.info("Open or create a session on **Analyze response** first.")
        return
    session = loaded.session
    st.markdown(
        f"**{session.title}** · {common.provenance_badge(session)} · `{session.session_id}`"
    )
    progress = compute_progress(session)
    cols = st.columns(4)
    cols[0].metric("Flagged claims", progress.marked_for_verification)
    cols[1].metric("Human-verified", progress.verified_by_human)
    cols[2].metric("Awaiting review", progress.awaiting_human_review)
    cols[3].metric("Drafts", progress.drafts_in_progress)
    st.caption(
        "Retrieval only suggested passages. Every verdict here is a human judgement and "
        "must cite passages you explicitly assessed."
    )

    claims = session.active_claims
    if not claims:
        st.warning("This session has no active claims. Add claims on Analyze response.")
        return
    order = sorted(
        range(len(claims)),
        key=lambda i: not _status_label(session, claims[i]).startswith(("pending", "draft")),
    )
    choice = st.selectbox(
        "Claim",
        order,
        format_func=lambda i: (
            f"[{_status_label(session, claims[i])}] {claims[i].normalized_claim[:90]}"
        ),
        key=f"claim_select_{session.session_id}",
    )
    claim = claims[choice]
    _show_claim(session, claim)

    reviewer = common.reviewer_id(required_for="record or correct verdicts")
    active = session.human_verdict(claim.claim_id)
    if active:
        _show_active(session, active)
        if reviewer:
            revise_key = f"revise_{claim.claim_id}"
            if not st.session_state.get(revise_key):
                if st.button("Correct this verdict (creates a new revision)"):
                    try:
                        # Early check for "already revised" / non-human records. The
                        # cross-reviewer authorization is entered in the form and checked
                        # for real when the correction is validated.
                        check_revision_allowed(
                            session,
                            active.verification_id,
                            reviewer,
                            authorization="entered-in-form",
                        )
                    except VerificationError as exc:
                        st.error(str(exc))
                    else:
                        st.session_state[revise_key] = True
                        st.rerun()
            else:
                st.markdown("#### Correction")
                st.caption(
                    "The current record stays unchanged; your correction is saved as a "
                    "new record that supersedes it."
                )
                if st.button("Cancel correction"):
                    st.session_state.pop(revise_key)
                    st.rerun()
                _confirmation(
                    ws, loaded, f"{claim.claim_id}:{reviewer}:{active.verification_id}", reviewer
                )
                _form(ws, loaded, claim, reviewer, active)
    elif reviewer:
        _confirmation(ws, loaded, f"{claim.claim_id}:{reviewer}:new", reviewer)
        _form(ws, loaded, claim, reviewer, None)
    common.mark_seen(loaded)
