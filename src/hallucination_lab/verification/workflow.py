"""Human verification workflow logic, independent of any user interface.

The interactive CLI (``interactive.py``) and the tests both drive these
functions. Rules enforced here, on top of the schema rules:

* A claim can have only one active human verification. A correction must be
  a revision of the active record (``revision_of``).
* Only the latest record in a revision chain can be revised (no branching).
  Revising another reviewer's record requires a written authorization.
* Evidence whose source changed or disappeared after retrieval cannot be cited.
* SUPPORTED and CONTRADICTED verdicts must cite passages the reviewer explicitly
  assessed as supporting or contradicting. Retrieval rank and score play no part.
* Rationales must pass the rationale error checks. Warnings are returned for
  the reviewer to consider.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from pydantic import ValidationError

from hallucination_lab.errors import VerificationError
from hallucination_lab.ids import VERIFICATION, make_id
from hallucination_lab.models import (
    Claim,
    EvidenceAssessment,
    EvidenceLabel,
    Provenance,
    RetrievalOutcome,
    Verdict,
    Verification,
    VerificationDraft,
    VerificationMethod,
    VerificationSession,
    utc_now,
)
from hallucination_lab.storage import summarize_validation_error
from hallucination_lab.verification.integrity import EvidenceCheck, SourceStatus
from hallucination_lab.verification.rationale import check_rationale, rationale_errors

METHOD_DETAIL = "human review of retrieved passages (hallab verify)"


@dataclass
class VerificationInput:
    claim_id: str
    verdict: Verdict
    rationale: str
    assessments: list[EvidenceAssessment] = field(default_factory=list)
    supporting_evidence_ids: list[str] = field(default_factory=list)
    contradicting_evidence_ids: list[str] = field(default_factory=list)
    revision_of: str | None = None
    revision_reason: str | None = None
    revision_authorization: str | None = None


# ---------------------------------------------------------------------- queries


def claims_awaiting_review(session: VerificationSession) -> list[Claim]:
    """Active claims flagged for verification without an active human verdict."""
    return session.pending_review()


def ordered_evidence_ids(session: VerificationSession, claim_id: str) -> list[str]:
    """Evidence for a claim: latest retrieval ranking first, then any older candidates."""
    latest = session.latest_retrieval(claim_id)
    ordered = list(latest.evidence_ids) if latest else []
    for ev in session.evidence_for(claim_id):
        if ev.evidence_id not in ordered:
            ordered.append(ev.evidence_id)
    return ordered


def check_revision_allowed(
    session: VerificationSession,
    verification_id: str,
    reviewer_id: str,
    authorization: str | None,
) -> Verification:
    try:
        previous = session.get_verification(verification_id)
    except KeyError as exc:
        raise VerificationError(f"no verification {verification_id} in this session") from exc
    if not previous.is_human_judgment:
        raise VerificationError("only human verifications can be revised")
    successor = next((v for v in session.verifications if v.revision_of == verification_id), None)
    if successor is not None:
        latest = session.human_verdict(previous.claim_id)
        raise VerificationError(
            f"{verification_id} was already revised by {successor.verification_id}; revise the "
            f"latest record ({latest.verification_id if latest else 'unknown'}) instead"
        )
    if previous.reviewer_id != reviewer_id and not (authorization and authorization.strip()):
        raise VerificationError(
            f"{verification_id} was recorded by {previous.reviewer_id}; revising another "
            "reviewer's verification requires --authorization explaining who approved it"
        )
    return previous


# ------------------------------------------------------------------- building


def _limitations(
    session: VerificationSession,
    inp: VerificationInput,
    checks: dict[str, EvidenceCheck],
) -> list[str]:
    notes: list[str] = []
    titles = {e.evidence_id: e.source_title for e in session.evidence}
    for eid in [*inp.supporting_evidence_ids, *inp.contradicting_evidence_ids]:
        check = checks.get(eid)
        title = titles.get(eid, eid)
        if check is None:
            notes.append(f"cited source '{title}' was not re-checked against its document")
            continue
        notes.append(f"cited source '{title}': {check.credibility_note}")
        if check.status in {SourceStatus.COLLECTION_UNAVAILABLE, SourceStatus.NOT_TRACEABLE}:
            notes.append(f"cited source '{title}': {check.detail}")
    latest = session.latest_retrieval(inp.claim_id)
    if latest is None:
        notes.append("no evidence retrieval was run for this claim before the verdict")
    elif latest.outcome is not RetrievalOutcome.CANDIDATES_FOUND:
        notes.append(f"latest retrieval outcome: {latest.outcome.value}")
    return list(dict.fromkeys(notes))


def build_verification(
    session: VerificationSession,
    reviewer_id: str,
    inp: VerificationInput,
    checks: dict[str, EvidenceCheck] | None = None,
    now: datetime | None = None,
) -> tuple[Verification, list[str]]:
    """Validate a reviewer's decision and return (verification, rationale warnings)."""
    now = now or utc_now()
    checks = checks or {}
    if not reviewer_id or not reviewer_id.strip():
        raise VerificationError("a reviewer ID is required")
    try:
        claim = session.get_claim(inp.claim_id)
    except KeyError as exc:
        raise VerificationError(f"claim {inp.claim_id} is not in this session") from exc
    if not claim.is_active:
        raise VerificationError(f"claim {inp.claim_id} is inactive (removed, split or merged)")

    if inp.revision_of is None:
        active = session.human_verdict(claim.claim_id)
        if active is not None:
            raise VerificationError(
                f"claim already has an active verification {active.verification_id} by "
                f"{active.reviewer_id}; correct it with --revise {active.verification_id}"
            )
    else:
        previous = check_revision_allowed(
            session, inp.revision_of, reviewer_id, inp.revision_authorization
        )
        if previous.claim_id != claim.claim_id:
            raise VerificationError("the revised verification belongs to a different claim")
        problems = rationale_errors(inp.revision_reason)
        if problems:
            raise VerificationError(f"correction reason rejected: {problems[0]}")

    evidence = {e.evidence_id: e for e in session.evidence_for(claim.claim_id)}
    cited = [*inp.supporting_evidence_ids, *inp.contradicting_evidence_ids]
    for eid in [*cited, *(a.evidence_id for a in inp.assessments)]:
        if eid not in evidence:
            raise VerificationError(f"evidence {eid} was not retrieved for this claim")
    for eid in cited:
        check = checks.get(eid)
        if check is not None and check.blocks_citation:
            raise VerificationError(
                f"evidence {eid} cannot be cited: {check.detail}. Re-run `hallab retrieve` to "
                "revalidate against the current documents"
            )

    assessments = [
        a
        if a.passage_sha256
        else a.model_copy(update={"passage_sha256": evidence[a.evidence_id].passage_sha256})
        for a in inp.assessments
    ]
    rationale_check = check_rationale(
        inp.rationale,
        inp.verdict.value,
        claim.normalized_claim,
        [evidence[e].quoted_passage for e in cited],
    )
    if rationale_check.errors:
        raise VerificationError(f"rationale rejected: {rationale_check.errors[0]}")

    try:
        verification = Verification(
            verification_id=make_id(
                VERIFICATION,
                session.session_id,
                claim.claim_id,
                reviewer_id.strip(),
                now.isoformat(),
                inp.revision_of,
            ),
            claim_id=claim.claim_id,
            verdict=inp.verdict,
            reasoning=inp.rationale,
            supporting_evidence_ids=inp.supporting_evidence_ids,
            contradicting_evidence_ids=inp.contradicting_evidence_ids,
            verification_method=VerificationMethod.HUMAN,
            reviewer_id=reviewer_id.strip(),
            verification_timestamp=now,
            provenance=Provenance(method_detail=METHOD_DETAIL),
            session_id=session.session_id,
            evidence_assessments=assessments,
            revision_of=inp.revision_of,
            revision_reason=inp.revision_reason if inp.revision_of else None,
            revision_authorization=inp.revision_authorization if inp.revision_of else None,
            noted_limitations=_limitations(session, inp, checks),
        )
    except ValidationError as exc:
        raise VerificationError(summarize_validation_error(exc)) from exc
    return verification, rationale_check.warnings


def record_verification(
    session: VerificationSession, verification: Verification
) -> VerificationSession:
    """Append a verification (never replacing anything) and drop the matching draft."""
    drafts = [
        d
        for d in session.verification_drafts
        if not (d.claim_id == verification.claim_id and d.reviewer_id == verification.reviewer_id)
    ]
    updated = session.model_copy(
        update={
            "verifications": [*session.verifications, verification],
            "verification_drafts": drafts,
            "updated_at": max(verification.verification_timestamp, session.updated_at),
        }
    )
    try:
        return VerificationSession.model_validate(updated.model_dump())
    except ValidationError as exc:
        raise VerificationError(summarize_validation_error(exc)) from exc


# ---------------------------------------------------------------------- drafts


def save_draft(session: VerificationSession, draft: VerificationDraft) -> VerificationSession:
    others = [
        d
        for d in session.verification_drafts
        if not (d.claim_id == draft.claim_id and d.reviewer_id == draft.reviewer_id)
    ]
    updated = session.model_copy(
        update={
            "verification_drafts": [*others, draft],
            "updated_at": max(draft.updated_at, session.updated_at),
        }
    )
    try:
        return VerificationSession.model_validate(updated.model_dump())
    except ValidationError as exc:
        raise VerificationError(summarize_validation_error(exc)) from exc


def discard_draft(
    session: VerificationSession, claim_id: str, reviewer_id: str
) -> VerificationSession:
    drafts = [
        d
        for d in session.verification_drafts
        if not (d.claim_id == claim_id and d.reviewer_id == reviewer_id)
    ]
    return session.model_copy(update={"verification_drafts": drafts})


# -------------------------------------------------------------------- progress


@dataclass(frozen=True)
class VerificationProgress:
    total_candidate_claims: int
    marked_for_verification: int
    awaiting_retrieval: int
    with_retrieval_results: int
    verified_by_human: int
    awaiting_human_review: int
    verdict_counts: dict[str, int]
    revised_records: int
    superseded_records: int
    drafts_in_progress: int
    evidence_blocked: int | None = None  # None when sources were not checked

    @property
    def complete(self) -> bool:
        return self.awaiting_human_review == 0 and self.marked_for_verification > 0


def compute_progress(
    session: VerificationSession, checks: dict[str, EvidenceCheck] | None = None
) -> VerificationProgress:
    active = session.active_claims
    marked = [c for c in active if c.requires_verification]
    verdicts = {v.value: 0 for v in Verdict}
    verified = 0
    for claim in active:
        verdict = session.human_verdict(claim.claim_id)
        if verdict is not None:
            verified += 1
            verdicts[verdict.verdict.value] += 1
    revised = sum(1 for v in session.verifications if v.revision_of)
    return VerificationProgress(
        total_candidate_claims=len(active),
        marked_for_verification=len(marked),
        awaiting_retrieval=sum(1 for c in marked if session.latest_retrieval(c.claim_id) is None),
        with_retrieval_results=sum(
            1 for c in active if session.latest_retrieval(c.claim_id) is not None
        ),
        verified_by_human=verified,
        awaiting_human_review=len(session.pending_review()),
        verdict_counts=verdicts,
        revised_records=revised,
        superseded_records=len(session.superseded_ids()),
        drafts_in_progress=len(session.verification_drafts),
        evidence_blocked=(
            sum(1 for c in checks.values() if c.blocks_citation) if checks is not None else None
        ),
    )


# --------------------------------------------------------------------- summary


def verification_summary(
    session: VerificationSession,
    checks: dict[str, EvidenceCheck] | None = None,
    now: datetime | None = None,
) -> dict:
    """Structured summary of active human verifications (JSON-serialisable)."""
    progress = compute_progress(session, checks)
    evidence = {e.evidence_id: e for e in session.evidence}
    records = []
    for claim in session.active_claims:
        v = session.human_verdict(claim.claim_id)
        if v is None:
            continue
        assessments = {a.evidence_id: a for a in v.evidence_assessments}
        used = []
        for role, ids in (
            ("supporting", v.supporting_evidence_ids),
            ("contradicting", v.contradicting_evidence_ids),
        ):
            for eid in ids:
                ev = evidence[eid]
                loc = ev.passage_location
                used.append(
                    {
                        "evidence_id": eid,
                        "role": role,
                        "source_title": ev.source_title,
                        "source_url": ev.source_url,
                        "document_reference": ev.document_reference,
                        "lines": f"{loc.line_start}-{loc.line_end}"
                        if loc and loc.line_start
                        else None,
                        "quoted_passage": ev.quoted_passage,
                        "reviewer_note": assessments[eid].explanation
                        if eid in assessments
                        else None,
                        "source_check": checks[eid].status.value
                        if checks and eid in checks
                        else None,
                    }
                )
        chain = session.revision_chain(v.verification_id)
        records.append(
            {
                "claim_id": claim.claim_id,
                "claim_text": claim.normalized_claim,
                "original_text": claim.original_text,
                "verdict": v.verdict.value,
                "rationale": v.reasoning,
                "reviewer_id": v.reviewer_id,
                "verification_id": v.verification_id,
                "verification_timestamp": v.verification_timestamp.isoformat(),
                "evidence_used": used,
                "passages_assessed": {
                    label.value: sum(1 for a in v.evidence_assessments if a.label is label)
                    for label in EvidenceLabel
                },
                "revision_number": len(chain),
                "revision_of": v.revision_of,
                "revision_reason": v.revision_reason,
                "revision_authorization": v.revision_authorization,
                "noted_limitations": v.noted_limitations,
                "legacy_record": v.session_id is None,
            }
        )
    return {
        "label": "human_verification_records",
        "note": "Verdicts are human judgements of retrieved evidence; retrieval scores are "
        "not part of them. Superseded revisions are kept in the session file.",
        "schema_version": session.schema_version,
        "session_id": session.session_id,
        "title": session.title,
        "generated_at": (now or utc_now()).isoformat(),
        "complete": progress.complete,
        "progress": progress.__dict__,
        "records": records,
    }
