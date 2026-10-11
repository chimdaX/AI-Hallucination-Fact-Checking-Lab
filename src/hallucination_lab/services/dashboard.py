"""Aggregate statistics across saved sessions, computed from the session files.

Nothing here is estimated or hardcoded. Active verdicts are the latest
unsuperseded human records, so revisions are never double-counted. Statistics
are split by response provenance so illustrative fixtures are never mixed up
with user-submitted or authentic model outputs.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from hallucination_lab.models import ResponseProvenance, Verdict, VerificationSession
from hallucination_lab.storage import JsonSessionStore
from hallucination_lab.verification.workflow import compute_progress

PROVENANCE_LABELS = {
    ResponseProvenance.SYNTHETIC_FIXTURE.value: "Illustrative fixture",
    ResponseProvenance.USER_SUBMITTED.value: "User-submitted",
    ResponseProvenance.AUTHENTIC_MODEL_OUTPUT.value: "Authentic model output",
}


@dataclass
class Totals:
    sessions: int = 0
    candidate_claims: int = 0
    flagged: int = 0
    verified: int = 0
    pending: int = 0
    awaiting_retrieval: int = 0
    drafts: int = 0
    revisions: int = 0
    superseded: int = 0
    legacy_records: int = 0
    evidence_retrieved: int = 0
    evidence_assessed: int = 0
    evidence_cited: int = 0
    verdicts: Counter = field(default_factory=lambda: Counter({v.value: 0 for v in Verdict}))

    def add(self, session: VerificationSession) -> None:
        p = compute_progress(session)
        self.sessions += 1
        self.candidate_claims += p.total_candidate_claims
        self.flagged += p.marked_for_verification
        self.verified += p.verified_by_human
        self.pending += p.awaiting_human_review
        self.awaiting_retrieval += p.awaiting_retrieval
        self.drafts += p.drafts_in_progress
        self.revisions += p.revised_records
        self.superseded += p.superseded_records
        self.verdicts.update(p.verdict_counts)
        self.evidence_retrieved += len(session.evidence)
        for claim in session.active_claims:
            v = session.human_verdict(claim.claim_id)
            if v is None:
                continue
            self.legacy_records += v.session_id is None
            self.evidence_assessed += len(v.evidence_assessments)
            self.evidence_cited += len(v.cited_evidence_ids)


@dataclass
class DashboardStats:
    totals: Totals
    by_provenance: dict[str, Totals]
    sessions: list[dict]
    unreadable: list[str]

    @property
    def empty(self) -> bool:
        return self.totals.sessions == 0


def session_row(session: VerificationSession) -> dict:
    p = compute_progress(session)
    return {
        "session_id": session.session_id,
        "title": session.title,
        "data": PROVENANCE_LABELS[session.response.provenance.value],
        "updated": session.updated_at.strftime("%Y-%m-%d %H:%M"),
        "claims": p.total_candidate_claims,
        "flagged": p.marked_for_verification,
        "verified": p.verified_by_human,
        "pending": p.awaiting_human_review,
        "status": (
            "nothing flagged"
            if p.marked_for_verification == 0
            else "complete"
            if p.complete
            else "incomplete"
        ),
        **{v: p.verdict_counts[v] for v in p.verdict_counts},
        "revisions": p.revised_records,
    }


def compute_dashboard(store: JsonSessionStore) -> DashboardStats:
    summaries, problems = store.list_sessions()
    totals = Totals()
    by_provenance: dict[str, Totals] = {}
    rows = []
    for summary in summaries:
        try:
            session = store.load(summary.session_id)
        except Exception as exc:  # file changed between listing and loading
            problems.append(f"{summary.session_id}: {exc}")
            continue
        totals.add(session)
        by_provenance.setdefault(session.response.provenance.value, Totals()).add(session)
        rows.append(session_row(session))
    return DashboardStats(totals, by_provenance, rows, problems)
