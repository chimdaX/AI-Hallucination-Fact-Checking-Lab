"""Verification reports (JSON, CSV, Markdown) built from validated session records.

Report guarantees:

* **Source of truth:** the report is derived only from a schema-validated
  :class:`VerificationSession`. No value is estimated or invented. Timestamps
  are the ones stored in the records, plus one ``generated_at``.
* **Active vs history:** each claim shows its active human verdict (the latest
  unsuperseded record) and, separately, the full revision history. Superseded
  records are preserved but never counted as additional verifications.
* **Candidates vs assessed evidence:** retrieval candidates (lexical matches)
  are listed apart from the passages a reviewer assessed and cited.
* **Disclosure:** unresolved claims, document integrity warnings and
  methodological limitations are always included.
* **Reproducibility:** output is deterministic for the same session.
  ``content_sha256`` covers everything except ``generated_at``, so two exports
  of unchanged data have the same fingerprint.

No accuracy or "hallucination rate" figures are produced. Verdict counts
describe the reviewed claims in one session only.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from datetime import datetime

from hallucination_lab import __version__
from hallucination_lab.models import (
    Claim,
    EvidenceLabel,
    ResponseProvenance,
    Verification,
    VerificationSession,
    utc_now,
)
from hallucination_lab.verification.integrity import EvidenceCheck, SourceStatus
from hallucination_lab.verification.workflow import compute_progress

REPORT_FORMAT = "hallab-verification-report"
REPORT_VERSION = "1.0"

STANDARD_LIMITATIONS = (
    "Verdicts are human judgements based only on the passages retrieved from the named "
    "reference collection(s); evidence outside those documents was not considered.",
    "Claim extraction is rule-based and may have omitted or misclassified claims in the "
    "original response.",
    "Retrieval is lexical (BM25): relevant evidence can be missed when wording differs, and "
    "retrieval scores measure word overlap, not support or truth.",
    "The tool does not assess source credibility; reliability is a reviewer judgement and "
    "is noted per verdict where known.",
    "Counts describe the claims reviewed in this session only. They are not an accuracy "
    "metric and not a general LLM hallucination rate.",
    "Reviewer identifiers are self-declared and not authenticated.",
)

REVIEW_STATUS_TEXT = {
    "verified": "Human verdict recorded",
    "draft_in_progress": "Draft in progress (no verdict)",
    "awaiting_review": "Awaiting human review",
    "awaiting_retrieval": "Awaiting evidence retrieval",
    "not_flagged": "Not flagged for verification",
    "inactive": "Inactive (removed, split or merged)",
}


def review_status(session: VerificationSession, claim: Claim) -> str:
    if not claim.is_active:
        return "inactive"
    if session.human_verdict(claim.claim_id) is not None:
        return "verified"
    if not claim.requires_verification:
        return "not_flagged"
    if any(d.claim_id == claim.claim_id for d in session.verification_drafts):
        return "draft_in_progress"
    if session.latest_retrieval(claim.claim_id) is None:
        return "awaiting_retrieval"
    return "awaiting_review"


def _record(v: Verification, superseded: set[str]) -> dict:
    return {
        "verification_id": v.verification_id,
        "status": "superseded" if v.verification_id in superseded else "active",
        "verdict": v.verdict.value,
        "method": v.verification_method.value,
        "reviewer_id": v.reviewer_id,
        "verification_timestamp": v.verification_timestamp.isoformat(),
        "rationale": v.reasoning,
        "revision_of": v.revision_of,
        "revision_reason": v.revision_reason,
        "revision_authorization": v.revision_authorization,
    }


def _verdict_block(
    session: VerificationSession, v: Verification, checks: dict[str, EvidenceCheck]
) -> dict:
    evidence = {e.evidence_id: e for e in session.evidence}
    assessments = {a.evidence_id: a for a in v.evidence_assessments}
    cited = []
    for role, ids in (
        ("supporting", v.supporting_evidence_ids),
        ("contradicting", v.contradicting_evidence_ids),
    ):
        for eid in ids:
            ev = evidence[eid]
            loc = ev.passage_location
            cited.append(
                {
                    "evidence_id": eid,
                    "role": role,
                    "source_id": ev.source_id,
                    "source_title": ev.source_title,
                    "source_url": ev.source_url,
                    "document_reference": ev.document_reference,
                    "section": loc.section if loc else None,
                    "lines": f"{loc.line_start}-{loc.line_end}" if loc and loc.line_start else None,
                    "document_version": ev.document_version,
                    "publication_date": ev.publication_date.isoformat()
                    if ev.publication_date
                    else None,
                    "passage_sha256": ev.passage_sha256,
                    "quoted_passage": ev.quoted_passage,
                    "reviewer_note": assessments[eid].explanation if eid in assessments else None,
                    "source_check": checks[eid].status.value if eid in checks else "not_checked",
                }
            )
    chain = session.revision_chain(v.verification_id)
    return {
        **_record(v, set()),
        "revision_number": len(chain),
        "cited_evidence": cited,
        "assessment_counts": {
            label.value: sum(1 for a in v.evidence_assessments if a.label is label)
            for label in EvidenceLabel
        },
        "noted_limitations": list(v.noted_limitations),
        "legacy_record": v.session_id is None,
    }


def build_report(
    session: VerificationSession,
    checks: dict[str, EvidenceCheck] | None = None,
    generated_at: datetime | None = None,
) -> dict:
    """Assemble the report dictionary (the canonical form behind every export)."""
    session = VerificationSession.model_validate(session.model_dump())  # re-validate
    checks = checks or {}
    superseded = session.superseded_ids()
    progress = compute_progress(session, checks or None)
    evidence = {e.evidence_id: e for e in session.evidence}

    claims = []
    unresolved = []
    for claim in session.claims:
        status = review_status(session, claim)
        active = session.human_verdict(claim.claim_id) if claim.is_active else None
        assessed = (
            {a.evidence_id: a.label.value for a in active.evidence_assessments} if active else {}
        )
        latest = session.latest_retrieval(claim.claim_id)
        candidates = []
        if latest:
            for rank, eid in enumerate(latest.evidence_ids, 1):
                ev = evidence[eid]
                candidates.append(
                    {
                        "evidence_id": eid,
                        "rank": rank,
                        "bm25_score": ev.retrieval_score,
                        "source_title": ev.source_title,
                        "source_url": ev.source_url,
                        "document_reference": ev.document_reference,
                        "reviewer_assessment": assessed.get(eid, "not assessed"),
                    }
                )
        history = [
            _record(v, superseded)
            for v in session.verifications_for(claim.claim_id)
            if v.is_human_judgment
        ]
        automated = [
            _record(v, superseded)
            for v in session.verifications_for(claim.claim_id)
            if not v.is_human_judgment
        ]
        entry = {
            "claim_id": claim.claim_id,
            "normalized_claim": claim.normalized_claim,
            "original_text": claim.original_text,
            "span": [claim.span.start, claim.span.end] if claim.span else None,
            "extraction_method": claim.extraction_method.value,
            "active": claim.is_active,
            "flagged_for_verification": claim.requires_verification,
            "non_checkable_reason": (
                claim.non_checkable_reason.value if claim.non_checkable_reason else None
            ),
            "reviewer_edits": len(claim.reviewer_edits),
            "review_status": status,
            "retrieval": {
                "outcome": latest.outcome.value if latest else "not_searched",
                "query_terms": list(latest.query_terms) if latest else [],
                "candidates": candidates,
                "note": "Retrieval candidates are lexical matches, not verified evidence.",
            },
            "human_verdict": _verdict_block(session, active, checks) if active else None,
            "revision_history": history,
            "automated_suggestions": automated,
        }
        claims.append(entry)
        if status in {"awaiting_review", "awaiting_retrieval", "draft_in_progress"}:
            unresolved.append(
                {
                    "claim_id": claim.claim_id,
                    "claim": claim.normalized_claim,
                    "reason": REVIEW_STATUS_TEXT[status],
                }
            )
        elif active and active.verdict.value == "INSUFFICIENT_EVIDENCE":
            unresolved.append(
                {
                    "claim_id": claim.claim_id,
                    "claim": claim.normalized_claim,
                    "reason": "Verified as INSUFFICIENT_EVIDENCE: the available "
                    "evidence did not settle the claim",
                }
            )

    cited_ids = {
        eid
        for c in claims
        if c["human_verdict"]
        for eid in (e["evidence_id"] for e in c["human_verdict"]["cited_evidence"])
    }
    integrity = [
        {
            "evidence_id": eid,
            "claim_id": evidence[eid].claim_id,
            "source_title": evidence[eid].source_title,
            "status": check.status.value,
            "detail": check.detail,
            "cited_by_active_verdict": eid in cited_ids,
        }
        for eid, check in sorted(checks.items())
        if check.status is not SourceStatus.VERIFIED
    ]
    limitations = list(STANDARD_LIMITATIONS)
    if session.response.provenance is ResponseProvenance.SYNTHETIC_FIXTURE:
        limitations.insert(
            0,
            "The analysed response is an illustrative synthetic fixture, not "
            "an output captured from an AI model.",
        )
    elif session.response.provenance is ResponseProvenance.USER_SUBMITTED:
        limitations.insert(
            0,
            "The analysed response was pasted by a user; its origin (which "
            "model, which prompt) is not independently known.",
        )
    if not checks:
        limitations.append("Evidence sources were not re-checked when this report was built.")
    if any(c["human_verdict"] and c["human_verdict"]["legacy_record"] for c in claims):
        limitations.append(
            "Some verdicts are legacy records created before per-passage "
            "evidence assessments existed."
        )

    body = {
        "report_format": REPORT_FORMAT,
        "report_version": REPORT_VERSION,
        "generated_by": f"hallucination_lab {__version__}",
        "judgement_type": "Verdicts are human judgements. Retrieval and extraction are "
        "automated; no automated verdicts are included unless listed under "
        "automated_suggestions.",
        "session": {
            "session_id": session.session_id,
            "title": session.title,
            "schema_version": session.schema_version,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
        },
        "response": {
            "response_id": session.response.response_id,
            "provenance": session.response.provenance.value,
            "model_name": session.response.model_name,
            "text": session.response.text,
        },
        "summary": {
            "complete": progress.complete,
            "total_candidate_claims": progress.total_candidate_claims,
            "flagged_for_verification": progress.marked_for_verification,
            "verified_by_human": progress.verified_by_human,
            "awaiting_human_review": progress.awaiting_human_review,
            "awaiting_retrieval": progress.awaiting_retrieval,
            "drafts_in_progress": progress.drafts_in_progress,
            "active_verdicts": progress.verdict_counts,
            "revisions": progress.revised_records,
            "superseded_records_preserved": progress.superseded_records,
            "note": "Counts cover this session only and are not an accuracy metric or a "
            "hallucination rate.",
        },
        "claims": claims,
        "unresolved_claims": unresolved,
        "integrity_warnings": integrity,
        "limitations": limitations,
    }
    fingerprint = hashlib.sha256(
        json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return {
        **body,
        "content_sha256": fingerprint,
        "generated_at": (generated_at or utc_now()).isoformat(),
    }


# -------------------------------------------------------------------- renderers


def to_json(report: dict) -> str:
    return json.dumps(report, indent=2, ensure_ascii=False) + "\n"


CSV_COLUMNS = [
    "session_id",
    "claim_id",
    "claim",
    "original_text",
    "active",
    "flagged",
    "review_status",
    "verdict",
    "reviewer_id",
    "verification_id",
    "verification_timestamp",
    "revision_number",
    "superseded_records",
    "rationale",
    "cited_evidence_ids",
    "cited_sources",
    "retrieval_outcome",
    "candidates_retrieved",
    "candidates_assessed",
    "integrity_warnings",
    "noted_limitations",
]


def to_csv(report: dict) -> str:
    """One row per claim. Lists inside a cell are separated by " | "."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    warnings_by_claim: dict[str, list[str]] = {}
    for w in report["integrity_warnings"]:
        warnings_by_claim.setdefault(w["claim_id"], []).append(f"{w['evidence_id']}: {w['status']}")
    for c in report["claims"]:
        v = c["human_verdict"] or {}
        cited = v.get("cited_evidence", [])
        candidates = c["retrieval"]["candidates"]
        writer.writerow(
            {
                "session_id": report["session"]["session_id"],
                "claim_id": c["claim_id"],
                "claim": c["normalized_claim"],
                "original_text": c["original_text"],
                "active": c["active"],
                "flagged": c["flagged_for_verification"],
                "review_status": c["review_status"],
                "verdict": v.get("verdict", ""),
                "reviewer_id": v.get("reviewer_id", ""),
                "verification_id": v.get("verification_id", ""),
                "verification_timestamp": v.get("verification_timestamp", ""),
                "revision_number": v.get("revision_number", ""),
                "superseded_records": sum(
                    r["status"] == "superseded" for r in c["revision_history"]
                ),
                "rationale": v.get("rationale", ""),
                "cited_evidence_ids": " | ".join(e["evidence_id"] for e in cited),
                "cited_sources": " | ".join(
                    f"{e['role']}: {e['source_title']} "
                    f"({e['source_url'] or e['document_reference']})"
                    for e in cited
                ),
                "retrieval_outcome": c["retrieval"]["outcome"],
                "candidates_retrieved": len(candidates),
                "candidates_assessed": sum(
                    x["reviewer_assessment"] != "not assessed" for x in candidates
                ),
                "integrity_warnings": " | ".join(warnings_by_claim.get(c["claim_id"], [])),
                "noted_limitations": " | ".join(v.get("noted_limitations", [])),
            }
        )
    return buffer.getvalue()


def _md(text: str | None) -> str:
    return (text or "").replace("|", "\\|").replace("\n", " ")


def _quote(text: str) -> str:
    return "\n".join(f"> {line}" if line.strip() else ">" for line in text.splitlines())


def to_markdown(report: dict) -> str:
    s, summary = report["session"], report["summary"]
    lines = [
        f"# Verification report: {s['title']}",
        "",
        f"- Session: `{s['session_id']}` (schema {s['schema_version']})",
        f"- Response provenance: **{report['response']['provenance']}**",
        f"- Generated: {report['generated_at']} by {report['generated_by']}",
        f"- Content fingerprint (excludes generation time): `{report['content_sha256']}`",
        "",
        f"> {report['judgement_type']}",
        "",
        "## Summary",
        "",
        f"**Status: {'COMPLETE' if summary['complete'] else 'INCOMPLETE'}.** "
        f"{summary['verified_by_human']} of {summary['flagged_for_verification']} flagged "
        f"claims have an active human verdict; {summary['awaiting_human_review']} await review.",
        "",
        "| Verdict | Claims |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in summary["active_verdicts"].items()],
        "",
        f"Revisions: {summary['revisions']} (superseded records preserved: "
        f"{summary['superseded_records_preserved']}). {summary['note']}",
        "",
        "## Original AI response",
        "",
        _quote(report["response"]["text"]),
        "",
        "## Claims",
        "",
    ]
    for n, c in enumerate(report["claims"], 1):
        v = c["human_verdict"]
        heading = v["verdict"] if v else REVIEW_STATUS_TEXT[c["review_status"]]
        lines += [f"### {n}. {heading}: {c['normalized_claim']}", ""]
        lines.append(f"- Claim ID: `{c['claim_id']}`; status: {c['review_status']}")
        if c["original_text"] != c["normalized_claim"]:
            lines.append(f'- Original wording: "{_md(c["original_text"])}"')
        if c["non_checkable_reason"]:
            lines.append(f"- Extractor flag: {c['non_checkable_reason']}")
        if v:
            lines += [
                f"- Reviewer: {v['reviewer_id']} at {v['verification_timestamp']} "
                f"(`{v['verification_id']}`, revision {v['revision_number']})",
                f"- Rationale: {_md(v['rationale'])}",
                f"- Passages assessed: {v['assessment_counts']}",
            ]
            for e in v["cited_evidence"]:
                where = e["source_url"] or e["document_reference"] or ""
                lines += [
                    "",
                    f"**Cited ({e['role']})**: {e['source_title']}, {e['document_reference']}"
                    + (f" (section: {e['section']})" if e["section"] else ""),
                    f"`{e['evidence_id']}`; source check: {e['source_check']}; {where}",
                    "",
                    _quote(e["quoted_passage"]),
                ]
                if e["reviewer_note"]:
                    lines += ["", f"Reviewer note: {_md(e['reviewer_note'])}"]
            for note in v["noted_limitations"]:
                lines.append(f"- Limitation: {_md(note)}")
        candidates = c["retrieval"]["candidates"]
        lines += [
            "",
            f"Retrieval: {c['retrieval']['outcome']}; {len(candidates)} candidate(s) "
            "(lexical matches, not verified evidence).",
        ]
        if candidates:
            lines += [
                "",
                "| Rank | Source | BM25 (relevance only) | Reviewer assessment |",
                "|---|---|---|---|",
            ]
            lines += [
                f"| {x['rank']} | {_md(x['source_title'])} {_md(x['document_reference'])}"
                f" | {x['bm25_score']} | {x['reviewer_assessment']} |"
                for x in candidates
            ]
        if len(c["revision_history"]) > 1:
            lines += ["", "Revision history (oldest first):", ""]
            lines += [
                f"- `{r['verification_id']}` {r['status']}: {r['verdict']} by "
                f"{r['reviewer_id']} at {r['verification_timestamp']}"
                + (f"; reason: {_md(r['revision_reason'])}" if r["revision_reason"] else "")
                for r in c["revision_history"]
            ]
        if c["automated_suggestions"]:
            lines += ["", "Automated suggestions (not human verdicts):", ""]
            lines += [f"- {r['verdict']} ({r['method']})" for r in c["automated_suggestions"]]
        lines.append("")
    lines += ["## Unresolved claims", ""]
    lines += [
        f"- `{u['claim_id']}` {u['claim']}: {u['reason']}" for u in report["unresolved_claims"]
    ] or ["None."]
    lines += ["", "## Document integrity warnings", ""]
    lines += [
        f"- `{w['evidence_id']}` ({w['source_title']}): {w['status']}, {w['detail']}"
        + (" **[cited by an active verdict]**" if w["cited_by_active_verdict"] else "")
        for w in report["integrity_warnings"]
    ] or ["None."]
    lines += ["", "## Methodological limitations", ""]
    lines += [f"- {item}" for item in report["limitations"]]
    return "\n".join(lines) + "\n"


EXPORTERS = {"json": to_json, "csv": to_csv, "md": to_markdown}
MEDIA_TYPES = {"json": "application/json", "csv": "text/csv", "md": "text/markdown"}


def export(report: dict, fmt: str) -> str:
    try:
        return EXPORTERS[fmt](report)
    except KeyError as exc:
        raise ValueError(f"unknown report format {fmt!r}; use json, csv or md") from exc


def report_filename(report: dict, fmt: str) -> str:
    return f"hallab-report-{report['session']['session_id']}.{fmt}"
