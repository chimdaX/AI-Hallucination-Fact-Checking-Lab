"""CLI commands for human verification.

hallab verify --session ID --reviewer-id NAME [--claim-id ID ...]
hallab verify --session ID --reviewer-id NAME --revise VERIFICATION_ID [--authorization TEXT]
hallab verify status  --session ID
hallab verify summary --session ID [--json]
"""

from __future__ import annotations

import argparse
import json
import textwrap

from hallucination_lab.config import Settings
from hallucination_lab.errors import HallucinationLabError
from hallucination_lab.models import VerificationSession
from hallucination_lab.retrieval.catalog import CollectionCatalog
from hallucination_lab.storage import JsonSessionStore
from hallucination_lab.verification.integrity import check_session_evidence
from hallucination_lab.verification.interactive import Prompter, run_verification
from hallucination_lab.verification.workflow import compute_progress, verification_summary

VERIFY_DESCRIPTION = """\
Human verification of extracted claims.

  hallab verify --session ID --reviewer-id NAME      review claims awaiting verification
  hallab verify status --session ID                  progress of the session
  hallab verify summary --session ID [--json]        verdicts with evidence and rationales

You assess each retrieved passage (supporting / contradicting / irrelevant /
uncertain), choose a verdict, cite the passages that justify it and write a
rationale. Nothing is saved without your confirmation; progress is kept as a
draft so you can quit (q) and resume. Corrections never overwrite: use --revise
to create a new record linked to the old one.
Guide: docs/HUMAN_VERIFICATION_GUIDE.md"""

# Tests replace this to script reviewer input.
PROMPTER_FACTORY = Prompter


def _wrap(text: str, indent: str = "      ") -> str:
    return textwrap.fill(
        " ".join(text.split()), 92, initial_indent=indent, subsequent_indent=indent
    )


def _load(args: argparse.Namespace) -> tuple[JsonSessionStore, VerificationSession]:
    settings = Settings.from_env()
    store = JsonSessionStore(settings.sessions_dir)
    return store, store.load(args.session)


def _print_status(session: VerificationSession, catalog: CollectionCatalog) -> None:
    checks = check_session_evidence(session, catalog)
    p = compute_progress(session, checks)
    print(f"VERIFICATION STATUS  {session.session_id}: {session.title}")
    if p.marked_for_verification == 0:
        print("State: NOTHING TO VERIFY (no active claims are flagged for verification)")
    elif p.complete:
        print("State: COMPLETE (every claim flagged for verification has an active human verdict)")
    else:
        print(f"State: INCOMPLETE ({p.awaiting_human_review} claim(s) still need human review)")
    rows = [
        ("Claims", None),
        ("total candidate claims (active)", p.total_candidate_claims),
        ("marked for verification", p.marked_for_verification),
        ("awaiting evidence retrieval", p.awaiting_retrieval),
        ("with retrieval results", p.with_retrieval_results),
        ("Human review", None),
        ("verified by a human (active verdicts)", p.verified_by_human),
        ("awaiting human review", p.awaiting_human_review),
        ("drafts in progress (not verifications)", p.drafts_in_progress),
        ("Verdicts (active records only)", None),
        *[(k, v) for k, v in p.verdict_counts.items()],
        ("Audit", None),
        ("revised records (corrections)", p.revised_records),
        ("superseded records kept for history", p.superseded_records),
        ("evidence with changed/missing sources", p.evidence_blocked),
    ]
    for label, value in rows:
        print(f"{label}" if value is None else f"  {label:42} {value}")
    pending = session.pending_review()
    if pending:
        print("Pending claims:")
        drafts = {d.claim_id: d.reviewer_id for d in session.verification_drafts}
        for c in pending:
            latest = session.latest_retrieval(c.claim_id)
            retrieval = latest.outcome.value if latest else "not searched"
            draft = f"; draft by {drafts[c.claim_id]}" if c.claim_id in drafts else ""
            print(f"  - {c.claim_id}  [{retrieval}{draft}] {c.normalized_claim[:70]}")


def _print_summary(summary: dict) -> None:
    print(f"VERIFICATION SUMMARY  {summary['session_id']}: {summary['title']}")
    print("Verdicts below are human judgements of retrieved evidence. Retrieval scores are not")
    print("part of them, and they cover only this session's claims and reference documents.")
    p = summary["progress"]
    counts = ", ".join(f"{k} {v}" for k, v in p["verdict_counts"].items())
    print(
        f"Verified {p['verified_by_human']} of {p['marked_for_verification']} flagged claims "
        f"({counts}); revisions: {p['revised_records']}"
    )
    if not summary["complete"]:
        print(f"INCOMPLETE: {p['awaiting_human_review']} flagged claim(s) not yet verified.")
    for n, r in enumerate(summary["records"], 1):
        print(f"\n[{n}] {r['verdict']}")
        print(_wrap(r["claim_text"], "    "))
        if not r["evidence_used"]:
            print("    Evidence used: none cited")
        for e in r["evidence_used"]:
            lines = f" lines {e['lines']}" if e["lines"] else ""
            print(f"    Evidence ({e['role']}): {e['source_title']}{lines} [{e['evidence_id']}]")
            if e["source_url"]:
                print(f"      {e['source_url']}")
            if e["reviewer_note"]:
                print(f"      note: {e['reviewer_note']}")
        print(_wrap(f"Rationale: {r['rationale']}", "    "))
        rev = (
            f"; revision {r['revision_number']} (revises {r['revision_of']}: "
            f"{r['revision_reason']})"
            if r["revision_of"]
            else ""
        )
        print(
            f"    Reviewer {r['reviewer_id']} at {r['verification_timestamp']} "
            f"[{r['verification_id']}]{rev}"
        )
        if r["legacy_record"]:
            print("    (legacy record created before evidence assessments existed)")
        for note in r["noted_limitations"]:
            print(_wrap(f"Limitation: {note}", "    "))


def cmd_verify(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    catalog = CollectionCatalog(settings)
    if args.action == "status":
        _, session = _load(args)
        _print_status(session, catalog)
        return 0
    if args.action == "summary":
        _, session = _load(args)
        summary = verification_summary(session, check_session_evidence(session, catalog))
        if args.json:
            print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
        else:
            _print_summary(summary)
        return 0
    if not args.reviewer_id:
        raise HallucinationLabError("--reviewer-id is required to record verifications")
    if args.authorization and not args.revise:
        raise HallucinationLabError("--authorization is only used together with --revise")
    run_verification(
        JsonSessionStore(settings.sessions_dir),
        args.session,
        args.reviewer_id,
        catalog=catalog,
        prompter=PROMPTER_FACTORY(),
        claim_ids=args.claim_id,
        revise=args.revise,
        authorization=args.authorization,
    )
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "verify",
        help="human verification of claims (interactive), status and summary",
        description=VERIFY_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "action",
        nargs="?",
        default="review",
        choices=["review", "status", "summary"],
        help="review (default), status, or summary",
    )
    p.add_argument("--session", required=True, help="session ID")
    p.add_argument("--reviewer-id", help="your reviewer identifier (required to review)")
    p.add_argument("--claim-id", action="append", help="review only this claim (repeatable)")
    p.add_argument(
        "--revise",
        metavar="VERIFICATION_ID",
        help="correct an existing verification (creates a new linked record)",
    )
    p.add_argument(
        "--authorization",
        help="required when revising another reviewer's verification: who approved it and why",
    )
    p.add_argument("--json", action="store_true", help="summary as JSON")
    p.set_defaults(func=cmd_verify)
