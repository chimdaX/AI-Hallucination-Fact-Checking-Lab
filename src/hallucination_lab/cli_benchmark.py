"""CLI: ``hallab benchmark ...``. Managing and evaluating the ground-truth benchmark.

status | list | show ITEM | add | mark ITEM | review ITEM | adjudicate ITEM | approve ITEM
validate | check-sources | export | evaluate | retrieval
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from dataclasses import asdict
from pathlib import Path

from hallucination_lab.benchmark import workflow as wf
from hallucination_lab.benchmark.evaluation import load_predictions, run_evaluation
from hallucination_lab.benchmark.integrity import check_sources
from hallucination_lab.benchmark.models import (
    AuthorType,
    BenchmarkItem,
    CandidateProvenance,
    ClaimType,
    Difficulty,
    Domain,
    EvidenceJudgement,
    Partition,
    Reliability,
    ReviewKind,
    ReviewStatus,
)
from hallucination_lab.benchmark.retrieval_analysis import analyse_retrieval
from hallucination_lab.benchmark.store import BenchmarkError, BenchmarkStore
from hallucination_lab.config import Settings
from hallucination_lab.errors import HallucinationLabError
from hallucination_lab.models import EvidenceLabel, Verdict
from hallucination_lab.retrieval.catalog import CollectionCatalog
from hallucination_lab.verification.interactive import Prompter
from hallucination_lab.verification.rationale import check_rationale

PROMPTER_FACTORY = Prompter  # tests substitute a scripted prompter
PROVISIONAL_NOTICE = (
    "Benchmark under development: candidates and proposed labels are "
    "AI-authored and provisional; only APPROVED items are ground truth."
)


def _store() -> BenchmarkStore:
    return BenchmarkStore()


def _wrap(text: str, indent: str = "    ") -> str:
    return textwrap.fill(
        " ".join(text.split()), 92, initial_indent=indent, subsequent_indent=indent
    )


# ------------------------------------------------------------------------ views


def cmd_status(args) -> int:
    summary = wf.status_summary(_store())
    print(PROVISIONAL_NOTICE)
    print(f"\nItems: {summary['total']}")
    for key in ("by_status", "by_partition", "by_domain", "by_difficulty"):
        print(
            f"  {key.replace('_', ' ')}: " + ", ".join(f"{k} {v}" for k, v in summary[key].items())
        )
    print(
        f"\nHuman-APPROVED labels: {sum(summary['approved_labels'].values())} "
        f"({summary['approved_labels'] or 'none yet'})"
    )
    print(f"Proposed labels (AI, NOT ground truth): {summary['proposed_labels_not_ground_truth']}")
    print(
        f"Pending review: {len(summary['pending_review'])}; disputed: "
        f"{len(summary['disputed'])}; open research tasks: {summary['open_research_tasks']}"
    )
    for problem in summary["unreadable"]:
        print(f"WARNING: {problem}", file=sys.stderr)
    return 0


def _matches(item: BenchmarkItem, args) -> bool:
    return (
        (not args.status or item.status.value == args.status.upper())
        and (not args.domain or item.domain.value == args.domain)
        and (not args.difficulty or item.difficulty.value == args.difficulty)
        and (not args.partition or item.partition.value == args.partition)
    )


def cmd_list(args) -> int:
    items, problems = _store().load_items()
    shown = [i for i in items if _matches(i, args)]
    for i in shown:
        label = i.current_approval.final_verdict.value if i.current_approval else "-"
        print(
            f"{i.item_id}  {i.status.value:17} {i.partition.value:4} {i.domain.value:22} "
            f"{i.difficulty.value:6} approved={label:22} {i.claim_text[:60]}"
        )
    print(f"{len(shown)} of {len(items)} item(s)", file=sys.stderr)
    for p in problems:
        print(f"WARNING: {p}", file=sys.stderr)
    return 0


def _show_item(store: BenchmarkStore, item: BenchmarkItem, reveal: bool, out=print) -> None:
    refs = store.references()
    out(f"{item.item_id}  [{item.status.value}]  partition={item.partition.value}")
    out(_wrap(f"Claim: {item.claim_text}", "  "))
    if item.claim_text != item.original_claim_text:
        out(_wrap(f"Original wording: {item.original_claim_text}", "  "))
    out(
        f"  Domain: {item.domain.value}; difficulty: {item.difficulty.value}; types: "
        + ", ".join(t.value for t in item.claim_types)
    )
    if item.temporal_scope:
        out(f"  Temporal scope: {item.temporal_scope}")
    if item.original_context:
        out(_wrap(f"Original context: {item.original_context}", "  "))
    if item.split_recommendation and item.split_recommendation.should_split:
        out("  Split recommended: " + " / ".join(item.split_recommendation.proposed_parts))
    out(
        f"  Candidate by {item.candidate_provenance.author} "
        f"({item.candidate_provenance.author_type.value}): {item.candidate_provenance.method}"
    )
    for n, e in enumerate(item.evidence, 1):
        ref = refs.get(e.reference_id)
        out(f"  Evidence {n} [{e.evidence_id}] ({e.capture_status.value})")
        if ref:
            out(
                f"    Source: {ref.title} | {ref.author_or_organization or '-'} | "
                f"{ref.source_type.value}"
            )
            out(
                f"    URL: {ref.url or '-'}  accessed {ref.accessed_on or '-'}  "
                f"version {ref.document_version or '-'}"
            )
            out(f"    Licence: {ref.license or 'unknown'} ({ref.redistribution.value})")
            if ref.reliability_notes:
                out(_wrap(f"Reliability: {ref.reliability_notes}", "    "))
        out(f"    Locator: {e.locator}")
        out(
            _wrap(f"Passage: {e.passage_text}", "    ")
            if e.passage_text
            else "    Passage text not stored (licence); open the URL and use the locator. "
            f"sha256 {(e.passage_sha256 or '')[:16]}..."
        )
    for s in item.research_log:
        out(
            _wrap(
                f"Search ({s.service}, {s.performed_at:%Y-%m-%d}): '{s.query}'. {s.result_summary}",
                "  ",
            )
        )
    for t in item.research_tasks:
        out(f"  Research task [{'done' if t.resolved else 'open'}]: {t.task} ({t.reason})")
    if reveal and item.proposed_label:
        p = item.proposed_label
        out(
            f"  AI-PROPOSED label (not ground truth): {p.verdict.value} by {p.proposed_by}: "
            f"{p.rationale}"
        )
    elif item.proposed_label:
        out(
            "  (An AI-proposed label exists; it is hidden to avoid biasing reviewers. "
            "Use --reveal-proposed to show it.)"
        )
    for a in item.approvals:
        out(
            f"  Approval {a.approval_id}: {a.final_verdict.value} ({a.review_level.value}) by "
            f"{a.approved_by} at {a.approved_at:%Y-%m-%d %H:%M}"
        )
    out("  History: " + " -> ".join(f"{e.to_status.value}({e.actor})" for e in item.status_events))


def cmd_show(args) -> int:
    store = _store()
    item = store.load_item(args.item_id)
    _show_item(store, item, args.reveal_proposed)
    for a in store.annotations(item.item_id):
        print(
            f"  Review {a.annotation_id} by {a.reviewer_id} ({a.review_kind.value}): "
            f"{a.verdict.value}" + (f" [revises {a.revision_of}]" if a.revision_of else "")
        )
        print(_wrap(f"Rationale: {a.rationale}", "      "))
    for d in store.adjudications(item.item_id):
        print(f"  Adjudication {d.adjudication_id} by {d.adjudicator_id}: {d.final_verdict.value}")
    decision = wf.decide_label(
        item, store.annotations(item.item_id), store.adjudications(item.item_id)
    )
    print(
        f"  Current human label: {decision.verdict.value if decision.verdict else 'none'}"
        + (f" ({decision.problem})" if decision.problem else "")
    )
    return 0


# ------------------------------------------------------------------ management


def cmd_add(args) -> int:
    author = wf.require_human(args.author_id, "adding a candidate from the CLI")
    from hallucination_lab.models import utc_now

    item, warnings = wf.add_candidate(
        _store(),
        claim_text=args.claim,
        domain=Domain(args.domain),
        difficulty=Difficulty(args.difficulty),
        claim_types=[ClaimType(t) for t in args.type],
        temporal_scope=args.scope,
        original_context=args.context,
        provenance=CandidateProvenance(
            author=author, author_type=AuthorType.HUMAN, created_at=utc_now(), method=args.method
        ),
    )
    print(f"Added {item.item_id} (CANDIDATE, partition {item.partition.value}).")
    for w in warnings:
        print(f"  near-duplicate warning: {w}")
    print(
        "Attach evidence with the Streamlit Benchmark page or scripts, then mark it "
        f"READY_FOR_REVIEW: hallab benchmark mark {item.item_id} --status READY_FOR_REVIEW ..."
    )
    return 0


def cmd_mark(args) -> int:
    item = wf.set_status(
        _store(), args.item_id, ReviewStatus(args.status.upper()), args.actor, args.reason
    )
    print(f"{item.item_id} is now {item.status.value}.")
    return 0


# ------------------------------------------------------------------- reviewing


class _Quit(Exception):
    pass


def _ask(io, prompt: str) -> str:
    try:
        answer = io.ask(prompt).strip()
    except EOFError as exc:
        raise _Quit from exc
    if answer.lower() == "q":
        raise _Quit
    return answer


def _choose(io, prompt: str, options: dict[str, object]) -> object:
    while True:
        answer = _ask(io, prompt).lower()
        if answer in options:
            return options[answer]
        io.say(f"  Please type one of: {', '.join(options)} (q quits without saving).")


def _yes(io, prompt: str) -> bool:
    return _choose(io, prompt, {"y": True, "n": False}) is True


def interactive_review(
    store: BenchmarkStore,
    item_id: str,
    reviewer_id: str,
    io,
    kind: ReviewKind | None = None,
    revise: str | None = None,
):
    reviewer_id = wf.require_human(reviewer_id, "a review")
    item = store.load_item(item_id)
    annotations = store.annotations(item_id)
    current = wf.current_annotations(item, annotations)
    if kind is None and not revise:
        if not current:
            kind = ReviewKind.FIRST_REVIEW
        elif any(a.reviewer_id == reviewer_id for a in current):
            kind = ReviewKind.REPEAT_REVIEW_SAME_REVIEWER
        else:
            kind = ReviewKind.INDEPENDENT_SECOND_REVIEW
    io.say("BENCHMARK REVIEW: your judgement only. The AI-proposed label is hidden.")
    _show_item(store, item, reveal=False, out=io.say)
    io.say(f"Review type: {kind.value if kind else 'revision of ' + revise}")
    seen = _yes(io, "Did you see the AI-proposed label for this item before this review? [y/n] ")
    judgements, cited_candidates = (
        [],
        {EvidenceLabel.SUPPORTING: [], EvidenceLabel.CONTRADICTING: []},
    )
    rel = {
        "s": EvidenceLabel.SUPPORTING,
        "c": EvidenceLabel.CONTRADICTING,
        "i": EvidenceLabel.IRRELEVANT,
        "u": EvidenceLabel.UNCERTAIN,
    }
    reliab = {
        "h": Reliability.HIGH,
        "m": Reliability.MEDIUM,
        "l": Reliability.LOW,
        "?": Reliability.UNKNOWN,
    }
    for n, e in enumerate(item.evidence, 1):
        relation = _choose(
            io, f"Evidence {n}: supports/contradicts/irrelevant/uncertain [s/c/i/u]: ", rel
        )
        reliability = _choose(
            io, f"Evidence {n}: source reliability high/medium/low/unknown [h/m/l/?]: ", reliab
        )
        note = _ask(io, f"Evidence {n}: note (optional): ") or None
        judgements.append(
            EvidenceJudgement(
                evidence_id=e.evidence_id, relation=relation, reliability=reliability, note=note
            )
        )
        if relation in cited_candidates:
            cited_candidates[relation].append(e.evidence_id)
    verdict = _choose(
        io,
        "Verdict SUPPORTED/CONTRADICTED/INSUFFICIENT_EVIDENCE/NOT_CHECKABLE [s/c/i/n]: ",
        {
            "s": Verdict.SUPPORTED,
            "c": Verdict.CONTRADICTED,
            "i": Verdict.INSUFFICIENT_EVIDENCE,
            "n": Verdict.NOT_CHECKABLE,
        },
    )
    cited: list[str] = []
    wanted = {
        Verdict.SUPPORTED: EvidenceLabel.SUPPORTING,
        Verdict.CONTRADICTED: EvidenceLabel.CONTRADICTING,
    }.get(verdict)
    if wanted:
        cited = cited_candidates[wanted]
        if not cited:
            io.say(
                f"  A {verdict.value} label needs evidence you judged {wanted.value}. "
                "Nothing was saved; review again."
            )
            return None
    while True:
        rationale = _ask(io, "Rationale (what the evidence shows and why):\n> ")
        check = check_rationale(
            rationale,
            verdict.value,
            item.claim_text,
            [e.passage_text or "" for e in item.evidence if e.evidence_id in cited],
        )
        if check.errors:
            io.say(f"  Not accepted: {check.errors[0]}")
            continue
        if check.warnings:
            for w in check.warnings:
                io.say(f"  Warning: {w}")
            if not _yes(io, "Keep this rationale? [y/n] "):
                continue
        break
    uncertainty = _ask(io, "Uncertainty note (optional): ") or None
    research = _yes(io, "Is more research needed before this can be labelled? [y/n] ")
    reason = None
    if revise:
        reason = _ask(io, "Reason for revising your earlier review: ")
    io.say(
        f"\nSUMMARY: {verdict.value}; cited {len(cited)} passage(s); research needed: {research}"
    )
    if not _yes(io, "I confirm this judgement is my own assessment of the evidence. Save? [y/n] "):
        io.say("Not saved.")
        return None
    annotation, item = wf.record_annotation(
        store,
        item_id,
        reviewer_id=reviewer_id,
        review_kind=kind or ReviewKind.FIRST_REVIEW,
        verdict=verdict,
        rationale=rationale,
        evidence_judgements=judgements,
        cited_evidence_ids=cited,
        confirmed_own_judgement=True,
        uncertainty_note=uncertainty,
        research_needed=research,
        proposed_label_seen_before_review=seen,
        revision_of=revise,
        revision_reason=reason,
    )
    io.say(f"Saved {annotation.annotation_id}. Item status: {item.status.value}.")
    return annotation


def cmd_review(args) -> int:
    io = PROMPTER_FACTORY()
    try:
        interactive_review(
            _store(),
            args.item_id,
            args.reviewer_id,
            io,
            ReviewKind(args.kind) if args.kind else None,
            args.revise,
        )
    except _Quit:
        io.say("Stopped; nothing was saved.")
    return 0


def cmd_adjudicate(args) -> int:
    store, io = _store(), PROMPTER_FACTORY()
    item = store.load_item(args.item_id)
    current = wf.current_annotations(item, store.annotations(item.item_id))
    io.say(f"ADJUDICATION of {item.item_id}: {item.claim_text}")
    for a in current:
        io.say(f"  {a.reviewer_id} ({a.review_kind.value}): {a.verdict.value}")
        io.say(_wrap(a.rationale, "      "))
    try:
        verdict = _choose(
            io,
            "Final verdict [s/c/i/n]: ",
            {
                "s": Verdict.SUPPORTED,
                "c": Verdict.CONTRADICTED,
                "i": Verdict.INSUFFICIENT_EVIDENCE,
                "n": Verdict.NOT_CHECKABLE,
            },
        )
        rationale = _ask(io, "Adjudication rationale:\n> ")
        if not _yes(io, "I confirm this adjudication is my own assessment. Save? [y/n] "):
            io.say("Not saved.")
            return 0
    except _Quit:
        io.say("Stopped; nothing was saved.")
        return 0
    adj, item = wf.adjudicate(
        store,
        item.item_id,
        adjudicator_id=args.adjudicator_id,
        final_verdict=verdict,
        rationale=rationale,
        confirmed_own_judgement=True,
    )
    io.say(f"Saved {adj.adjudication_id}. Item status: {item.status.value}.")
    return 0


def cmd_approve(args) -> int:
    if not args.confirm_own_judgement:
        raise HallucinationLabError(
            "approval needs --confirm-own-judgement: you confirm that you checked the reviews "
            "and evidence yourself"
        )
    item = wf.approve(
        _store(),
        args.item_id,
        approver_id=args.approver_id,
        confirmed_own_judgement=True,
        note=args.note,
    )
    a = item.current_approval
    print(f"Approved {item.item_id} as {a.final_verdict.value} ({a.review_level.value}).")
    return 0


# ------------------------------------------------------------- integrity/export


def cmd_validate(args) -> int:
    errors, warnings = wf.validate_benchmark(_store())
    for w in warnings:
        print(f"warning: {w}")
    for e in errors:
        print(f"ERROR: {e}")
    print(f"{len(errors)} error(s), {len(warnings)} warning(s)")
    return 1 if errors else 0


def cmd_check_sources(args) -> int:
    results = check_sources(_store(), CollectionCatalog(Settings.from_env()), online=args.online)
    problems = [r for r in results if r.status not in {"ok", "not_checked"}]
    for r in results:
        if r.status != "ok" or args.verbose:
            print(f"{r.status:15} {r.item_id} {r.evidence_id} {r.reference_title}: {r.detail}")
    counts = {s: sum(r.status == s for r in results) for s in sorted({r.status for r in results})}
    print(f"checked {len(results)} evidence passage(s): {counts}")
    return 1 if problems else 0


def cmd_export(args) -> int:
    store = _store()
    if args.provisional:
        data = wf.provisional_export(store)
        text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
        if args.output:
            Path(args.output).write_text(text, encoding="utf-8")
            print(f"Wrote PROVISIONAL candidates (not ground truth) to {args.output}")
        else:
            sys.stdout.write(text)
        return 0
    if not args.version or not args.created_by:
        raise HallucinationLabError("an approved export needs --version X.Y.Z and --created-by")
    export, path = wf.export_approved(store, args.version, args.created_by)
    print(f"Exported {len(export.items)} approved item(s) as version {export.version} to {path}")
    print(f"dataset_sha256 {export.dataset_sha256}")
    return 0


def cmd_evaluate(args) -> int:
    store = _store()
    exports = store.exports()
    if args.export:
        path = Path(args.export)
    elif exports:
        path = exports[-1]
    else:
        raise BenchmarkError(
            "no approved export exists yet: approve items and run "
            "`hallab benchmark export` first. Metrics need human-approved "
            "labels."
        )
    export = store.load_export(path)
    predictions = load_predictions(args.predictions)
    result, info = run_evaluation(store, export, predictions, args.partition)
    if args.json:
        print(json.dumps({**result.to_dict(), **info}, indent=2, default=str))
        return 0
    print(
        f"Benchmark {result.benchmark_version} ({result.dataset_sha256[:12]}...), partition "
        f"{result.partition}, system {result.system['name']} {result.system['version']}"
    )
    print(
        f"Items: {result.n}; answered {result.answered}, abstained {result.abstained}, "
        f"missing {result.missing}"
    )
    print(
        f"Coverage {result.coverage:.3f}; conditional accuracy "
        f"{result.conditional_accuracy if result.conditional_accuracy is not None else 'n/a'}; "
        f"strict accuracy {result.accuracy:.3f}; macro-F1 {result.macro_f1}"
    )
    for label, m in result.per_class.items():
        print(
            f"  {label:22} support {m.support:3}  predicted {m.predicted:3}  "
            f"P {m.precision}  R {m.recall}  F1 {m.f1}"
        )
    print("Confusion (rows gold, columns predicted):")
    for gold, row in result.confusion.items():
        print(f"  {gold:22} " + "  ".join(f"{k[:5]}={v}" for k, v in row.items()))
    for note in result.notes:
        print(f"note: {note}")
    print(f"run saved as {info['run_id']}")
    return 0


def cmd_retrieval(args) -> int:
    catalog = CollectionCatalog(Settings.from_env())
    searcher, _ = catalog.searcher(args.collection)
    analysis = analyse_retrieval(_store(), searcher, approved_only=args.approved_only)
    print(analysis.notice)
    print(
        f"Scope: {analysis.scope}; items considered {analysis.items_considered}; with "
        f"evidence in '{analysis.collection}': {analysis.items_with_local_evidence}; "
        f"evidence coverage {analysis.evidence_coverage}"
    )
    if analysis.scope == "candidates":
        print("Relevance criteria come from provisional (AI-selected) evidence passages.")
    if analysis.report:
        for key, value in analysis.report.summary().items():
            if not key.startswith(("negative", "mean_top_score_negative", "max_top")):
                print(f"  {key:28} {value}")
        print("By evidence relation (found in top 5 / items):")
        for rel, d in sorted(analysis.by_relation.items()):
            print(f"  {rel:28} {d['found_in_top_5']}/{d['items']}")
    for p in analysis.label_problems:
        print(f"warning: {p}")
    if args.json and analysis.report:
        print(
            json.dumps(
                {
                    "summary": analysis.report.summary(),
                    "per_query": [asdict(o) for o in analysis.report.outcomes],
                },
                indent=2,
                default=str,
            )
        )
    return 0


# ---------------------------------------------------------------------- parser


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "benchmark",
        help="ground-truth benchmark: review, approve, export, evaluate",
        description=PROVISIONAL_NOTICE,
    )
    b = p.add_subparsers(dest="action", required=True)

    b.add_parser(
        "status", help="counts by status, partition, domain; approved labels"
    ).set_defaults(func=cmd_status)

    p_list = b.add_parser("list", help="list items with filters")
    p_list.add_argument(
        "--status",
        choices=[s.value for s in ReviewStatus] + [s.value.lower() for s in ReviewStatus],
    )
    p_list.add_argument("--domain", choices=[d.value for d in Domain])
    p_list.add_argument("--difficulty", choices=[d.value for d in Difficulty])
    p_list.add_argument("--partition", choices=[x.value for x in Partition])
    p_list.set_defaults(func=cmd_list)

    p_show = b.add_parser("show", help="show one item with evidence and reviews")
    p_show.add_argument("item_id")
    p_show.add_argument(
        "--reveal-proposed",
        action="store_true",
        help="also show the AI-proposed label (may bias your review)",
    )
    p_show.set_defaults(func=cmd_show)

    p_add = b.add_parser("add", help="add a human-authored candidate claim")
    p_add.add_argument("--claim", required=True)
    p_add.add_argument("--domain", required=True, choices=[d.value for d in Domain])
    p_add.add_argument("--difficulty", required=True, choices=[d.value for d in Difficulty])
    p_add.add_argument(
        "--type", action="append", required=True, choices=[t.value for t in ClaimType]
    )
    p_add.add_argument("--scope", help="temporal scope, e.g. 'as of 2024'")
    p_add.add_argument("--context", help="original response context")
    p_add.add_argument("--author-id", required=True)
    p_add.add_argument("--method", default="written by a human contributor")
    p_add.set_defaults(func=cmd_add)

    p_mark = b.add_parser("mark", help="change status (research needed, ready, excluded, ...)")
    p_mark.add_argument("item_id")
    p_mark.add_argument("--status", required=True)
    p_mark.add_argument("--actor", required=True)
    p_mark.add_argument("--reason", required=True)
    p_mark.set_defaults(func=cmd_mark)

    p_rev = b.add_parser("review", help="record your human review of an item (interactive)")
    p_rev.add_argument("item_id")
    p_rev.add_argument("--reviewer-id", required=True)
    p_rev.add_argument(
        "--kind",
        choices=[k.value for k in ReviewKind],
        help="default: chosen from the item's review history",
    )
    p_rev.add_argument("--revise", metavar="ANNOTATION_ID", help="correct your earlier review")
    p_rev.set_defaults(func=cmd_review)

    p_adj = b.add_parser("adjudicate", help="resolve disagreeing reviews (interactive)")
    p_adj.add_argument("item_id")
    p_adj.add_argument("--adjudicator-id", required=True)
    p_adj.set_defaults(func=cmd_adjudicate)

    p_apr = b.add_parser("approve", help="approve a reviewed item's label")
    p_apr.add_argument("item_id")
    p_apr.add_argument("--approver-id", required=True)
    p_apr.add_argument("--confirm-own-judgement", action="store_true")
    p_apr.add_argument("--note")
    p_apr.set_defaults(func=cmd_approve)

    b.add_parser(
        "validate", help="check files, references, labels and near-duplicates"
    ).set_defaults(func=cmd_validate)

    p_chk = b.add_parser("check-sources", help="detect changed or missing evidence sources")
    p_chk.add_argument("--online", action="store_true", help="re-fetch web pages")
    p_chk.add_argument("--verbose", action="store_true")
    p_chk.set_defaults(func=cmd_check_sources)

    p_exp = b.add_parser("export", help="export approved items as an immutable version")
    p_exp.add_argument(
        "--approved-only",
        action="store_true",
        default=True,
        help="(default) only human-APPROVED items",
    )
    p_exp.add_argument(
        "--provisional",
        action="store_true",
        help="export all candidates, clearly marked as NOT ground truth",
    )
    p_exp.add_argument("--version", help="semantic version, e.g. 1.0.0")
    p_exp.add_argument("--created-by", help="your reviewer ID")
    p_exp.add_argument("--output", help="file for --provisional output")
    p_exp.set_defaults(func=cmd_export)

    p_eval = b.add_parser("evaluate", help="score predictions against an approved export")
    p_eval.add_argument("--predictions", required=True)
    p_eval.add_argument("--export", help="approved export file (default: latest)")
    p_eval.add_argument("--partition", choices=["dev", "test", "all"], default="dev")
    p_eval.add_argument("--json", action="store_true")
    p_eval.set_defaults(func=cmd_evaluate)

    p_ret = b.add_parser(
        "retrieval", help="retrieval metrics on benchmark evidence (separate from verdict metrics)"
    )
    p_ret.add_argument("--collection", default="demo")
    p_ret.add_argument("--approved-only", action="store_true")
    p_ret.add_argument("--json", action="store_true")
    p_ret.set_defaults(func=cmd_retrieval)
