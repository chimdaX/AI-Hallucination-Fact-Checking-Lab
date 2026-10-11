"""Benchmark workflow: candidates, status, reviews, adjudication, approval, export.

The official label of an item is computed only from **current human annotations**:
reviews of the item's current content fingerprint that were not superseded by a
revision, plus an adjudication when they disagree. A ``ProposedLabel`` is never
consulted. Every state change is appended to the item's history with its actor
and reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from hallucination_lab.benchmark.ids import (
    ADJUDICATION,
    ANNOTATION,
    APPROVAL,
    dataset_hash,
    item_id_for,
    normalize_claim,
)
from hallucination_lab.benchmark.models import (
    RESERVED_AI_IDS,
    Adjudication,
    Annotation,
    Approval,
    ApprovedExport,
    BenchmarkItem,
    CandidateProvenance,
    CaptureStatus,
    ClaimEdit,
    EvidenceJudgement,
    EvidenceReference,
    ExportedEvidence,
    ExportedItem,
    Partition,
    ReviewKind,
    ReviewLevel,
    ReviewStatus,
    StatusEvent,
    _human_reviewer,
)
from hallucination_lab.benchmark.store import BenchmarkError, BenchmarkStore
from hallucination_lab.ids import make_id
from hallucination_lab.models import Verdict, utc_now
from hallucination_lab.retrieval.tokenizer import tokenize

S = ReviewStatus
# Manual transitions (reviews, adjudications and approvals move status automatically).
MANUAL_TRANSITIONS: dict[ReviewStatus, set[ReviewStatus]] = {
    S.CANDIDATE: {S.RESEARCH_NEEDED, S.READY_FOR_REVIEW, S.EXCLUDED},
    S.RESEARCH_NEEDED: {S.READY_FOR_REVIEW, S.EXCLUDED},
    S.READY_FOR_REVIEW: {S.RESEARCH_NEEDED, S.EXCLUDED},
    S.REVIEWED: {S.RESEARCH_NEEDED, S.EXCLUDED},
    S.DISPUTED: {S.RESEARCH_NEEDED, S.EXCLUDED},
    S.APPROVED: {S.DISPUTED},  # reopening an approved label requires a documented reason
    S.EXCLUDED: {S.CANDIDATE},
}
# Transitions an automated author may make (preparation only, never decisions).
AUTOMATED_ALLOWED = {
    (S.CANDIDATE, S.RESEARCH_NEEDED),
    (S.CANDIDATE, S.READY_FOR_REVIEW),
    (S.RESEARCH_NEEDED, S.READY_FOR_REVIEW),
    (S.READY_FOR_REVIEW, S.RESEARCH_NEEDED),
}
REVIEWABLE = {S.READY_FOR_REVIEW, S.REVIEWED, S.DISPUTED}
EDITABLE = {S.CANDIDATE, S.RESEARCH_NEEDED, S.READY_FOR_REVIEW}
NEAR_DUPLICATE_THRESHOLD = 0.75


def is_automated_actor(actor: str) -> bool:
    low = actor.strip().lower()
    return low in RESERVED_AI_IDS or low.startswith(("claude", "ai-", "bot-"))


def require_human(actor: str, what: str) -> str:
    try:
        return _human_reviewer(actor)
    except ValueError as exc:
        raise BenchmarkError(f"{what} requires a human reviewer: {exc}") from exc


def assign_partition(item_id: str) -> Partition:
    """Deterministic split from the ID hash: about 1/3 dev, 2/3 held-out test."""
    return Partition.DEV if int(item_id.split("_", 1)[1], 16) % 3 == 0 else Partition.TEST


def _event(
    item: BenchmarkItem, to: ReviewStatus, actor: str, reason: str, now: datetime
) -> BenchmarkItem:
    if not reason or not reason.strip():
        raise BenchmarkError("a status change needs a reason")
    event = StatusEvent(from_status=item.status, to_status=to, actor=actor, reason=reason, at=now)
    return item.model_copy(update={"status_events": [*item.status_events, event]})


# ------------------------------------------------------------------- candidates


def similarity(a: str, b: str) -> float:
    ta, tb = set(tokenize(a)), set(tokenize(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def near_duplicates(
    items: list[BenchmarkItem], threshold: float = NEAR_DUPLICATE_THRESHOLD
) -> list[tuple[str, str, float]]:
    pairs = []
    for i, a in enumerate(items):
        for b in items[i + 1 :]:
            score = similarity(a.claim_text, b.claim_text)
            if score >= threshold:
                pairs.append((a.item_id, b.item_id, round(score, 3)))
    return pairs


def add_candidate(
    store: BenchmarkStore,
    *,
    claim_text: str,
    domain,
    difficulty,
    claim_types,
    provenance: CandidateProvenance,
    temporal_scope: str | None = None,
    original_context: str | None = None,
    split_recommendation=None,
    proposed_label=None,
    evidence: list[EvidenceReference] | None = None,
    research_log=None,
    research_tasks=None,
    notes: str | None = None,
    now: datetime | None = None,
) -> tuple[BenchmarkItem, list[str]]:
    """Create a CANDIDATE item. Exact duplicates are refused; near duplicates are warned."""
    now = now or utc_now()
    item_id = item_id_for(claim_text)
    items, _ = store.load_items()
    wanted = normalize_claim(claim_text)
    for other in items:
        if wanted in {
            normalize_claim(other.claim_text),
            normalize_claim(other.original_claim_text),
        }:
            raise BenchmarkError(f"duplicate claim: already present as {other.item_id}")
    item = BenchmarkItem(
        item_id=item_id,
        original_claim_text=claim_text,
        claim_text=claim_text,
        domain=domain,
        difficulty=difficulty,
        claim_types=claim_types,
        temporal_scope=temporal_scope,
        original_context=original_context,
        split_recommendation=split_recommendation,
        candidate_provenance=provenance,
        proposed_label=proposed_label,
        evidence=evidence or [],
        research_log=research_log or [],
        research_tasks=research_tasks or [],
        partition=assign_partition(item_id),
        notes=notes,
        status_events=[
            StatusEvent(
                from_status=None,
                to_status=S.CANDIDATE,
                actor=provenance.author,
                reason="candidate created",
                at=now,
            )
        ],
    )
    warnings = [
        f"similar to {other.item_id} ({score:.2f}): {other.claim_text}"
        for other in items
        if (score := similarity(claim_text, other.claim_text)) >= NEAR_DUPLICATE_THRESHOLD
    ]
    known = store.references()
    for e in item.evidence:
        if e.reference_id not in known:
            raise BenchmarkError(f"evidence cites unknown reference {e.reference_id}")
    store.create_item(item)
    return item, warnings


def set_status(
    store: BenchmarkStore,
    item_id: str,
    to: ReviewStatus,
    actor: str,
    reason: str,
    now: datetime | None = None,
) -> BenchmarkItem:
    item = store.load_item(item_id)
    allowed = MANUAL_TRANSITIONS.get(item.status, set())
    if to not in allowed:
        raise BenchmarkError(
            f"cannot move {item_id} from {item.status.value} to {to.value} manually"
            + (f"; allowed: {', '.join(sorted(s.value for s in allowed))}" if allowed else "")
        )
    if is_automated_actor(actor) and (item.status, to) not in AUTOMATED_ALLOWED:
        raise BenchmarkError(
            f"{item.status.value} -> {to.value} is a review decision and requires a human"
        )
    if not is_automated_actor(actor):
        require_human(actor, "this status change")
    updated = _event(item, to, actor, reason, now or utc_now())
    store.save_item(updated)
    return updated


def edit_claim(
    store: BenchmarkStore,
    item_id: str,
    new_text: str,
    editor: str,
    reason: str,
    now: datetime | None = None,
) -> BenchmarkItem:
    item = store.load_item(item_id)
    if item.status not in EDITABLE:
        raise BenchmarkError(
            f"claim wording can only change before review (status is {item.status.value}); "
            "exclude the item and add a new candidate instead"
        )
    new_text = " ".join(new_text.split())
    for other in store.load_items()[0]:
        if other.item_id != item_id and normalize_claim(other.claim_text) == normalize_claim(
            new_text
        ):
            raise BenchmarkError(f"the new wording duplicates {other.item_id}")
    edit = ClaimEdit(
        previous_text=item.claim_text,
        new_text=new_text,
        editor=editor,
        reason=reason,
        at=now or utc_now(),
    )
    updated = item.model_copy(
        update={"claim_text": new_text, "claim_edits": [*item.claim_edits, edit]}
    )
    store.save_item(updated)
    return updated


def add_evidence(
    store: BenchmarkStore,
    item_id: str,
    evidence: EvidenceReference,
    actor: str,
    now: datetime | None = None,
) -> BenchmarkItem:
    item = store.load_item(item_id)
    if item.status in {S.APPROVED, S.EXCLUDED}:
        raise BenchmarkError(f"cannot add evidence to an item that is {item.status.value}")
    store.load_reference(evidence.reference_id)  # must exist
    if any(e.evidence_id == evidence.evidence_id for e in item.evidence):
        raise BenchmarkError("this evidence passage is already attached")
    updated = item.model_copy(update={"evidence": [*item.evidence, evidence]})
    if item.status in {S.REVIEWED, S.DISPUTED}:
        updated = _event(
            updated,
            S.READY_FOR_REVIEW,
            actor,
            "evidence added; earlier reviews kept but no longer current",
            now or utc_now(),
        )
    store.save_item(updated)
    return updated


# ---------------------------------------------------------------------- reviews


def current_annotations(item: BenchmarkItem, annotations: list[Annotation]) -> list[Annotation]:
    """Reviews of the current content that have not been superseded by a revision."""
    superseded = {a.revision_of for a in annotations if a.revision_of}
    return [
        a
        for a in annotations
        if a.item_content_sha256 == item.content_sha256 and a.annotation_id not in superseded
    ]


@dataclass
class LabelDecision:
    verdict: Verdict | None
    review_level: ReviewLevel | None
    basis: list[Annotation] = field(default_factory=list)
    adjudication: Adjudication | None = None
    problem: str | None = None


def decide_label(
    item: BenchmarkItem, annotations: list[Annotation], adjudications: list[Adjudication]
) -> LabelDecision:
    current = current_annotations(item, annotations)
    if not current:
        return LabelDecision(None, None, problem="no current human review")
    ids = {a.annotation_id for a in current}
    for adj in reversed(adjudications):
        if adj.item_content_sha256 == item.content_sha256 and ids <= set(
            adj.considered_annotation_ids
        ):
            return LabelDecision(adj.final_verdict, ReviewLevel.ADJUDICATED, current, adj)
    verdicts = {a.verdict for a in current}
    if len(verdicts) > 1:
        return LabelDecision(None, None, current, problem="reviews disagree: adjudication required")
    reviewers = {a.reviewer_id for a in current}
    if len(reviewers) >= 2:
        level = ReviewLevel.INDEPENDENT_DOUBLE_REVIEW
    elif len(current) >= 2:
        level = ReviewLevel.REPEAT_REVIEW_SAME_REVIEWER
    else:
        level = ReviewLevel.SINGLE_REVIEW
    return LabelDecision(current[0].verdict, level, current)


def _status_after_review(
    item: BenchmarkItem, annotations: list[Annotation], adjudications: list[Adjudication]
) -> tuple[ReviewStatus, str]:
    current = current_annotations(item, annotations)
    if any(a.research_needed for a in current):
        return S.RESEARCH_NEEDED, "a reviewer flagged missing research"
    decision = decide_label(item, annotations, adjudications)
    if decision.verdict is None and decision.problem and "disagree" in decision.problem:
        return S.DISPUTED, "reviews disagree"
    return S.REVIEWED, "human review recorded"


def record_annotation(
    store: BenchmarkStore,
    item_id: str,
    *,
    reviewer_id: str,
    review_kind: ReviewKind,
    verdict: Verdict,
    rationale: str,
    evidence_judgements: list[EvidenceJudgement],
    cited_evidence_ids: list[str],
    confirmed_own_judgement: bool,
    uncertainty_note: str | None = None,
    research_needed: bool = False,
    proposed_label_seen_before_review: bool = False,
    revision_of: str | None = None,
    revision_reason: str | None = None,
    now: datetime | None = None,
) -> tuple[Annotation, BenchmarkItem]:
    now = now or utc_now()
    reviewer_id = require_human(reviewer_id, "a review")
    if not confirmed_own_judgement:
        raise BenchmarkError("confirm that this judgement is your own assessment")
    item = store.load_item(item_id)
    if item.status not in REVIEWABLE:
        raise BenchmarkError(
            f"{item_id} is {item.status.value}; only READY_FOR_REVIEW, REVIEWED or DISPUTED "
            "items can be reviewed"
        )
    evidence = {e.evidence_id: e for e in item.evidence}
    for j in evidence_judgements:
        if j.evidence_id not in evidence:
            raise BenchmarkError(f"{j.evidence_id} is not an evidence passage of this item")
    for eid in cited_evidence_ids:
        if eid not in evidence:
            raise BenchmarkError(f"{eid} is not an evidence passage of this item")
        if evidence[eid].capture_status is not CaptureStatus.CAPTURED:
            raise BenchmarkError(f"{eid} is an unverified lead and cannot be cited")

    annotations = store.annotations(item_id)
    current = current_annotations(item, annotations)
    mine = [a for a in current if a.reviewer_id == reviewer_id]
    policy = store.load_manifest().policy if store.exists() else None
    if revision_of:
        target = next((a for a in annotations if a.annotation_id == revision_of), None)
        if target is None:
            raise BenchmarkError(f"no annotation {revision_of} for this item")
        if target.reviewer_id != reviewer_id:
            raise BenchmarkError(
                "only the original reviewer can revise their review; "
                "others add a review of their own"
            )
        if any(a.revision_of == revision_of for a in annotations):
            raise BenchmarkError(f"{revision_of} was already revised; revise the latest version")
        if target not in current:
            raise BenchmarkError(f"{revision_of} is not a current review of this item")
        review_kind = target.review_kind
    elif review_kind is ReviewKind.FIRST_REVIEW:
        if current:
            raise BenchmarkError(
                "this item already has a current review; add an independent "
                "second review, a delayed repeat review, or revise yours"
            )
    elif review_kind is ReviewKind.INDEPENDENT_SECOND_REVIEW:
        if not current:
            raise BenchmarkError("an independent second review needs a first review")
        if mine:
            raise BenchmarkError(
                "an independent second review must come from a different "
                "reviewer; use repeat_review_same_reviewer instead"
            )
    elif review_kind is ReviewKind.REPEAT_REVIEW_SAME_REVIEWER:
        if not mine:
            raise BenchmarkError("a repeat review needs an earlier review by the same reviewer")
        if any(a.review_kind is ReviewKind.REPEAT_REVIEW_SAME_REVIEWER for a in mine):
            raise BenchmarkError("you already recorded a repeat review for this item")
        delay = policy.min_repeat_review_delay_hours if policy else 24
        if now - max(a.created_at for a in mine) < timedelta(hours=delay):
            raise BenchmarkError(
                f"a repeat review must be at least {delay} hours after your "
                "previous review of this item"
            )
    annotation = Annotation(
        annotation_id=make_id(ANNOTATION, item_id, reviewer_id, now.isoformat(), revision_of),
        item_id=item_id,
        item_content_sha256=item.content_sha256,
        reviewer_id=reviewer_id,
        review_kind=review_kind,
        verdict=verdict,
        rationale=rationale,
        evidence_judgements=evidence_judgements,
        cited_evidence_ids=cited_evidence_ids,
        uncertainty_note=uncertainty_note,
        research_needed=research_needed,
        created_at=now,
        revision_of=revision_of,
        revision_reason=revision_reason,
        proposed_label_seen_before_review=proposed_label_seen_before_review,
        confirmed_own_judgement=True,
    )
    store.save_annotation(annotation)
    status, reason = _status_after_review(
        item, [*annotations, annotation], store.adjudications(item_id)
    )
    if status is not item.status:
        item = _event(item, status, reviewer_id, reason, now)
        store.save_item(item)
    return annotation, item


def adjudicate(
    store: BenchmarkStore,
    item_id: str,
    *,
    adjudicator_id: str,
    final_verdict: Verdict,
    rationale: str,
    confirmed_own_judgement: bool,
    now: datetime | None = None,
) -> tuple[Adjudication, BenchmarkItem]:
    now = now or utc_now()
    adjudicator_id = require_human(adjudicator_id, "adjudication")
    if not confirmed_own_judgement:
        raise BenchmarkError("confirm that this adjudication is your own assessment")
    item = store.load_item(item_id)
    if item.status is not S.DISPUTED:
        raise BenchmarkError(
            f"only DISPUTED items are adjudicated ({item_id} is {item.status.value})"
        )
    current = current_annotations(item, store.annotations(item_id))
    if len(current) < 2:
        raise BenchmarkError("adjudication needs at least two current reviews")
    adjudication = Adjudication(
        adjudication_id=make_id(ADJUDICATION, item_id, adjudicator_id, now.isoformat()),
        item_id=item_id,
        item_content_sha256=item.content_sha256,
        adjudicator_id=adjudicator_id,
        considered_annotation_ids=[a.annotation_id for a in current],
        final_verdict=final_verdict,
        rationale=rationale,
        created_at=now,
        confirmed_own_judgement=True,
    )
    store.save_adjudication(adjudication)
    item = _event(item, S.REVIEWED, adjudicator_id, "disagreement adjudicated", now)
    store.save_item(item)
    return adjudication, item


def approve(
    store: BenchmarkStore,
    item_id: str,
    *,
    approver_id: str,
    confirmed_own_judgement: bool,
    note: str | None = None,
    now: datetime | None = None,
) -> BenchmarkItem:
    now = now or utc_now()
    approver_id = require_human(approver_id, "approval")
    if not confirmed_own_judgement:
        raise BenchmarkError("confirm that you checked the reviews and evidence yourself")
    item = store.load_item(item_id)
    if item.status not in {S.REVIEWED, S.DISPUTED}:
        raise BenchmarkError(
            f"only REVIEWED (or adjudicated) items can be approved; "
            f"{item_id} is {item.status.value}"
        )
    decision = decide_label(item, store.annotations(item_id), store.adjudications(item_id))
    if decision.verdict is None:
        raise BenchmarkError(f"cannot approve {item_id}: {decision.problem}")
    policy = store.load_manifest().policy
    if len(decision.basis) < policy.min_reviews:
        raise BenchmarkError(f"policy requires {policy.min_reviews} current review(s)")
    if policy.require_independent_second_review and decision.review_level not in {
        ReviewLevel.INDEPENDENT_DOUBLE_REVIEW,
        ReviewLevel.ADJUDICATED,
    }:
        raise BenchmarkError("policy requires an independent second review")
    if decision.verdict in {Verdict.SUPPORTED, Verdict.CONTRADICTED}:
        evidence = {e.evidence_id: e for e in item.evidence}
        cited = {
            eid
            for a in decision.basis
            if a.verdict is decision.verdict
            for eid in a.cited_evidence_ids
        }
        if not cited:
            raise BenchmarkError("a SUPPORTED/CONTRADICTED label needs cited evidence")
        missing = [
            eid
            for eid in cited
            if eid not in evidence or evidence[eid].capture_status is not CaptureStatus.CAPTURED
        ]
        if missing:
            raise BenchmarkError(f"cited evidence is not captured: {missing}")
    approval = Approval(
        approval_id=make_id(APPROVAL, item_id, approver_id, now.isoformat()),
        approved_by=approver_id,
        approved_at=now,
        final_verdict=decision.verdict,
        review_level=decision.review_level,
        basis_annotation_ids=[a.annotation_id for a in decision.basis],
        adjudication_id=decision.adjudication.adjudication_id if decision.adjudication else None,
        item_content_sha256=item.content_sha256,
        note=note,
        confirmed_own_judgement=True,
    )
    updated = item.model_copy(update={"approvals": [*item.approvals, approval]})
    updated = _event(
        updated,
        S.APPROVED,
        approver_id,
        f"approved as {decision.verdict.value} ({decision.review_level.value})",
        now,
    )
    store.save_item(updated)
    return updated


# ---------------------------------------------------------------------- export


def export_approved(
    store: BenchmarkStore, version: str, created_by: str, now: datetime | None = None
) -> tuple[ApprovedExport, object]:
    created_by = require_human(created_by, "exporting an approved benchmark version")
    manifest = store.load_manifest()
    items, problems = store.load_items()
    if problems:
        raise BenchmarkError("fix unreadable items before exporting: " + "; ".join(problems))
    refs = store.references()
    exported = []
    for item in items:
        approval = item.current_approval
        if approval is None:
            continue
        if approval.item_content_sha256 != item.content_sha256:
            raise BenchmarkError(f"{item.item_id}: approval does not match current content")
        evidence = [
            ExportedEvidence(
                evidence_id=e.evidence_id,
                reference_id=e.reference_id,
                title=refs[e.reference_id].title,
                url=refs[e.reference_id].url,
                locator=e.locator,
                passage_sha256=e.passage_sha256,
                passage_text=e.passage_text,
                redistribution=refs[e.reference_id].redistribution,
            )
            for e in item.evidence
            if e.capture_status is CaptureStatus.CAPTURED
        ]
        exported.append(
            ExportedItem(
                item_id=item.item_id,
                claim_text=item.claim_text,
                domain=item.domain,
                difficulty=item.difficulty,
                claim_types=item.claim_types,
                temporal_scope=item.temporal_scope,
                partition=item.partition,
                label=approval.final_verdict,
                review_level=approval.review_level,
                approved_by=approval.approved_by,
                approved_at=approval.approved_at,
                approval_id=approval.approval_id,
                item_content_sha256=approval.item_content_sha256,
                evidence=evidence,
            )
        )
    if not exported:
        raise BenchmarkError("no APPROVED items yet: nothing can be exported as ground truth")
    exported.sort(key=lambda i: i.item_id)
    export = ApprovedExport(
        benchmark_name=manifest.name,
        version=version,
        created_at=now or utc_now(),
        created_by=created_by,
        policy=manifest.policy,
        dataset_sha256=dataset_hash(exported),
        items=exported,
        notice=(
            "Human-approved labels only. Review levels are recorded per item; "
            "single_review and repeat_review_same_reviewer items have no independent "
            "inter-annotator check."
        ),
    )
    path = store.save_export(export)
    return export, path


def provisional_export(store: BenchmarkStore) -> dict:
    """All candidates for inspection. Proposed labels are clearly marked; not ground truth."""
    items, _ = store.load_items()
    return {
        "format": "hallab-benchmark-provisional-candidates",
        "notice": "PROVISIONAL. Not ground truth. Proposed labels were not human-reviewed "
        "and must not be used for evaluation.",
        "items": [
            {
                "item_id": i.item_id,
                "claim_text": i.claim_text,
                "status": i.status.value,
                "domain": i.domain.value,
                "difficulty": i.difficulty.value,
                "partition": i.partition.value,
                "proposed_label": i.proposed_label.model_dump(mode="json")
                if i.proposed_label
                else None,
                "approved_label": i.current_approval.final_verdict.value
                if i.current_approval
                else None,
            }
            for i in items
        ],
    }


# ---------------------------------------------------------------------- status


def status_summary(store: BenchmarkStore) -> dict:
    items, problems = store.load_items()
    count = lambda values: {k: values.count(k) for k in sorted(set(values))}  # noqa: E731
    approved = [i for i in items if i.status is S.APPROVED]
    pending = [i.item_id for i in items if i.status is S.READY_FOR_REVIEW]
    return {
        "total": len(items),
        "by_status": {s.value: sum(i.status is s for i in items) for s in ReviewStatus},
        "by_domain": count([i.domain.value for i in items]),
        "by_difficulty": count([i.difficulty.value for i in items]),
        "by_partition": count([i.partition.value for i in items]),
        "approved_labels": count([i.current_approval.final_verdict.value for i in approved]),
        "approved_by_partition": count([i.partition.value for i in approved]),
        "proposed_labels_not_ground_truth": count(
            [i.proposed_label.verdict.value for i in items if i.proposed_label]
        ),
        "pending_review": pending,
        "disputed": [i.item_id for i in items if i.status is S.DISPUTED],
        "open_research_tasks": sum(1 for i in items for t in i.research_tasks if not t.resolved),
        "unreadable": problems,
    }


def validate_benchmark(store: BenchmarkStore) -> tuple[list[str], list[str]]:
    """Return (errors, warnings) for the whole benchmark."""
    errors: list[str] = []
    warnings: list[str] = []
    try:
        store.load_manifest()
    except BenchmarkError as exc:
        errors.append(str(exc))
    items, problems = store.load_items()
    errors.extend(problems)
    try:
        refs = store.references()
    except BenchmarkError as exc:
        errors.append(str(exc))
        refs = {}
    for item in items:
        for e in item.evidence:
            if e.reference_id not in refs:
                errors.append(f"{item.item_id}: evidence cites unknown reference {e.reference_id}")
        try:
            annotations = store.annotations(item.item_id)
            adjudications = store.adjudications(item.item_id)
        except BenchmarkError as exc:
            errors.append(str(exc))
            continue
        if item.status is S.APPROVED:
            approval = item.current_approval
            decision = decide_label(item, annotations, adjudications)
            if approval.item_content_sha256 != item.content_sha256:
                errors.append(f"{item.item_id}: approved content changed after approval")
            elif decision.verdict is not approval.final_verdict:
                errors.append(
                    f"{item.item_id}: approved label no longer matches current reviews"
                    " (reopen the item for a documented review)"
                )
        if (
            item.status in {S.READY_FOR_REVIEW, S.REVIEWED}
            and not item.evidence
            and not (item.research_log)
        ):
            warnings.append(
                f"{item.item_id}: no evidence or search record attached "
                "(acceptable only if the claim is not objectively checkable)"
            )
        open_tasks = [t for t in item.research_tasks if not t.resolved]
        if open_tasks:
            warnings.append(f"{item.item_id}: {len(open_tasks)} open research task(s)")
        if item.split_recommendation and item.split_recommendation.should_split:
            warnings.append(f"{item.item_id}: compound claim; split recommended before review")
    for a, b, score in near_duplicates(items):
        warnings.append(f"near-duplicate claims {a} and {b} (similarity {score})")
    for path in store.exports():
        try:
            store.load_export(path)
        except BenchmarkError as exc:
            errors.append(str(exc))
    return errors, warnings
