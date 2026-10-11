"""Benchmark workflow: candidates, status, reviews, adjudication, approval, audit."""

import json
from datetime import timedelta

import pytest

from hallucination_lab.benchmark import workflow as wf
from hallucination_lab.benchmark.models import Partition, ReviewLevel, ReviewStatus
from hallucination_lab.benchmark.store import BenchmarkError
from hallucination_lab.models import Verdict, utc_now

CLAIM = "The first bridge across Lake Verenna was completed in 1880."


def test_candidate_creation_and_duplicates(bench):
    item = bench.item(CLAIM)
    assert item.status is ReviewStatus.READY_FOR_REVIEW
    assert [e.to_status for e in item.status_events] == [
        ReviewStatus.CANDIDATE,
        ReviewStatus.READY_FOR_REVIEW,
    ]
    with pytest.raises(BenchmarkError, match="duplicate"):
        bench.item("  the FIRST bridge across lake verenna was completed in 1880 ")


def test_near_duplicates_flagged(bench):
    bench.item(CLAIM)
    _, warnings = wf.add_candidate(
        bench.store,
        claim_text="The first bridge across Lake Verenna was finished in 1880.",
        domain="history",
        difficulty="easy",
        claim_types=["incorrect_date"],
        provenance=bench.store.load_item(wf.item_id_for(CLAIM)).candidate_provenance,
    )
    assert warnings and "similar" in warnings[0]
    errors, warns = wf.validate_benchmark(bench.store)
    assert not errors and any("near-duplicate" in w for w in warns)


def test_partition_is_deterministic(bench):
    item = bench.item(CLAIM)
    assert item.partition is wf.assign_partition(item.item_id)


def test_unknown_evidence_reference_rejected(bench):
    item = bench.item(CLAIM, ready=False)
    bad = item.evidence[0].model_copy(update={"reference_id": "ref_ffffffffffffffff"})
    with pytest.raises(BenchmarkError, match="unknown reference"):
        wf.add_candidate(
            bench.store,
            claim_text="Another distinct synthetic claim here.",
            domain="history",
            difficulty="easy",
            claim_types=["ambiguous"],
            provenance=item.candidate_provenance,
            evidence=[bad],
        )


def test_status_transitions(bench):
    item = bench.item(CLAIM)
    with pytest.raises(BenchmarkError, match="cannot move"):
        wf.set_status(bench.store, item.item_id, ReviewStatus.APPROVED, "alice", "no shortcut")
    with pytest.raises(BenchmarkError, match="requires a human"):
        wf.set_status(
            bench.store,
            item.item_id,
            ReviewStatus.EXCLUDED,
            "claude-assistant",
            "AI cannot exclude",
        )
    excluded = wf.set_status(
        bench.store, item.item_id, ReviewStatus.EXCLUDED, "alice", "not useful"
    )
    assert excluded.status is ReviewStatus.EXCLUDED
    with pytest.raises(BenchmarkError, match="reason"):
        wf.set_status(bench.store, item.item_id, ReviewStatus.CANDIDATE, "alice", "  ")


def test_claim_edits_keep_history_and_stop_after_review(bench):
    item = bench.item(CLAIM)
    edited = wf.edit_claim(
        bench.store,
        item.item_id,
        "The first Verenna bridge opened in 1880.",
        "alice",
        "clearer wording",
    )
    assert edited.original_claim_text == CLAIM and edited.claim_edits[0].previous_text == CLAIM
    bench.review(edited)
    with pytest.raises(BenchmarkError, match="before review"):
        wf.edit_claim(bench.store, item.item_id, "Something else entirely here.", "alice", "x")


def test_first_review_and_label(bench):
    item = bench.item(CLAIM, proposed="SUPPORTED")  # AI proposed the WRONG label on purpose
    _, item = bench.review(item, verdict="CONTRADICTED")
    assert item.status is ReviewStatus.REVIEWED
    decision = wf.decide_label(item, bench.store.annotations(item.item_id), [])
    assert decision.verdict is Verdict.CONTRADICTED  # human label, never the proposal
    assert decision.review_level is ReviewLevel.SINGLE_REVIEW


def test_review_kind_rules(bench):
    item = bench.item(CLAIM)
    bench.review(item)
    with pytest.raises(BenchmarkError, match="already has a current review"):
        bench.review(item, reviewer="bob")
    with pytest.raises(BenchmarkError, match="different reviewer"):
        bench.review(item, reviewer="alice", kind="independent_second_review")
    with pytest.raises(BenchmarkError, match="24 hours"):
        bench.review(item, reviewer="alice", kind="repeat_review_same_reviewer")
    with pytest.raises(BenchmarkError, match="earlier review by the same reviewer"):
        bench.review(item, reviewer="carol", kind="repeat_review_same_reviewer")
    _, updated = bench.review(item, reviewer="bob", kind="independent_second_review")
    decision = wf.decide_label(updated, bench.store.annotations(item.item_id), [])
    assert decision.review_level is ReviewLevel.INDEPENDENT_DOUBLE_REVIEW


def test_delayed_repeat_review_is_labelled_as_such(bench):
    item = bench.item(CLAIM)
    bench.review(item)
    later = utc_now() + timedelta(hours=25)
    _, updated = bench.review(item, kind="repeat_review_same_reviewer", now=later)
    decision = wf.decide_label(updated, bench.store.annotations(item.item_id), [])
    assert decision.review_level is ReviewLevel.REPEAT_REVIEW_SAME_REVIEWER
    with pytest.raises(BenchmarkError, match="already recorded a repeat"):
        bench.review(item, kind="repeat_review_same_reviewer", now=later + timedelta(days=2))


def test_disagreement_and_adjudication(bench):
    item = bench.item(CLAIM)
    bench.review(item)
    _, item = bench.review(
        item,
        reviewer="bob",
        kind="independent_second_review",
        verdict="INSUFFICIENT_EVIDENCE",
        relation="UNCERTAIN",
        rationale="The passage may refer to a different bridge; unclear.",
    )
    assert item.status is ReviewStatus.DISPUTED
    with pytest.raises(BenchmarkError, match="disagree"):
        wf.approve(bench.store, item.item_id, approver_id="carol", confirmed_own_judgement=True)
    adj, item = wf.adjudicate(
        bench.store,
        item.item_id,
        adjudicator_id="carol",
        final_verdict=Verdict.CONTRADICTED,
        rationale="The passage names the first bridge and gives 1874.",
        confirmed_own_judgement=True,
    )
    assert item.status is ReviewStatus.REVIEWED
    approved = wf.approve(
        bench.store, item.item_id, approver_id="carol", confirmed_own_judgement=True
    )
    a = approved.current_approval
    assert a.final_verdict is Verdict.CONTRADICTED and a.review_level is ReviewLevel.ADJUDICATED
    assert a.adjudication_id == adj.adjudication_id
    assert len(bench.store.annotations(item.item_id)) == 2  # originals preserved


def test_unauthorized_approvals(bench):
    item = bench.item(CLAIM)
    with pytest.raises(BenchmarkError, match="REVIEWED"):
        wf.approve(bench.store, item.item_id, approver_id="alice", confirmed_own_judgement=True)
    bench.review(item)
    with pytest.raises(BenchmarkError, match="human reviewer"):
        wf.approve(bench.store, item.item_id, approver_id="claude", confirmed_own_judgement=True)
    with pytest.raises(BenchmarkError, match="confirm"):
        wf.approve(bench.store, item.item_id, approver_id="alice", confirmed_own_judgement=False)
    manifest = bench.store.load_manifest()
    bench.store.save_manifest(
        manifest.model_copy(
            update={
                "policy": manifest.policy.model_copy(
                    update={"require_independent_second_review": True}
                )
            }
        )
    )
    with pytest.raises(BenchmarkError, match="independent second review"):
        wf.approve(bench.store, item.item_id, approver_id="alice", confirmed_own_judgement=True)


def test_review_requires_human_and_confirmation(bench):
    item = bench.item(CLAIM)
    with pytest.raises(BenchmarkError, match="human reviewer"):
        bench.review(item, reviewer="claude-assistant")
    with pytest.raises(BenchmarkError, match="confirm"):
        bench.review(item, confirmed=False)


def test_review_not_allowed_before_ready_or_after_approval(bench):
    item = bench.item(CLAIM, ready=False)
    with pytest.raises(BenchmarkError, match="only READY_FOR_REVIEW"):
        bench.review(item)
    item = bench.item(
        "A second bridge opened on 3 May 1921 across the lake.",
        passages=(("3 May 1921", "SUPPORTING"),),
    )
    bench.review(
        item,
        verdict="SUPPORTED",
        relation="SUPPORTING",
        rationale="Passage gives 3 May 1921 for the second bridge.",
    )
    wf.approve(bench.store, item.item_id, approver_id="alice", confirmed_own_judgement=True)
    with pytest.raises(BenchmarkError, match="only READY_FOR_REVIEW"):
        bench.review(item, reviewer="bob", kind="independent_second_review")


def test_reopen_approved_requires_documented_human_review(bench):
    item = bench.item(CLAIM)
    bench.review(item)
    wf.approve(bench.store, item.item_id, approver_id="alice", confirmed_own_judgement=True)
    with pytest.raises(BenchmarkError, match="requires a human"):
        wf.set_status(bench.store, item.item_id, ReviewStatus.DISPUTED, "claude", "reopen")
    reopened = wf.set_status(
        bench.store,
        item.item_id,
        ReviewStatus.DISPUTED,
        "alice",
        "new evidence suggests the date is wrong",
    )
    assert reopened.status is ReviewStatus.DISPUTED and len(reopened.approvals) == 1


def test_annotation_revisions(bench):
    item = bench.item(CLAIM)
    first, _ = bench.review(item)
    with pytest.raises(BenchmarkError, match="original reviewer"):
        bench.review(
            item,
            reviewer="bob",
            revision_of=first.annotation_id,
            revision_reason="I disagree with this.",
        )
    second, updated = bench.review(
        item,
        verdict="INSUFFICIENT_EVIDENCE",
        relation="UNCERTAIN",
        revision_of=first.annotation_id,
        revision_reason="On reflection the passage is ambiguous.",
        rationale="Unclear which bridge the passage refers to.",
    )
    current = wf.current_annotations(updated, bench.store.annotations(item.item_id))
    assert [a.annotation_id for a in current] == [second.annotation_id]
    with pytest.raises(BenchmarkError, match="already revised"):
        bench.review(item, revision_of=first.annotation_id, revision_reason="again, really.")


def test_research_needed_flag(bench):
    item = bench.item(CLAIM)
    _, updated = bench.review(
        item,
        verdict="INSUFFICIENT_EVIDENCE",
        relation="UNCERTAIN",
        rationale="Need a primary source for the bridge date.",
        research_needed=True,
    )
    assert updated.status is ReviewStatus.RESEARCH_NEEDED


def test_new_evidence_makes_old_reviews_stale(bench):
    item = bench.item(CLAIM)
    bench.review(item)
    from hallucination_lab.benchmark.capture import extract_sentence
    from hallucination_lab.benchmark.ids import evidence_id_for

    cap = extract_sentence(bench.lake_text, "3 May 1921")
    extra = item.evidence[0].model_copy(
        update={
            "evidence_id": evidence_id_for(bench.lake_ref.reference_id, cap.sha256, "x"),
            "passage_text": cap.text,
            "passage_sha256": cap.sha256,
            "locator": "x",
        }
    )
    updated = wf.add_evidence(bench.store, item.item_id, extra, "alice")
    assert updated.status is ReviewStatus.READY_FOR_REVIEW
    assert wf.current_annotations(updated, bench.store.annotations(item.item_id)) == []
    assert len(bench.store.annotations(item.item_id)) == 1


def test_files_are_write_once_and_history_append_only(bench):
    item = bench.item(CLAIM)
    annotation, _ = bench.review(item)
    with pytest.raises(BenchmarkError, match="write-once"):
        bench.store.save_annotation(annotation)
    loaded = bench.store.load_item(item.item_id)
    with pytest.raises(BenchmarkError, match="append-only"):
        bench.store.save_item(loaded.model_copy(update={"status_events": loaded.status_events[:1]}))
    with pytest.raises(BenchmarkError, match="partition"):
        other = Partition.DEV if loaded.partition is Partition.TEST else Partition.TEST
        bench.store.save_item(loaded.model_copy(update={"partition": other}))


def test_tampered_item_file_detected(bench):
    item = bench.item(CLAIM)
    path = bench.store.item_path(item.item_id)
    data = json.loads(path.read_text())
    data["evidence"][0]["passage_text"] += " (edited)"
    path.write_text(json.dumps(data))
    errors, _ = wf.validate_benchmark(bench.store)
    assert errors and "altered" in errors[0]


def test_unverified_leads_cannot_be_cited(bench):
    item = bench.item(CLAIM)
    path = bench.store.item_path(item.item_id)
    data = json.loads(path.read_text())
    data["evidence"][0]["capture_status"] = "unverified_lead"
    path.write_text(json.dumps(data))
    with pytest.raises(BenchmarkError, match="unverified lead"):
        bench.review(item)
