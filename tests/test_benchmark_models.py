"""Benchmark schema rules."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from hallucination_lab.benchmark.ids import dataset_hash, item_id_for, normalize_claim
from hallucination_lab.benchmark.models import (
    Annotation,
    ApprovalPolicy,
    ApprovedExport,
    CaptureStatus,
    EvidenceJudgement,
    EvidenceReference,
    ExportedItem,
    ProposedLabel,
    SplitRecommendation,
)
from hallucination_lab.ids import content_sha256
from hallucination_lab.models import Verdict

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
EID = "bev_0000000000000001"
H = "a" * 64


def _ann(**kw):
    data = dict(
        annotation_id="ann_0000000000000001",
        item_id="bmk_0000000000000001",
        item_content_sha256=H,
        reviewer_id="alice",
        review_kind="first_review",
        verdict="SUPPORTED",
        rationale="The cited passage states the same date.",
        evidence_judgements=[
            EvidenceJudgement(evidence_id=EID, relation="SUPPORTING", reliability="high")
        ],
        cited_evidence_ids=[EID],
        created_at=NOW,
        confirmed_own_judgement=True,
    )
    data.update(kw)
    return Annotation(**data)


def test_valid_annotation():
    assert _ann().reviewer_id == "alice"


@pytest.mark.parametrize(
    "reviewer",
    ["claude", "Claude-assistant", "ai", "assistant", "bot-1", "claude-reviewer", "has space"],
)
def test_ai_or_invalid_reviewer_ids_rejected(reviewer):
    with pytest.raises(ValidationError):
        _ann(reviewer_id=reviewer)


def test_own_judgement_confirmation_required():
    with pytest.raises(ValidationError):
        _ann(confirmed_own_judgement=False)


@pytest.mark.parametrize("rationale", ["n/a", "tbd", "4/5", "correct", ""])
def test_invalid_rationales_rejected(rationale):
    with pytest.raises(ValidationError, match="rationale"):
        _ann(rationale=rationale)


def test_supported_needs_cited_supporting_evidence():
    with pytest.raises(ValidationError, match="must cite"):
        _ann(cited_evidence_ids=[])
    with pytest.raises(ValidationError, match="must be judged SUPPORTING"):
        _ann(
            evidence_judgements=[
                EvidenceJudgement(evidence_id=EID, relation="IRRELEVANT", reliability="high")
            ]
        )


def test_citations_only_for_supported_or_contradicted():
    with pytest.raises(ValidationError, match="only used"):
        _ann(verdict="INSUFFICIENT_EVIDENCE")
    assert _ann(verdict="INSUFFICIENT_EVIDENCE", cited_evidence_ids=[]).verdict


def test_revision_needs_reason():
    with pytest.raises(ValidationError, match="revision_reason"):
        _ann(revision_of="ann_0000000000000009")
    assert _ann(
        revision_of="ann_0000000000000009",
        revision_reason="Re-read the evidence and corrected my verdict.",
    )


def test_captured_evidence_integrity():
    text = "The first bridge was completed in 1874."
    ok = EvidenceReference(
        evidence_id=EID,
        reference_id="ref_0000000000000001",
        capture_status=CaptureStatus.CAPTURED,
        locator="line 1",
        passage_text=text,
        passage_sha256=content_sha256(text),
        captured_at=NOW,
        captured_by="test",
    )
    assert ok
    with pytest.raises(ValidationError, match="altered"):
        ok.model_copy(update={"passage_text": text + "!"}).model_validate(
            {**ok.model_dump(), "passage_text": text + "!"}
        )
    with pytest.raises(ValidationError, match="passage_sha256"):
        EvidenceReference(
            evidence_id=EID,
            reference_id="ref_0000000000000001",
            capture_status=CaptureStatus.CAPTURED,
            locator="x",
            captured_by="t",
        )


def test_split_recommendation_needs_parts():
    with pytest.raises(ValidationError):
        SplitRecommendation(should_split=True, proposed_parts=["only one"])


def test_proposed_label_always_carries_notice():
    p = ProposedLabel(
        verdict="SUPPORTED",
        proposed_by="claude-assistant",
        proposer_type="ai_assistant",
        rationale="x",
        proposed_at=NOW,
    )
    assert p.notice == "proposed label: not human-reviewed ground truth"


def test_item_ids_are_stable_and_normalised():
    assert item_id_for("Pi is approximately 3.14159.") == item_id_for(
        "  pi IS approximately 3.14159 "
    )
    assert normalize_claim("Café\u2019s") == "cafe's"


def test_export_hash_detects_tampering():
    item = ExportedItem(
        item_id="bmk_0000000000000001",
        claim_text="x y z",
        domain="history",
        difficulty="easy",
        claim_types=["incorrect_date"],
        partition="dev",
        label="CONTRADICTED",
        review_level="single_review",
        approved_by="alice",
        approved_at=NOW,
        approval_id="apr_0000000000000001",
        item_content_sha256=H,
    )
    data = dict(
        benchmark_name="b",
        version="1.0.0",
        created_at=NOW,
        created_by="alice",
        policy=ApprovalPolicy(),
        items=[item],
        notice="n",
        dataset_sha256=dataset_hash([item]),
    )
    assert ApprovedExport(**data)
    with pytest.raises(ValidationError, match="dataset_sha256"):
        ApprovedExport(**{**data, "items": [item.model_copy(update={"label": Verdict.SUPPORTED})]})
    with pytest.raises(ValidationError):
        ApprovedExport(**{**data, "version": "v1"})
