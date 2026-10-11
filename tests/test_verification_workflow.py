"""Workflow rules: evidence citation, revisions, integrity, progress, summary."""

from datetime import timedelta

import pytest

from hallucination_lab.errors import VerificationError
from hallucination_lab.models import (
    EvidenceAssessment,
    EvidenceLabel,
    Verdict,
    VerificationDraft,
    utc_now,
)
from hallucination_lab.verification.integrity import SourceStatus, check_session_evidence
from hallucination_lab.verification.workflow import (
    VerificationInput,
    build_verification,
    compute_progress,
    discard_draft,
    record_verification,
    save_draft,
    verification_summary,
)
from tests.conftest import evidence_containing


@pytest.fixture
def env(verification_env):
    e = dict(verification_env)
    e["session"] = e["store"].load(e["session_id"])
    return e


def _assess(eid, label):
    return EvidenceAssessment(evidence_id=eid, label=label)


def _contradict_bridge(env, reviewer="r1", now=None, **kw):
    s = env["session"]
    eid = evidence_containing(s, env["bridge"], "completed in 1874")
    inp = VerificationInput(
        claim_id=env["bridge"],
        verdict=Verdict.CONTRADICTED,
        rationale="The passage says the first bridge was completed in 1874, not 1880.",
        assessments=[_assess(eid, EvidenceLabel.CONTRADICTING)],
        contradicting_evidence_ids=[eid],
        **kw,
    )
    return build_verification(
        s, reviewer, inp, checks=check_session_evidence(s, env["catalog"]), now=now or utc_now()
    )


def test_top_ranked_passage_cannot_become_supported_without_assessment(env):
    s = env["session"]
    top = s.latest_retrieval(env["railway"]).evidence_ids[0]  # rank 1, highest BM25
    with pytest.raises(VerificationError, match="no evidence assessment"):
        build_verification(
            s,
            "r1",
            VerificationInput(
                claim_id=env["railway"],
                verdict=Verdict.SUPPORTED,
                rationale="The railway passage states it is 48 km long.",
                supporting_evidence_ids=[top],
            ),
        )
    with pytest.raises(VerificationError, match="assessed as UNCERTAIN"):
        build_verification(
            s,
            "r1",
            VerificationInput(
                claim_id=env["railway"],
                verdict=Verdict.SUPPORTED,
                rationale="The railway passage states it is 48 km long.",
                assessments=[_assess(top, EvidenceLabel.UNCERTAIN)],
                supporting_evidence_ids=[top],
            ),
        )


def test_supported_with_explicit_assessment(env):
    s = env["session"]
    eid = evidence_containing(s, env["railway"], "48 km long")
    v, warnings = build_verification(
        s,
        "r1",
        VerificationInput(
            claim_id=env["railway"],
            verdict=Verdict.SUPPORTED,
            rationale="The railway passage states it is 48 km long.",
            assessments=[_assess(eid, EvidenceLabel.SUPPORTING)],
            supporting_evidence_ids=[eid],
        ),
        checks=check_session_evidence(s, env["catalog"]),
    )
    assert warnings == []
    assert v.session_id == s.session_id and v.reviewer_id == "r1"
    assert v.evidence_assessments[0].passage_sha256  # hash of the passage actually read
    assert any("synthetic test fixture" in n for n in v.noted_limitations)


def test_contradicted_requires_contradicting_evidence(env):
    s = env["session"]
    with pytest.raises(VerificationError, match="CONTRADICTED"):
        build_verification(
            s,
            "r1",
            VerificationInput(
                claim_id=env["population"],
                verdict=Verdict.CONTRADICTED,
                rationale="Nothing in the sources mentions the population.",
            ),
        )


def test_insufficient_evidence_without_citations(env):
    s = env["session"]
    ids = s.latest_retrieval(env["population"]).evidence_ids
    v, _ = build_verification(
        s,
        "r1",
        VerificationInput(
            claim_id=env["population"],
            verdict=Verdict.INSUFFICIENT_EVIDENCE,
            rationale="No retrieved passage gives any population figure for Brindle.",
            assessments=[_assess(e, EvidenceLabel.IRRELEVANT) for e in ids],
        ),
    )
    assert v.cited_evidence_ids == [] and len(v.evidence_assessments) == len(ids)


def test_evidence_from_another_claim_rejected(env):
    s = env["session"]
    foreign = evidence_containing(s, env["railway"], "48 km long")
    with pytest.raises(VerificationError, match="not retrieved for this claim"):
        build_verification(
            s,
            "r1",
            VerificationInput(
                claim_id=env["bridge"],
                verdict=Verdict.SUPPORTED,
                rationale="Bridge passage 1874.",
                assessments=[_assess(foreign, EvidenceLabel.SUPPORTING)],
                supporting_evidence_ids=[foreign],
            ),
        )


@pytest.mark.parametrize("rationale", ["", "n/a", "tbd", "4/5"])
def test_bad_rationales_rejected(env, rationale):
    with pytest.raises(VerificationError, match="rationale rejected"):
        build_verification(
            env["session"],
            "r1",
            VerificationInput(
                claim_id=env["population"],
                verdict=Verdict.INSUFFICIENT_EVIDENCE,
                rationale=rationale,
            ),
        )


def test_missing_reviewer(env):
    with pytest.raises(VerificationError, match="reviewer ID"):
        build_verification(
            env["session"],
            " ",
            VerificationInput(
                claim_id=env["population"],
                verdict=Verdict.NOT_CHECKABLE,
                rationale="Too vague to be objective.",
            ),
        )


def test_duplicate_active_verification_requires_revision(env):
    v, _ = _contradict_bridge(env)
    env["session"] = record_verification(env["session"], v)
    with pytest.raises(VerificationError, match=f"--revise {v.verification_id}"):
        _contradict_bridge(env, now=utc_now() + timedelta(seconds=1))


def test_revision_chain_and_supersession(env):
    v1, _ = _contradict_bridge(env)
    env["session"] = record_verification(env["session"], v1)
    later = utc_now() + timedelta(seconds=5)
    v2, _ = _contradict_bridge(
        env,
        revision_of=v1.verification_id,
        revision_reason="Added the second bridge context.",
        now=later,
    )
    session = record_verification(env["session"], v2)
    assert session.human_verdict(env["bridge"]) == v2
    assert session.get_verification(v1.verification_id) == v1  # original kept intact
    progress = compute_progress(session)
    assert progress.verified_by_human == 1  # superseded record not counted again
    assert progress.revised_records == 1 and progress.superseded_records == 1
    env["session"] = session
    with pytest.raises(VerificationError, match="already revised"):
        _contradict_bridge(
            env,
            revision_of=v1.verification_id,
            revision_reason="Another fix.",
            now=later + timedelta(seconds=5),
        )


def test_revision_needs_correction_reason(env):
    v1, _ = _contradict_bridge(env)
    env["session"] = record_verification(env["session"], v1)
    with pytest.raises(VerificationError, match="correction reason"):
        _contradict_bridge(
            env,
            revision_of=v1.verification_id,
            revision_reason="tbd",
            now=utc_now() + timedelta(seconds=1),
        )


def test_cross_reviewer_revision(env):
    v1, _ = _contradict_bridge(env, reviewer="alice")
    env["session"] = record_verification(env["session"], v1)
    later = utc_now() + timedelta(seconds=2)
    with pytest.raises(VerificationError, match="--authorization"):
        _contradict_bridge(
            env,
            reviewer="bob",
            revision_of=v1.verification_id,
            revision_reason="Second opinion.",
            now=later,
        )
    v2, _ = _contradict_bridge(
        env,
        reviewer="bob",
        revision_of=v1.verification_id,
        revision_reason="Second opinion.",
        now=later,
        revision_authorization="Approved by lead reviewer alice.",
    )
    assert record_verification(env["session"], v2).human_verdict(env["bridge"]).reviewer_id == "bob"


def test_modified_source_blocks_citation(env):
    doc = env["catalog"].documents_dir("testrefs") / "lake_verenna.md"
    doc.write_text(doc.read_text().replace("1874", "1875"))
    checks = check_session_evidence(env["session"], env["catalog"], env["bridge"])
    blocked = [c for c in checks.values() if c.status is SourceStatus.MODIFIED]
    assert blocked and all(c.blocks_citation for c in blocked)
    with pytest.raises(VerificationError, match="cannot be cited"):
        _contradict_bridge(env)


def test_missing_collection_is_a_visible_limitation_not_a_block(env, settings):
    from hallucination_lab.config import Settings
    from hallucination_lab.retrieval.catalog import CollectionCatalog

    elsewhere = CollectionCatalog(Settings(data_dir=settings.data_dir / "other"))
    env["catalog"] = elsewhere
    v, _ = _contradict_bridge(env)
    assert any("cannot be re-checked" in n for n in v.noted_limitations)


def test_drafts_saved_and_discarded(env):
    now = utc_now()
    draft = VerificationDraft(
        claim_id=env["bridge"],
        reviewer_id="r1",
        started_at=now,
        updated_at=now,
        verdict=Verdict.CONTRADICTED,
    )
    session = save_draft(env["session"], draft)
    assert session.get_draft(env["bridge"], "r1") == draft
    assert compute_progress(session).drafts_in_progress == 1
    assert compute_progress(session).verified_by_human == 0
    assert discard_draft(session, env["bridge"], "r1").verification_drafts == []


def test_recording_removes_the_draft(env):
    now = utc_now()
    session = save_draft(
        env["session"],
        VerificationDraft(claim_id=env["bridge"], reviewer_id="r1", started_at=now, updated_at=now),
    )
    env["session"] = session
    v, _ = _contradict_bridge(env, now=now + timedelta(seconds=1))
    assert record_verification(session, v).verification_drafts == []


def test_progress_counts(env):
    s = env["session"]
    p = compute_progress(s)
    assert (p.total_candidate_claims, p.marked_for_verification, p.awaiting_retrieval) == (3, 3, 0)
    assert p.with_retrieval_results == 3 and p.awaiting_human_review == 3 and not p.complete
    v, _ = _contradict_bridge(env)
    p = compute_progress(record_verification(s, v))
    assert p.verified_by_human == 1 and p.awaiting_human_review == 2
    assert p.verdict_counts["CONTRADICTED"] == 1 and not p.complete


def test_summary_json(env):
    v, _ = _contradict_bridge(env)
    session = record_verification(env["session"], v)
    summary = verification_summary(session)
    assert summary["complete"] is False
    (record,) = summary["records"]
    assert record["verdict"] == "CONTRADICTED" and record["reviewer_id"] == "r1"
    used = record["evidence_used"][0]
    assert used["role"] == "contradicting" and "1874" in used["quoted_passage"]
    assert used["document_reference"].startswith("testrefs/lake_verenna.md#L")
    assert record["revision_number"] == 1
