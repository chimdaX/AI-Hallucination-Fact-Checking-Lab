"""Workspace service: the layer the web interface uses for every read and write."""

from datetime import timedelta

import pytest

from hallucination_lab.errors import (
    ClaimEditError,
    ExtractionError,
    InvalidIdentifierError,
    SessionConflictError,
    SessionNotFoundError,
)
from hallucination_lab.models import (
    EvidenceAssessment,
    EvidenceLabel,
    ResponseProvenance,
    Verdict,
    VerificationDraft,
    utc_now,
)
from hallucination_lab.services.workspace import Workspace
from hallucination_lab.verification.workflow import VerificationInput
from tests.conftest import evidence_containing


@pytest.fixture
def ws(settings, test_collection):
    return Workspace(settings)


@pytest.fixture
def loaded(ws, verification_env):
    return ws.load(verification_env["session_id"])


def _contradict(ws, loaded, env, **kw):
    eid = evidence_containing(loaded.session, env["bridge"], "completed in 1874")
    inp = VerificationInput(
        claim_id=env["bridge"],
        verdict=Verdict.CONTRADICTED,
        rationale="The passage dates the first bridge to 1874, not 1880.",
        assessments=[EvidenceAssessment(evidence_id=eid, label=EvidenceLabel.CONTRADICTING)],
        contradicting_evidence_ids=[eid],
        **kw,
    )
    return ws.prepare_verification(loaded, "r1", inp)


def test_create_session_stores_text_verbatim(ws):
    text = "  Lake Verenna is deep.\n\nThe Osterland Railway opened in 1902.  "
    loaded = ws.create_session(text, title="t", synthetic=True)
    assert loaded.session.response.text == text
    assert loaded.session.response.provenance is ResponseProvenance.SYNTHETIC_FIXTURE
    assert ws.load(loaded.session_id).session == loaded.session


@pytest.mark.parametrize("text", ["", "   \n"])
def test_empty_submission_rejected(ws, text):
    with pytest.raises(ExtractionError, match="empty"):
        ws.create_session(text)


def test_text_without_claims_still_creates_session(ws):
    loaded = ws.create_session("???")
    assert all(not c.requires_verification for c in loaded.session.claims)


@pytest.mark.parametrize("bad", ["../etc", "nope", ""])
def test_invalid_session_ids(ws, bad):
    with pytest.raises((InvalidIdentifierError, SessionNotFoundError)):
        ws.load(bad)


def test_missing_session(ws):
    with pytest.raises(SessionNotFoundError):
        ws.load("ses_0000000000000000")


def test_stale_view_cannot_overwrite(ws, loaded, verification_env):
    newer = ws.add_claim(loaded, "Brindle has a harbour.", "r1")
    assert len(newer.session.claims) == len(loaded.session.claims) + 1
    with pytest.raises(SessionConflictError):
        ws.add_claim(loaded, "Another claim from an old view.", "r1")  # stale version


def test_claim_editing_through_service(ws, loaded, verification_env):
    cid = verification_env["population"]
    loaded = ws.edit_claim(loaded, cid, "Brindle has about 90,000 residents.", "r1")
    assert loaded.session.get_claim(cid).normalized_claim == "Brindle has about 90,000 residents."
    loaded = ws.set_flag(loaded, cid, False, "r1")
    assert not loaded.session.get_claim(cid).requires_verification
    loaded = ws.remove_claim(loaded, cid, "r1")
    assert not loaded.session.get_claim(cid).is_active
    loaded = ws.restore_claim(loaded, cid, "r1")
    loaded = ws.split_claim(loaded, cid, ["Brindle is a town.", "Brindle has 90,000 people."], "r1")
    assert len(loaded.session.active_claims) == 4


def test_verified_claim_cannot_be_edited(ws, loaded, verification_env):
    env = verification_env
    v, _ = _contradict(ws, loaded, env)
    loaded = ws.commit_verification(loaded, v)
    for action in (
        lambda: ws.edit_claim(loaded, env["bridge"], "New wording.", "r1"),
        lambda: ws.remove_claim(loaded, env["bridge"], "r1"),
        lambda: ws.set_flag(loaded, env["bridge"], False, "r1"),
        lambda: ws.split_claim(loaded, env["bridge"], ["a b c", "d e f"], "r1"),
    ):
        with pytest.raises(ClaimEditError, match="human verdict"):
            action()


def test_claim_with_draft_cannot_be_edited(ws, loaded, verification_env):
    now = utc_now()
    loaded = ws.save_draft(
        loaded,
        VerificationDraft(
            claim_id=verification_env["bridge"], reviewer_id="r1", started_at=now, updated_at=now
        ),
    )
    with pytest.raises(ClaimEditError, match="draft"):
        ws.edit_claim(loaded, verification_env["bridge"], "Changed.", "r1")


def test_retrieval_through_service(ws, verification_env):
    loaded = ws.create_session("The Osterland Railway is 48 km long.")
    updated, run, _ = ws.retrieve(loaded, "testrefs")
    assert run.results[0].outcome.value == "candidates_found"
    assert updated.session.evidence and updated.session.verifications == []


def test_commit_is_idempotent_and_blocks_second_verdict(ws, loaded, verification_env):
    v, _ = _contradict(ws, loaded, verification_env)
    once = ws.commit_verification(loaded, v)
    again = ws.commit_verification(once, v)  # repeated click with the same record
    assert len(again.session.verifications) == 1
    from hallucination_lab.errors import VerificationError

    with pytest.raises(VerificationError, match="--revise"):
        _contradict(ws, again, verification_env)


def test_revision_through_service(ws, loaded, verification_env):
    v1, _ = _contradict(ws, loaded, verification_env)
    loaded = ws.commit_verification(loaded, v1)
    eid = v1.contradicting_evidence_ids[0]
    inp = VerificationInput(
        claim_id=verification_env["bridge"],
        verdict=Verdict.CONTRADICTED,
        rationale="The passage gives 1874 for the first bridge; the claim says 1880.",
        assessments=[EvidenceAssessment(evidence_id=eid, label=EvidenceLabel.CONTRADICTING)],
        contradicting_evidence_ids=[eid],
        revision_of=v1.verification_id,
        revision_reason="Clarified which bridge the passage refers to.",
    )
    v2, _ = ws.prepare_verification(loaded, "r1", inp)
    assert v2.verification_timestamp >= v1.verification_timestamp - timedelta(seconds=1)
    loaded = ws.commit_verification(loaded, v2)
    assert loaded.session.human_verdict(verification_env["bridge"]) == v2
    assert loaded.session.get_verification(v1.verification_id) == v1


def test_draft_save_and_discard(ws, loaded, verification_env):
    now = utc_now()
    draft = VerificationDraft(
        claim_id=verification_env["bridge"],
        reviewer_id="r1",
        started_at=now,
        updated_at=now,
        rationale="half-written",
    )
    loaded = ws.save_draft(loaded, draft)
    assert ws.load(loaded.session_id).session.get_draft(verification_env["bridge"], "r1")
    loaded = ws.discard_draft(loaded, verification_env["bridge"], "r1")
    assert loaded.session.verification_drafts == []


def test_list_sessions_reports_corrupt_files(ws, verification_env, settings):
    (settings.sessions_dir / "ses_ffffffffffffffff.json").write_text("{broken")
    summaries, problems = ws.list_sessions()
    assert [s.session_id for s in summaries] == [verification_env["session_id"]]
    assert len(problems) == 1
