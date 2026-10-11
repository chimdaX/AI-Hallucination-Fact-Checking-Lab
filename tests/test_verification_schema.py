"""Schema rules for human verification records, revisions and drafts."""

from datetime import timedelta

import pytest
from pydantic import ValidationError

from hallucination_lab.ids import VERIFICATION, make_id
from hallucination_lab.models import (
    EvidenceAssessment,
    EvidenceLabel,
    Provenance,
    Verdict,
    Verification,
    VerificationDraft,
    VerificationMethod,
    VerificationSession,
)
from tests.conftest import FIXED_TIME, make_evidence, make_verification

CID = "clm_0000000000000001"
EID = "evd_0000000000000001"
SID = "ses_0000000000000001"


def _human(
    verdict=Verdict.SUPPORTED,
    supporting=(EID,),
    contradicting=(),
    assessments=None,
    session_id=SID,
    **kw,
):
    data = dict(
        verification_id=kw.pop("verification_id", make_id(VERIFICATION, "x", str(kw))),
        claim_id=CID,
        verdict=verdict,
        reasoning="Passage gives 1874, matching the claim.",
        supporting_evidence_ids=list(supporting),
        contradicting_evidence_ids=list(contradicting),
        verification_method=VerificationMethod.HUMAN,
        reviewer_id="r1",
        verification_timestamp=FIXED_TIME,
        provenance=Provenance(method_detail="test"),
        session_id=session_id,
        evidence_assessments=(
            assessments
            if assessments is not None
            else [EvidenceAssessment(evidence_id=EID, label=EvidenceLabel.SUPPORTING)]
        ),
    )
    data.update(kw)
    return Verification(**data)


def test_valid_human_record():
    v = _human()
    assert v.is_human_judgment and v.cited_evidence_ids == [EID]


def test_cited_evidence_must_be_explicitly_assessed():
    # The core guarantee: retrieval alone never makes a passage "supporting".
    with pytest.raises(ValidationError, match="never treated as supporting automatically"):
        _human(assessments=[])


def test_cited_label_must_match_assessment():
    with pytest.raises(ValidationError, match="assessed as IRRELEVANT"):
        _human(assessments=[EvidenceAssessment(evidence_id=EID, label=EvidenceLabel.IRRELEVANT)])
    with pytest.raises(ValidationError, match="assessed as SUPPORTING"):
        _human(verdict=Verdict.CONTRADICTED, supporting=(), contradicting=(EID,))


def test_duplicate_assessments_rejected():
    a = EvidenceAssessment(evidence_id=EID, label=EvidenceLabel.SUPPORTING)
    with pytest.raises(ValidationError, match="more than once"):
        _human(assessments=[a, a])


def test_legacy_records_without_session_id_keep_old_rules():
    assert _human(session_id=None, assessments=[]).session_id is None


@pytest.mark.parametrize("reasoning", ["n/a", "tbd", "5/5", "correct"])
def test_placeholder_reasoning_rejected_by_schema(reasoning):
    with pytest.raises(ValidationError, match="reasoning rejected"):
        _human(reasoning=reasoning)


def test_unsupported_and_uncontradicted_verdicts_need_evidence():
    with pytest.raises(ValidationError, match="SUPPORTED"):
        _human(supporting=(), assessments=[])
    with pytest.raises(ValidationError, match="CONTRADICTED"):
        _human(verdict=Verdict.CONTRADICTED, supporting=(), assessments=[])


def test_missing_reviewer_rejected():
    with pytest.raises(ValidationError, match="reviewer_id"):
        _human(reviewer_id="  ")


def test_revision_requires_reason():
    with pytest.raises(ValidationError, match="correction reason"):
        _human(revision_of="ver_0000000000000009")
    with pytest.raises(ValidationError, match="correction reason"):
        _human(revision_of="ver_0000000000000009", revision_reason="tbd")
    with pytest.raises(ValidationError, match="require revision_of"):
        _human(revision_reason="Typo in the passage reference.")


def test_cannot_revise_itself():
    vid = "ver_0000000000000005"
    with pytest.raises(ValidationError, match="cannot revise itself"):
        _human(verification_id=vid, revision_of=vid, revision_reason="Fixing an error.")


# ----------------------------------------------------------------- session level


@pytest.fixture
def base(session):
    cid = session.claims[0].claim_id
    ev = make_evidence(cid)
    s = session.model_copy(update={"evidence": [ev]})
    return s, cid, ev.evidence_id


def _record(
    s,
    cid,
    eid,
    n=0,
    reviewer="r1",
    revision_of=None,
    authorization=None,
    verdict=Verdict.SUPPORTED,
    minutes=0,
):
    label = (
        EvidenceLabel.SUPPORTING if verdict is Verdict.SUPPORTED else (EvidenceLabel.CONTRADICTING)
    )
    return Verification(
        verification_id=make_id(VERIFICATION, "rec", n),
        claim_id=cid,
        verdict=verdict,
        reasoning="The passage states the completion year 1889 directly.",
        supporting_evidence_ids=[eid] if verdict is Verdict.SUPPORTED else [],
        contradicting_evidence_ids=[eid] if verdict is Verdict.CONTRADICTED else [],
        verification_method=VerificationMethod.HUMAN,
        reviewer_id=reviewer,
        verification_timestamp=FIXED_TIME + timedelta(minutes=minutes),
        provenance=Provenance(method_detail="test"),
        session_id=s.session_id,
        evidence_assessments=[EvidenceAssessment(evidence_id=eid, label=label)],
        revision_of=revision_of,
        revision_authorization=authorization,
        revision_reason="Re-read the passage and corrected my verdict." if revision_of else None,
    )


def _validate(s, records, drafts=()):
    return VerificationSession.model_validate(
        s.model_copy(
            update={"verifications": list(records), "verification_drafts": list(drafts)}
        ).model_dump()
    )


def test_revision_chain_supersedes_and_keeps_history(base):
    s, cid, eid = base
    v1 = _record(s, cid, eid, 1)
    v2 = _record(
        s, cid, eid, 2, revision_of=v1.verification_id, verdict=Verdict.CONTRADICTED, minutes=1
    )
    v3 = _record(s, cid, eid, 3, revision_of=v2.verification_id, minutes=2)
    session = _validate(s, [v1, v2, v3])
    assert session.human_verdict(cid) == v3
    assert [v.verification_id for v in session.revision_chain(v3.verification_id)] == [
        v1.verification_id,
        v2.verification_id,
        v3.verification_id,
    ]
    assert session.superseded_ids() == {v1.verification_id, v2.verification_id}
    assert cid not in {c.claim_id for c in session.pending_review()}


def test_branching_revisions_rejected(base):
    s, cid, eid = base
    v1 = _record(s, cid, eid, 1)
    a = _record(s, cid, eid, 2, revision_of=v1.verification_id, minutes=1)
    b = _record(s, cid, eid, 3, revision_of=v1.verification_id, minutes=2)
    with pytest.raises(ValidationError, match="branches"):
        _validate(s, [v1, a, b])


def test_circular_revisions_rejected(base):
    s, cid, eid = base
    v1 = _record(s, cid, eid, 1, revision_of=make_id(VERIFICATION, "rec", 2))
    v2 = _record(s, cid, eid, 2, revision_of=v1.verification_id, minutes=1)
    with pytest.raises(ValidationError):
        _validate(s, [v1, v2])


def test_revision_must_be_later_and_same_claim(base):
    s, cid, eid = base
    v1 = _record(s, cid, eid, 1, minutes=5)
    early = _record(s, cid, eid, 2, revision_of=v1.verification_id, minutes=1)
    with pytest.raises(ValidationError, match="later"):
        _validate(s, [v1, early])
    with pytest.raises(ValidationError, match="unknown verification"):
        _validate(s, [_record(s, cid, eid, 9, revision_of="ver_ffffffffffffffff", minutes=1)])


def test_cross_reviewer_revision_needs_authorization(base):
    s, cid, eid = base
    v1 = _record(s, cid, eid, 1, reviewer="alice")
    by_bob = _record(s, cid, eid, 2, reviewer="bob", revision_of=v1.verification_id, minutes=1)
    with pytest.raises(ValidationError, match="without a recorded revision_authorization"):
        _validate(s, [v1, by_bob])
    authorized = _record(
        s,
        cid,
        eid,
        3,
        reviewer="bob",
        revision_of=v1.verification_id,
        minutes=1,
        authorization="Lead reviewer approved re-review.",
    )
    assert _validate(s, [v1, authorized]).human_verdict(cid).reviewer_id == "bob"


def test_duplicate_active_verifications_rejected(base):
    s, cid, eid = base
    with pytest.raises(ValidationError, match="active human verifications"):
        _validate(s, [_record(s, cid, eid, 1), _record(s, cid, eid, 2, minutes=1)])


def test_legacy_duplicates_still_load(base):
    s, cid, eid = base
    old1 = make_verification(cid, Verdict.SUPPORTED, supporting=[eid])
    old2 = make_verification(
        cid, Verdict.INSUFFICIENT_EVIDENCE, timestamp=FIXED_TIME + timedelta(hours=1)
    )
    assert _validate(s, [old1, old2]).human_verdict(cid) == old2


def test_record_must_belong_to_session(base):
    s, cid, eid = base
    v = _record(s, cid, eid, 1).model_copy(update={"session_id": "ses_ffffffffffffffff"})
    with pytest.raises(ValidationError, match="belongs to session"):
        _validate(s, [v])


def test_assessed_evidence_must_exist_for_claim(base):
    s, cid, eid = base
    v = _record(s, cid, eid, 1)
    bad = v.model_copy(
        update={
            "evidence_assessments": [
                *v.evidence_assessments,
                EvidenceAssessment(
                    evidence_id="evd_ffffffffffffffff", label=EvidenceLabel.IRRELEVANT
                ),
            ]
        }
    )
    with pytest.raises(ValidationError, match="unknown evidence"):
        _validate(s, [bad])


def test_drafts_validated_and_unique(base):
    s, cid, eid = base
    draft = VerificationDraft(
        claim_id=cid,
        reviewer_id="r1",
        started_at=FIXED_TIME,
        updated_at=FIXED_TIME,
        evidence_assessments=[EvidenceAssessment(evidence_id=eid, label=EvidenceLabel.UNCERTAIN)],
    )
    session = _validate(s, [], [draft])
    assert session.get_draft(cid, "r1") == draft
    assert session.pending_review()  # a draft is not a verification
    with pytest.raises(ValidationError, match="more than one draft"):
        _validate(s, [], [draft, draft])
    foreign = draft.model_copy(
        update={
            "evidence_assessments": [
                EvidenceAssessment(
                    evidence_id="evd_ffffffffffffffff", label=EvidenceLabel.UNCERTAIN
                )
            ]
        }
    )
    with pytest.raises(ValidationError, match="foreign evidence"):
        _validate(s, [], [foreign])
