"""Schema validation: the integrity rules must be enforced, not just documented."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from hallucination_lab.ids import response_id_for
from hallucination_lab.models import (
    SCHEMA_VERSION,
    AIResponse,
    Claim,
    ExtractionMethod,
    NonCheckableReason,
    Provenance,
    ResponseProvenance,
    Source,
    TextSpan,
    Verdict,
    VerificationMethod,
    VerificationSession,
)
from tests.conftest import FIXED_TIME, make_evidence, make_verification

# ---------------------------------------------------------------- AIResponse


def test_response_text_is_preserved_exactly():
    text = "  Leading spaces and trailing newline.\n"
    assert AIResponse.from_text(text).text == text


def test_response_id_must_match_text():
    with pytest.raises(ValidationError, match="does not match the text hash"):
        AIResponse(
            response_id=response_id_for("original"),
            text="tampered",
            provenance=ResponseProvenance.USER_SUBMITTED,
        )


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_blank_response_rejected(text):
    with pytest.raises(ValidationError):
        AIResponse.from_text(text)


def test_authentic_output_requires_model_name():
    with pytest.raises(ValidationError, match="must name the model"):
        AIResponse.from_text("Some output.", ResponseProvenance.AUTHENTIC_MODEL_OUTPUT)


def test_unknown_fields_rejected():
    with pytest.raises(ValidationError):
        AIResponse.from_text("Text here.", unexpected="x")


# --------------------------------------------------------------------- Claim


def _claim(**overrides) -> Claim:
    data = dict(
        claim_id="clm_0000000000000001",
        original_text="Paris is in France.",
        normalized_claim="Paris is in France.",
        source_response_id="rsp_0000000000000001",
        extraction_method=ExtractionMethod.RULE_BASED,
        requires_verification=True,
    )
    data.update(overrides)
    return Claim(**data)


def test_valid_claim_defaults():
    claim = _claim()
    assert claim.extraction_confidence is None
    assert claim.is_active and not claim.is_human_modified


def test_confidence_without_calibration_rejected():
    with pytest.raises(ValidationError, match="confidence_calibration"):
        _claim(extraction_confidence=0.9)


def test_confidence_with_calibration_accepted():
    claim = _claim(extraction_confidence=0.8, confidence_calibration="isotonic on dev set v1")
    assert claim.extraction_confidence == 0.8


def test_checkable_claim_cannot_have_non_checkable_reason():
    with pytest.raises(ValidationError):
        _claim(non_checkable_reason=NonCheckableReason.OPINION)


@pytest.mark.parametrize("start,end", [(5, 5), (10, 2)])
def test_bad_spans_rejected(start, end):
    with pytest.raises(ValidationError):
        TextSpan(start=start, end=end)


def test_malformed_claim_id_rejected():
    with pytest.raises(ValidationError):
        _claim(claim_id="claim-1")


# ---------------------------------------------------------- Source / Evidence


def test_source_requires_url_or_reference():
    with pytest.raises(ValidationError, match="url or a document_reference"):
        Source(source_id="src_0000000000000001", title="Untraceable")


def test_source_rejects_non_http_url():
    with pytest.raises(ValidationError):
        Source(source_id="src_0000000000000001", title="X", url="javascript:alert(1)")


def test_evidence_requires_attribution():
    ev = make_evidence("clm_0000000000000001")
    data = ev.model_dump()
    data["document_reference"] = None
    with pytest.raises(ValidationError, match="attribution"):
        type(ev).model_validate(data)


def test_naive_timestamp_rejected():
    ev = make_evidence("clm_0000000000000001")
    data = ev.model_dump()
    data["retrieved_at"] = datetime(2026, 1, 1, 12, 0)
    with pytest.raises(ValidationError, match="timezone-aware"):
        type(ev).model_validate(data)


# ---------------------------------------------------------------- Verification

CID = "clm_0000000000000001"
EID = "evd_0000000000000001"


def test_supported_requires_supporting_evidence():
    with pytest.raises(ValidationError, match="SUPPORTED"):
        make_verification(CID, Verdict.SUPPORTED)


def test_contradicted_requires_contradicting_evidence():
    # Missing evidence must never become a contradiction by default.
    with pytest.raises(ValidationError, match="CONTRADICTED"):
        make_verification(CID, Verdict.CONTRADICTED, supporting=[EID])


def test_insufficient_evidence_needs_no_evidence():
    v = make_verification(CID, Verdict.INSUFFICIENT_EVIDENCE)
    assert v.verdict is Verdict.INSUFFICIENT_EVIDENCE


def test_human_verification_requires_reviewer():
    with pytest.raises(ValidationError, match="reviewer_id"):
        make_verification(CID, Verdict.NOT_CHECKABLE, reviewer_id=None)


def test_automated_verification_must_not_claim_reviewer():
    with pytest.raises(ValidationError, match="must not carry a reviewer_id"):
        make_verification(
            CID, Verdict.SUPPORTED, supporting=[EID], method=VerificationMethod.AUTOMATED_RULE
        )


def test_automated_supported_still_requires_evidence():
    with pytest.raises(ValidationError):
        make_verification(
            CID, Verdict.SUPPORTED, method=VerificationMethod.AUTOMATED_RULE, reviewer_id=None
        )


def test_model_suggestion_requires_model_name():
    with pytest.raises(ValidationError, match="model_name"):
        make_verification(
            CID,
            Verdict.INSUFFICIENT_EVIDENCE,
            method=VerificationMethod.MODEL_SUGGESTION,
            reviewer_id=None,
        )


def test_evidence_cannot_both_support_and_contradict():
    with pytest.raises(ValidationError, match="both support and contradict"):
        make_verification(CID, Verdict.SUPPORTED, supporting=[EID], contradicting=[EID])


def test_blank_reasoning_rejected():
    v = make_verification(CID, Verdict.NOT_CHECKABLE)
    data = v.model_dump()
    data["reasoning"] = "   "
    with pytest.raises(ValidationError):
        type(v).model_validate(data)


def test_provenance_records_tool_version():
    assert Provenance(method_detail="x").tool_version


# ------------------------------------------------------------------- Session


def test_session_defaults(session):
    assert session.schema_version == SCHEMA_VERSION
    assert session.claims and all(
        c.source_response_id == session.response.response_id for c in session.claims
    )


def test_session_rejects_unknown_major_schema(session):
    data = session.model_dump()
    data["schema_version"] = "2.0.0"
    with pytest.raises(ValidationError, match="unsupported schema major"):
        VerificationSession.model_validate(data)


def test_session_rejects_evidence_for_unknown_claim(session):
    data = session.model_dump()
    data["evidence"] = [make_evidence("clm_ffffffffffffffff").model_dump()]
    with pytest.raises(ValidationError, match="unknown claim"):
        VerificationSession.model_validate(data)


def test_session_rejects_verification_citing_unknown_evidence(session):
    cid = session.claims[0].claim_id
    data = session.model_dump()
    data["verifications"] = [
        make_verification(cid, Verdict.SUPPORTED, supporting=[EID]).model_dump()
    ]
    with pytest.raises(ValidationError, match="unknown evidence"):
        VerificationSession.model_validate(data)


def test_session_rejects_evidence_from_another_claim(session):
    first, second = session.claims[0].claim_id, session.claims[1].claim_id
    ev = make_evidence(second)
    data = session.model_dump()
    data["evidence"] = [ev.model_dump()]
    data["verifications"] = [
        make_verification(first, Verdict.SUPPORTED, supporting=[ev.evidence_id]).model_dump()
    ]
    with pytest.raises(ValidationError, match="different claim"):
        VerificationSession.model_validate(data)


def test_session_rejects_duplicate_claim_ids(session):
    data = session.model_dump()
    data["claims"] = [data["claims"][0], data["claims"][0]]
    with pytest.raises(ValidationError, match="duplicate claim_id"):
        VerificationSession.model_validate(data)


def test_session_rejects_claim_from_other_response(session):
    data = session.model_dump()
    data["claims"][0]["source_response_id"] = "rsp_ffffffffffffffff"
    with pytest.raises(ValidationError, match="different response"):
        VerificationSession.model_validate(data)


def test_human_verdict_ignores_automated_suggestions(session):
    cid = session.claims[0].claim_id
    ev = make_evidence(cid)
    auto = make_verification(
        cid,
        Verdict.SUPPORTED,
        supporting=[ev.evidence_id],
        method=VerificationMethod.AUTOMATED_RULE,
        reviewer_id=None,
    )
    s = session.model_copy(update={"evidence": [ev], "verifications": [auto]})
    s = VerificationSession.model_validate(s.model_dump())
    assert s.human_verdict(cid) is None
    assert cid in {c.claim_id for c in s.pending_review()}
    assert s.human_cited_evidence_ids() == set()  # retrieved, not verified


def test_latest_human_verdict_wins_and_clears_pending(session):
    cid = session.claims[0].claim_id
    ev = make_evidence(cid)
    first = make_verification(cid, Verdict.INSUFFICIENT_EVIDENCE)
    later = make_verification(
        cid,
        Verdict.SUPPORTED,
        supporting=[ev.evidence_id],
        timestamp=FIXED_TIME.replace(hour=13),
    )
    s = VerificationSession.model_validate(
        session.model_copy(update={"evidence": [ev], "verifications": [later, first]}).model_dump()
    )
    assert s.human_verdict(cid).verdict is Verdict.SUPPORTED
    assert cid not in {c.claim_id for c in s.pending_review()}
    assert s.human_cited_evidence_ids() == {ev.evidence_id}


def test_session_json_round_trip(session):
    restored = VerificationSession.model_validate_json(session.model_dump_json())
    assert restored == session
