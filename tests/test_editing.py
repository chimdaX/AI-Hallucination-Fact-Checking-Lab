"""Human corrections to the extracted claim list."""

import pytest

from hallucination_lab.errors import ClaimEditError
from hallucination_lab.extraction.editing import (
    add_manual_claim,
    edit_claim,
    merge_claims,
    remove_claim,
    restore_claim,
    set_requires_verification,
    split_claim,
)
from hallucination_lab.models import EditType, ExtractionMethod, VerificationSession

R = "reviewer-1"


def _valid(session: VerificationSession, claims) -> VerificationSession:
    return VerificationSession.model_validate(
        session.model_copy(update={"claims": claims}).model_dump()
    )


def test_edit_keeps_original_and_records_history(session):
    claim = session.claims[1]
    claims = edit_claim(session.claims, claim.claim_id, "A leap year has 366 days.", R, "typo")
    edited = claims[1]
    assert edited.claim_id == claim.claim_id
    assert edited.original_text == claim.original_text
    assert edited.normalized_claim == "A leap year has 366 days."
    assert edited.features.numbers == ["366"]
    assert edited.reviewer_edits[-1].previous_text == claim.normalized_claim
    assert edited.is_human_modified
    _valid(session, claims)


def test_edit_rejects_empty_text(session):
    with pytest.raises(ClaimEditError):
        edit_claim(session.claims, session.claims[0].claim_id, "   ", R)


def test_split_preserves_provenance(session):
    parent = session.claims[0]
    claims = split_claim(
        session.claims,
        parent.claim_id,
        ["The Eiffel Tower was completed in 1889.", "The Eiffel Tower is in Paris."],
        R,
    )
    retired = next(c for c in claims if c.claim_id == parent.claim_id)
    children = [
        c
        for c in claims
        if c.is_active
        and parent.claim_id in [i for e in c.reviewer_edits for i in e.related_claim_ids]
    ]
    assert not retired.is_active
    assert retired.reviewer_edits[-1].edit_type is EditType.SPLIT_INTO
    assert len(children) == 2
    for child in children:
        assert child.original_text == parent.original_text
        assert child.span == parent.span
    _valid(session, claims)


def test_split_needs_two_parts(session):
    with pytest.raises(ClaimEditError):
        split_claim(session.claims, session.claims[0].claim_id, ["only one"], R)


def test_merge(session):
    a, b = session.claims[0], session.claims[1]
    claims = merge_claims(session.claims, [a.claim_id, b.claim_id], "Combined fact.", R)
    merged = [
        c
        for c in claims
        if c.is_active
        and c.reviewer_edits
        and c.reviewer_edits[0].edit_type is EditType.MERGED_FROM
    ]
    assert len(merged) == 1
    assert merged[0].span.start == a.span.start and merged[0].span.end == b.span.end
    assert not any(c.is_active for c in claims if c.claim_id in {a.claim_id, b.claim_id})
    _valid(session, claims)


def test_merge_requires_two_distinct(session):
    cid = session.claims[0].claim_id
    with pytest.raises(ClaimEditError):
        merge_claims(session.claims, [cid, cid], "x", R)


def test_add_manual_claim_verbatim_and_not(session, response):
    claims = add_manual_claim(session.claims, response, "World's Fair in Paris", R)
    verbatim = claims[-1]
    assert verbatim.extraction_method is ExtractionMethod.MANUAL
    assert response.text[verbatim.span.start : verbatim.span.end] == "World's Fair in Paris"

    claims = add_manual_claim(claims, response, "Gustave Eiffel's firm built the tower.", R)
    assert claims[-1].span is None
    assert "not verbatim" in claims[-1].extraction_notes[0]
    _valid(session, claims)

    with pytest.raises(ClaimEditError):
        add_manual_claim(claims, response, "World's Fair in Paris", R)


def test_remove_and_restore_never_delete(session):
    cid = session.claims[0].claim_id
    claims = remove_claim(session.claims, cid, R, "not relevant")
    assert len(claims) == len(session.claims)
    assert not claims[0].is_active
    with pytest.raises(ClaimEditError, match="inactive"):
        edit_claim(claims, cid, "new", R)
    claims = restore_claim(claims, cid, R)
    assert claims[0].is_active
    assert [e.edit_type for e in claims[0].reviewer_edits] == [EditType.REMOVED, EditType.RESTORED]


def test_override_non_checkable_flag(session):
    opinion = next(c for c in session.claims if not c.requires_verification)
    claims = set_requires_verification(session.claims, opinion.claim_id, True, R, "contains fact")
    updated = next(c for c in claims if c.claim_id == opinion.claim_id)
    assert updated.requires_verification and updated.non_checkable_reason is None
    assert updated.reviewer_edits[-1].edit_type is EditType.VERIFICATION_FLAG_CHANGED


def test_unknown_claim_id(session):
    with pytest.raises(ClaimEditError, match="unknown claim"):
        remove_claim(session.claims, "clm_ffffffffffffffff", R)
