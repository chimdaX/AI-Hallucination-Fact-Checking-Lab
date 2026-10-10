"""Human review operations on a claim list: edit, split, merge, add, remove, restore.

All functions are pure: they take a list of claims and return a new list. Nothing
is ever deleted. Removed, split and merged claims are kept with
``is_active=False`` and every change is appended to ``reviewer_edits``, so the
path from the original AI wording to the final verified claim stays auditable.
"""

from __future__ import annotations

from hallucination_lab.errors import ClaimEditError
from hallucination_lab.extraction.features import detect_features, temporal_context
from hallucination_lab.ids import CLAIM, make_id
from hallucination_lab.models import (
    AIResponse,
    Claim,
    EditType,
    ExtractionMethod,
    ReviewerEdit,
    TextSpan,
)


def _index(claims: list[Claim], claim_id: str) -> int:
    for i, claim in enumerate(claims):
        if claim.claim_id == claim_id:
            return i
    raise ClaimEditError(f"unknown claim {claim_id}")


def _clean(text: str, what: str = "claim text") -> str:
    cleaned = " ".join(text.split())
    if not cleaned:
        raise ClaimEditError(f"{what} must not be empty")
    return cleaned


def _require_active(claim: Claim) -> None:
    if not claim.is_active:
        raise ClaimEditError(f"claim {claim.claim_id} is inactive; restore it first")


def _with_text(text: str) -> dict:
    features = detect_features(text)
    return {
        "normalized_claim": text,
        "features": features,
        "temporal_context": temporal_context(features),
    }


def edit_claim(
    claims: list[Claim], claim_id: str, new_text: str, reviewer_id: str, note: str | None = None
) -> list[Claim]:
    """Reword a claim's normalized text. ``original_text`` is never changed."""
    i = _index(claims, claim_id)
    claim = claims[i]
    _require_active(claim)
    text = _clean(new_text)
    if text == claim.normalized_claim:
        return list(claims)
    edit = ReviewerEdit(
        edit_type=EditType.EDITED,
        reviewer_id=reviewer_id,
        previous_text=claim.normalized_claim,
        new_text=text,
        note=note,
    )
    updated = claim.model_copy(
        update={**_with_text(text), "reviewer_edits": [*claim.reviewer_edits, edit]}
    )
    return [*claims[:i], Claim.model_validate(updated.model_dump()), *claims[i + 1 :]]


def set_requires_verification(
    claims: list[Claim], claim_id: str, value: bool, reviewer_id: str, note: str | None = None
) -> list[Claim]:
    """Override the extractor's checkable/non-checkable flag."""
    i = _index(claims, claim_id)
    claim = claims[i]
    if claim.requires_verification == value:
        return list(claims)
    edit = ReviewerEdit(
        edit_type=EditType.VERIFICATION_FLAG_CHANGED,
        reviewer_id=reviewer_id,
        previous_text=str(claim.requires_verification),
        new_text=str(value),
        note=note,
    )
    update = {
        "requires_verification": value,
        "non_checkable_reason": None if value else claim.non_checkable_reason,
        "reviewer_edits": [*claim.reviewer_edits, edit],
    }
    return [*claims[:i], claim.model_copy(update=update), *claims[i + 1 :]]


def split_claim(
    claims: list[Claim], claim_id: str, parts: list[str], reviewer_id: str
) -> list[Claim]:
    """Replace one claim with several. Children keep the parent's original wording and span."""
    i = _index(claims, claim_id)
    parent = claims[i]
    _require_active(parent)
    cleaned = [_clean(p, "split part") for p in parts]
    if len(cleaned) < 2:
        raise ClaimEditError("a split needs at least two parts")
    children = []
    for n, text in enumerate(cleaned):
        child_id = make_id(CLAIM, parent.claim_id, "split", n, text)
        edit = ReviewerEdit(
            edit_type=EditType.SPLIT_FROM,
            reviewer_id=reviewer_id,
            previous_text=parent.normalized_claim,
            new_text=text,
            related_claim_ids=[parent.claim_id],
        )
        children.append(
            parent.model_copy(
                update={
                    **_with_text(text),
                    "claim_id": child_id,
                    "requires_verification": True,
                    "non_checkable_reason": None,
                    "reviewer_edits": [*parent.reviewer_edits, edit],
                }
            )
        )
    retired = parent.model_copy(
        update={
            "is_active": False,
            "reviewer_edits": [
                *parent.reviewer_edits,
                ReviewerEdit(
                    edit_type=EditType.SPLIT_INTO,
                    reviewer_id=reviewer_id,
                    related_claim_ids=[c.claim_id for c in children],
                ),
            ],
        }
    )
    existing = {c.claim_id for c in claims}
    if any(c.claim_id in existing for c in children):
        raise ClaimEditError("this split already exists")
    return [*claims[:i], retired, *children, *claims[i + 1 :]]


def merge_claims(
    claims: list[Claim], claim_ids: list[str], merged_text: str, reviewer_id: str
) -> list[Claim]:
    """Combine several claims into one (e.g. a fact the extractor wrongly split)."""
    if len(set(claim_ids)) < 2:
        raise ClaimEditError("a merge needs at least two distinct claims")
    indices = sorted(_index(claims, cid) for cid in claim_ids)
    sources = [claims[i] for i in indices]
    for claim in sources:
        _require_active(claim)
    text = _clean(merged_text, "merged text")
    methods = {c.extraction_method for c in sources}
    spans = [c.span for c in sources if c.span]
    contiguous_span = None
    if len(spans) == len(sources):
        contiguous_span = TextSpan(start=min(s.start for s in spans), end=max(s.end for s in spans))
    merged = Claim(
        claim_id=make_id(CLAIM, "merge", sorted(claim_ids), text),
        original_text=" ".join(c.original_text for c in sources),
        normalized_claim=text,
        source_response_id=sources[0].source_response_id,
        span=contiguous_span,
        extraction_method=methods.pop() if len(methods) == 1 else ExtractionMethod.MANUAL,
        extractor_version=sources[0].extractor_version,
        requires_verification=True,
        features=detect_features(text),
        temporal_context=temporal_context(detect_features(text)),
        reviewer_edits=[
            ReviewerEdit(
                edit_type=EditType.MERGED_FROM,
                reviewer_id=reviewer_id,
                new_text=text,
                related_claim_ids=[c.claim_id for c in sources],
            )
        ],
    )
    if any(c.claim_id == merged.claim_id for c in claims):
        raise ClaimEditError("this merge already exists")
    merge_edit = ReviewerEdit(
        edit_type=EditType.MERGED_INTO, reviewer_id=reviewer_id, related_claim_ids=[merged.claim_id]
    )
    result = list(claims)
    for i in indices:
        result[i] = result[i].model_copy(
            update={"is_active": False, "reviewer_edits": [*result[i].reviewer_edits, merge_edit]}
        )
    result.insert(indices[-1] + 1, merged)
    return result


def add_manual_claim(
    claims: list[Claim], response: AIResponse, text: str, reviewer_id: str, note: str | None = None
) -> list[Claim]:
    """Add a claim the extractor missed.

    If the text appears verbatim in the response, its span is recorded;
    otherwise the claim is the reviewer's own wording and has no span.
    """
    cleaned = _clean(text)
    position = response.text.find(cleaned)
    span = TextSpan(start=position, end=position + len(cleaned)) if position >= 0 else None
    location_note = "found verbatim in response" if span else "reviewer wording; not verbatim"
    features = detect_features(cleaned)
    claim = Claim(
        claim_id=make_id(CLAIM, response.response_id, "manual", cleaned),
        original_text=cleaned,
        normalized_claim=cleaned,
        source_response_id=response.response_id,
        span=span,
        extraction_method=ExtractionMethod.MANUAL,
        requires_verification=True,
        features=features,
        temporal_context=temporal_context(features),
        extraction_notes=[f"added manually by {reviewer_id}; {location_note}"],
        reviewer_edits=[
            ReviewerEdit(
                edit_type=EditType.ADDED, reviewer_id=reviewer_id, new_text=cleaned, note=note
            )
        ],
    )
    if any(c.claim_id == claim.claim_id for c in claims):
        raise ClaimEditError("an identical manual claim already exists")
    return [*claims, claim]


def _set_active(
    claims: list[Claim], claim_id: str, active: bool, reviewer_id: str, note: str | None
) -> list[Claim]:
    i = _index(claims, claim_id)
    claim = claims[i]
    if claim.is_active == active:
        return list(claims)
    edit = ReviewerEdit(
        edit_type=EditType.RESTORED if active else EditType.REMOVED,
        reviewer_id=reviewer_id,
        note=note,
    )
    updated = claim.model_copy(
        update={"is_active": active, "reviewer_edits": [*claim.reviewer_edits, edit]}
    )
    return [*claims[:i], updated, *claims[i + 1 :]]


def remove_claim(
    claims: list[Claim], claim_id: str, reviewer_id: str, note: str | None = None
) -> list[Claim]:
    return _set_active(claims, claim_id, False, reviewer_id, note)


def restore_claim(
    claims: list[Claim], claim_id: str, reviewer_id: str, note: str | None = None
) -> list[Claim]:
    return _set_active(claims, claim_id, True, reviewer_id, note)
