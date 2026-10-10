"""Pydantic data models for claims, sources, evidence, verifications and sessions.

Design principles encoded as validation rules (not just documentation):

* A verdict of SUPPORTED or CONTRADICTED must cite evidence. The system never
  asserts truth on its own authority.
* Human judgments require a reviewer ID; automated outputs must not carry one,
  so the two can never be confused.
* Extraction confidence may only be stored together with a description of how
  it was calibrated. Uncalibrated scores are not recorded at all.
* An AI response's ID is a hash of its exact text, so any modification of the
  stored original is detected on load.
* A session is referentially consistent: evidence and verifications only point
  at claims and evidence that exist in the same session.
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import date, datetime, timezone
from enum import Enum
from typing import Annotated, Any

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from hallucination_lab import __version__
from hallucination_lab.ids import ID_PATTERN, response_id_for

SCHEMA_VERSION = "1.0.0"
SUPPORTED_SCHEMA_MAJOR = 1

Identifier = Annotated[str, StringConstraints(pattern=ID_PATTERN)]
NonBlankStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
ScalarValue = str | int | float | bool | None


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _require_tz(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware (use UTC)")
    return value


AwareDatetime = Annotated[datetime, AfterValidator(_require_tz)]

_URL_RE = re.compile(r"^https?://\S+$")


def _check_url(value: str | None) -> str | None:
    if value is not None and not _URL_RE.match(value):
        raise ValueError("URL must start with http:// or https:// and contain no spaces")
    return value


OptionalUrl = Annotated[str | None, AfterValidator(_check_url)]


class StrictModel(BaseModel):
    """Base model: unknown fields are rejected so typos never pass silently."""

    model_config = ConfigDict(extra="forbid", validate_default=True)


# --------------------------------------------------------------------------- enums


class ResponseProvenance(str, Enum):
    """Where an AI response text came from. Fixtures must never pose as real outputs."""

    SYNTHETIC_FIXTURE = "synthetic_fixture"  # hand-written illustrative text
    USER_SUBMITTED = "user_submitted"  # pasted by a user; origin not independently known
    AUTHENTIC_MODEL_OUTPUT = "authentic_model_output"  # captured from a named model


class ExtractionMethod(str, Enum):
    RULE_BASED = "rule_based"
    MANUAL = "manual"
    LLM_ASSISTED = "llm_assisted"  # reserved for the optional future LLM module


class NonCheckableReason(str, Enum):
    """Why the rule-based extractor thinks a segment is not a checkable fact."""

    QUESTION = "question"
    INSTRUCTION = "instruction"
    OPINION = "opinion"
    HYPOTHETICAL = "hypothetical"
    NO_ASSERTION = "no_assertion"  # fragments, headings, greetings


class EditType(str, Enum):
    EDITED = "edited"
    SPLIT_FROM = "split_from"
    SPLIT_INTO = "split_into"
    MERGED_FROM = "merged_from"
    MERGED_INTO = "merged_into"
    ADDED = "added"
    REMOVED = "removed"
    RESTORED = "restored"
    VERIFICATION_FLAG_CHANGED = "verification_flag_changed"


class RetrievalMethod(str, Enum):
    BM25 = "bm25"
    TFIDF = "tfidf"
    MANUAL = "manual"  # evidence attached by a reviewer, not by a ranking algorithm


class Verdict(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    NOT_CHECKABLE = "NOT_CHECKABLE"


class VerificationMethod(str, Enum):
    HUMAN = "human"
    AUTOMATED_RULE = "automated_rule"  # deterministic baseline (e.g. exact numeric match)
    MODEL_SUGGESTION = "model_suggestion"  # never treated as a verified judgment


# --------------------------------------------------------------------- AI response


class AIResponse(StrictModel):
    """The AI-generated text under investigation, stored exactly as submitted."""

    response_id: Identifier
    text: str = Field(min_length=1)
    provenance: ResponseProvenance
    model_name: str | None = None
    prompt: str | None = None
    collected_at: AwareDatetime | None = None
    notes: str | None = None

    @field_validator("text")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        # Intentionally *not* stripped: the original must be preserved byte for byte.
        if not value.strip():
            raise ValueError("response text must not be blank")
        return value

    @model_validator(mode="after")
    def _check_integrity(self) -> AIResponse:
        expected = response_id_for(self.text)
        if self.response_id != expected:
            raise ValueError(
                f"response_id {self.response_id} does not match the text hash ({expected}); "
                "the stored response text may have been modified"
            )
        if self.provenance is ResponseProvenance.AUTHENTIC_MODEL_OUTPUT and not self.model_name:
            raise ValueError("authentic model outputs must name the model that produced them")
        return self

    @classmethod
    def from_text(
        cls,
        text: str,
        provenance: ResponseProvenance = ResponseProvenance.USER_SUBMITTED,
        **kw: Any,
    ) -> AIResponse:
        return cls(response_id=response_id_for(text), text=text, provenance=provenance, **kw)


# --------------------------------------------------------------------------- claims


class TextSpan(StrictModel):
    """Character offsets into ``AIResponse.text`` (half-open: ``text[start:end]``)."""

    start: int = Field(ge=0)
    end: int = Field(gt=0)

    @model_validator(mode="after")
    def _ordered(self) -> TextSpan:
        if self.end <= self.start:
            raise ValueError("span end must be greater than start")
        return self


class ClaimFeatures(StrictModel):
    """Surface features detected by transparent regex rules (see extraction.features)."""

    numbers: list[str] = Field(default_factory=list)
    dates: list[str] = Field(default_factory=list)
    measurements: list[str] = Field(default_factory=list)
    named_entities: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    quotations: list[str] = Field(default_factory=list)
    temporal_markers: list[str] = Field(default_factory=list)
    starts_with_pronoun: bool = False


class ReviewerEdit(StrictModel):
    """One entry in a claim's audit trail."""

    edit_type: EditType
    reviewer_id: NonBlankStr
    timestamp: AwareDatetime = Field(default_factory=utc_now)
    previous_text: str | None = None
    new_text: str | None = None
    related_claim_ids: list[Identifier] = Field(default_factory=list)
    note: str | None = None


class Claim(StrictModel):
    claim_id: Identifier
    original_text: NonBlankStr
    normalized_claim: NonBlankStr
    source_response_id: Identifier
    span: TextSpan | None = None
    extraction_method: ExtractionMethod
    extractor_version: str | None = None
    extraction_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence_calibration: str | None = None
    requires_verification: bool
    non_checkable_reason: NonCheckableReason | None = None
    temporal_context: str | None = None
    features: ClaimFeatures = Field(default_factory=ClaimFeatures)
    # Which extraction rules fired and why, so the extractor's decisions are inspectable.
    extraction_notes: list[str] = Field(default_factory=list)
    reviewer_edits: list[ReviewerEdit] = Field(default_factory=list)
    is_active: bool = True

    @model_validator(mode="after")
    def _consistency(self) -> Claim:
        if self.extraction_confidence is not None and not self.confidence_calibration:
            raise ValueError(
                "extraction_confidence requires confidence_calibration describing how the "
                "value was calibrated; leave it empty when no calibrated value exists"
            )
        if self.requires_verification and self.non_checkable_reason is not None:
            raise ValueError(
                "a claim that requires verification cannot have a non-checkable reason"
            )
        return self

    @property
    def is_human_modified(self) -> bool:
        return bool(self.reviewer_edits)


# ----------------------------------------------------------------- sources/evidence


class Source(StrictModel):
    """A reference document in a collection."""

    source_id: Identifier
    title: NonBlankStr
    url: OptionalUrl = None
    document_reference: str | None = None
    publication_date: date | None = None
    document_version: str | None = None
    license: str | None = None
    content_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")] | None = None
    ingested_at: AwareDatetime | None = None

    @model_validator(mode="after")
    def _locatable(self) -> Source:
        if not self.url and not self.document_reference:
            raise ValueError("a source needs a url or a document_reference so it can be cited")
        return self


class PassageLocation(StrictModel):
    """Where a quoted passage sits inside its source document."""

    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    paragraph_index: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _ordered(self) -> PassageLocation:
        if self.char_end <= self.char_start:
            raise ValueError("char_end must be greater than char_start")
        return self


class Evidence(StrictModel):
    """A passage *retrieved* for a claim.

    Retrieval only means the passage looked relevant to a ranking algorithm.
    Evidence counts as verified only once a human verification cites it
    (see ``VerificationSession.human_cited_evidence_ids``).
    """

    evidence_id: Identifier
    claim_id: Identifier
    source_id: Identifier
    source_title: NonBlankStr
    source_url: OptionalUrl = None
    document_reference: str | None = None
    retrieved_at: AwareDatetime
    quoted_passage: NonBlankStr
    passage_location: PassageLocation | None = None
    document_version: str | None = None
    publication_date: date | None = None
    retrieval_method: RetrievalMethod
    retrieval_rank: int | None = Field(default=None, ge=1)
    # Raw ranking score (e.g. BM25). Not a probability and not a truth estimate.
    retrieval_score: float | None = None
    retrieval_parameters: dict[str, ScalarValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _locatable(self) -> Evidence:
        if not self.source_url and not self.document_reference:
            raise ValueError("evidence needs a source_url or document_reference for attribution")
        return self


# --------------------------------------------------------------------- verification


class Provenance(StrictModel):
    """How a verification record was produced, for reproducibility."""

    tool: str = "hallucination_lab"
    tool_version: str = __version__
    method_detail: NonBlankStr
    model_name: str | None = None
    parameters: dict[str, ScalarValue] = Field(default_factory=dict)


class Verification(StrictModel):
    verification_id: Identifier
    claim_id: Identifier
    verdict: Verdict
    reasoning: NonBlankStr
    supporting_evidence_ids: list[Identifier] = Field(default_factory=list)
    contradicting_evidence_ids: list[Identifier] = Field(default_factory=list)
    verification_method: VerificationMethod
    reviewer_id: str | None = None
    verification_timestamp: AwareDatetime = Field(default_factory=utc_now)
    provenance: Provenance

    @model_validator(mode="after")
    def _rules(self) -> Verification:
        human = self.verification_method is VerificationMethod.HUMAN
        if human and not (self.reviewer_id and self.reviewer_id.strip()):
            raise ValueError("human verifications must record a reviewer_id")
        if not human and self.reviewer_id is not None:
            raise ValueError("automated or model verifications must not carry a reviewer_id")
        if self.verdict is Verdict.SUPPORTED and not self.supporting_evidence_ids:
            raise ValueError("a SUPPORTED verdict must cite at least one supporting evidence ID")
        if self.verdict is Verdict.CONTRADICTED and not self.contradicting_evidence_ids:
            raise ValueError(
                "a CONTRADICTED verdict must cite at least one contradicting evidence ID"
            )
        overlap = set(self.supporting_evidence_ids) & set(self.contradicting_evidence_ids)
        if overlap:
            raise ValueError(f"evidence cannot both support and contradict: {sorted(overlap)}")
        if self.verification_method is VerificationMethod.MODEL_SUGGESTION and not (
            self.provenance.model_name
        ):
            raise ValueError("model suggestions must record provenance.model_name")
        return self

    @property
    def is_human_judgment(self) -> bool:
        return self.verification_method is VerificationMethod.HUMAN


# -------------------------------------------------------------------------- session


def _check_schema_version(value: str) -> str:
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", value)
    if not match:
        raise ValueError(f"schema_version must look like MAJOR.MINOR.PATCH, got {value!r}")
    if int(match.group(1)) != SUPPORTED_SCHEMA_MAJOR:
        raise ValueError(
            f"unsupported schema major version {match.group(1)} "
            f"(this build reads {SUPPORTED_SCHEMA_MAJOR}.x)"
        )
    return value


class VerificationSession(StrictModel):
    """Everything about one investigated response: claims, evidence, verdicts."""

    schema_version: Annotated[str, AfterValidator(_check_schema_version)] = SCHEMA_VERSION
    session_id: Identifier
    title: NonBlankStr
    created_at: AwareDatetime = Field(default_factory=utc_now)
    updated_at: AwareDatetime = Field(default_factory=utc_now)
    response: AIResponse
    claims: list[Claim] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    verifications: list[Verification] = Field(default_factory=list)
    notes: str | None = None

    @model_validator(mode="after")
    def _referential_integrity(self) -> VerificationSession:
        _require_unique("claim_id", [c.claim_id for c in self.claims])
        _require_unique("evidence_id", [e.evidence_id for e in self.evidence])
        _require_unique("verification_id", [v.verification_id for v in self.verifications])

        for claim in self.claims:
            if claim.source_response_id != self.response.response_id:
                raise ValueError(f"claim {claim.claim_id} belongs to a different response")
            if claim.span and claim.span.end > len(self.response.text):
                raise ValueError(f"claim {claim.claim_id} span exceeds the response length")

        claim_ids = {c.claim_id for c in self.claims}
        evidence_claim = {e.evidence_id: e.claim_id for e in self.evidence}
        for ev in self.evidence:
            if ev.claim_id not in claim_ids:
                raise ValueError(
                    f"evidence {ev.evidence_id} references unknown claim {ev.claim_id}"
                )
        for ver in self.verifications:
            if ver.claim_id not in claim_ids:
                raise ValueError(
                    f"verification {ver.verification_id} references unknown claim {ver.claim_id}"
                )
            for eid in [*ver.supporting_evidence_ids, *ver.contradicting_evidence_ids]:
                if eid not in evidence_claim:
                    raise ValueError(
                        f"verification {ver.verification_id} cites unknown evidence {eid}"
                    )
                if evidence_claim[eid] != ver.claim_id:
                    raise ValueError(
                        f"verification {ver.verification_id} cites evidence {eid} "
                        "that was retrieved for a different claim"
                    )
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot be earlier than created_at")
        return self

    # --- read helpers -------------------------------------------------------------

    @property
    def active_claims(self) -> list[Claim]:
        return [c for c in self.claims if c.is_active]

    def get_claim(self, claim_id: str) -> Claim:
        for claim in self.claims:
            if claim.claim_id == claim_id:
                return claim
        raise KeyError(claim_id)

    def evidence_for(self, claim_id: str) -> list[Evidence]:
        return [e for e in self.evidence if e.claim_id == claim_id]

    def verifications_for(self, claim_id: str) -> list[Verification]:
        return sorted(
            (v for v in self.verifications if v.claim_id == claim_id),
            key=lambda v: v.verification_timestamp,
        )

    def human_verdict(self, claim_id: str) -> Verification | None:
        """Latest human judgment for a claim; automated suggestions are ignored."""
        human = [v for v in self.verifications_for(claim_id) if v.is_human_judgment]
        return human[-1] if human else None

    def pending_review(self) -> list[Claim]:
        """Active claims flagged for verification that have no human verdict yet."""
        return [
            c
            for c in self.active_claims
            if c.requires_verification and self.human_verdict(c.claim_id) is None
        ]

    def human_cited_evidence_ids(self) -> set[str]:
        """Evidence a human reviewer actually relied on (vs. merely retrieved)."""
        cited: set[str] = set()
        for v in self.verifications:
            if v.is_human_judgment:
                cited.update(v.supporting_evidence_ids)
                cited.update(v.contradicting_evidence_ids)
        return cited

    def touch(self) -> VerificationSession:
        """Return a copy with ``updated_at`` set to now."""
        return self.model_copy(update={"updated_at": utc_now()})


def _require_unique(name: str, values: list[str]) -> None:
    dupes = [value for value, count in Counter(values).items() if count > 1]
    if dupes:
        raise ValueError(f"duplicate {name}: {sorted(dupes)}")
