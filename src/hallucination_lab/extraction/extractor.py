"""Rule-based candidate claim extraction.

Pipeline: sentence segmentation -> optional compound-clause splitting ->
normalisation -> non-checkable rules -> surface features.

The output is a list of *candidate* claims for human review. Sentence
segmentation is not the same as factual claim extraction: one sentence can
hold several facts, and some facts only make sense with surrounding context.
Every claim therefore keeps its exact original wording and character span, and
reviewers can split, merge, edit, add or remove claims (see ``editing``).

No confidence value is produced: rule outputs are not calibrated probabilities.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from hallucination_lab.errors import ExtractionError
from hallucination_lab.extraction.features import detect_features, temporal_context
from hallucination_lab.extraction.rules import classify, normalize_claim_text
from hallucination_lab.extraction.segmenter import split_compound, split_sentences
from hallucination_lab.ids import CLAIM, make_id
from hallucination_lab.models import (
    AIResponse,
    Claim,
    ExtractionMethod,
    ResponseProvenance,
    TextSpan,
)

logger = logging.getLogger(__name__)

EXTRACTOR_VERSION = "rule_based-0.1.0"


@dataclass(frozen=True)
class ExtractionConfig:
    split_compound_sentences: bool = True
    min_words: int = 3


class RuleBasedClaimExtractor:
    def __init__(self, config: ExtractionConfig | None = None) -> None:
        self.config = config or ExtractionConfig()

    def extract(self, response: AIResponse) -> list[Claim]:
        text = response.text
        claims: list[Claim] = []
        for sentence in split_sentences(text):
            parts = (
                split_compound(text, sentence)
                if self.config.split_compound_sentences
                else [sentence]
            )
            for part in parts:
                notes = []
                if len(parts) > 1:
                    notes.append(f"compound sentence split into {len(parts)} clauses by rule")
                claims.append(self._build_claim(response, part.text, part.start, part.end, notes))
        logger.info("extracted %d candidate claims from %s", len(claims), response.response_id)
        return claims

    def _build_claim(
        self, response: AIResponse, original: str, start: int, end: int, notes: list[str]
    ) -> Claim:
        normalized = normalize_claim_text(original)
        rule = classify(normalized, self.config.min_words)
        features = detect_features(original)
        notes.append(rule.describe())
        if features.starts_with_pronoun:
            notes.append("starts with a pronoun; may need surrounding context to verify")
        return Claim(
            claim_id=make_id(CLAIM, response.response_id, start, end, EXTRACTOR_VERSION),
            original_text=original,
            normalized_claim=normalized,
            source_response_id=response.response_id,
            span=TextSpan(start=start, end=end),
            extraction_method=ExtractionMethod.RULE_BASED,
            extractor_version=EXTRACTOR_VERSION,
            requires_verification=rule.is_checkable,
            non_checkable_reason=rule.reason,
            temporal_context=temporal_context(features),
            features=features,
            extraction_notes=notes,
        )


def extract_claims(
    text: str, config: ExtractionConfig | None = None
) -> tuple[AIResponse, list[Claim]]:
    """Convenience wrapper: wrap raw text as a user-submitted response and extract."""
    if not text or not text.strip():
        raise ExtractionError("cannot extract claims from empty text")
    response = AIResponse.from_text(text, ResponseProvenance.USER_SUBMITTED)
    return response, RuleBasedClaimExtractor(config).extract(response)
