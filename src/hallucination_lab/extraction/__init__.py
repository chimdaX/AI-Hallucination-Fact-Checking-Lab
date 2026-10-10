"""Candidate claim extraction (rule-based) and human claim editing."""

from hallucination_lab.extraction.extractor import (
    EXTRACTOR_VERSION,
    ExtractionConfig,
    RuleBasedClaimExtractor,
    extract_claims,
)

__all__ = ["EXTRACTOR_VERSION", "ExtractionConfig", "RuleBasedClaimExtractor", "extract_claims"]
