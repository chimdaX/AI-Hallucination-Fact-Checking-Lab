"""Regex-based surface feature detection for candidate claims.

These features help a reviewer triage (e.g. "this claim contains a statistic
and a citation") and will drive the deterministic numeric checks in Phase 4.
They are heuristics: named-entity detection is just capitalised-word runs, and
spelled-out numbers ("three") are not recognised.
"""

from __future__ import annotations

import re

from hallucination_lab.models import ClaimFeatures

_MONTHS = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|"
    r"Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)
_UNITS = (
    r"(?:%|percent|per cent|km/h|mph|km|kilomet(?:re|er)s?|metres?|meters?|m|cm|mm|miles?|"
    r"feet|foot|ft|inches|kg|kilograms?|g|grams?|tonnes?|tons?|pounds?|lbs?|"
    r"°\s?[CF]|degrees(?:\s+(?:Celsius|Fahrenheit))?|K|kelvin|"
    r"seconds?|minutes?|hours?|days?|weeks?|months?|years?|light[- ]years?|AU|"
    r"people|inhabitants|residents|species|atoms?|molecules?|"
    r"million|billion|trillion|thousand|hundred)"
)
_NUMBER = r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"

_DATE_RES = [
    re.compile(rf"\b{_MONTHS}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}\b"),
    re.compile(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTHS}\s+\d{{4}}\b"),
    re.compile(rf"\b{_MONTHS}\s+\d{{4}}\b"),
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    re.compile(r"\b\d{1,2}(?:st|nd|rd|th)\s+century\b", re.IGNORECASE),
    # Bare years 1000-2099 not followed by a unit ("1500 metres" is not a year).
    re.compile(rf"(?<![\w.,])(?:1\d{{3}}|20\d{{2}})s?\b(?!\s*{_UNITS}\b)(?![.,]\d)"),
]
_NUMBER_RE = re.compile(rf"{_NUMBER}(?:\s*%)?")
_MEASUREMENT_RE = re.compile(rf"{_NUMBER}\s*{_UNITS}(?![\w])")
_CITATION_RES = [
    re.compile(
        r"\b[A-Z][A-Za-z'\-]+(?:\s+(?:and|&)\s+[A-Z][A-Za-z'\-]+)?\s+et\s+al\.?(?:\s*\(\d{4}\))?"
    ),
    re.compile(r"\b[A-Z][A-Za-z'\-]+\s+\(\d{4}[a-z]?\)"),
    re.compile(r"\b(?:doi:\s*)?10\.\d{4,9}/\S+[\w)]", re.IGNORECASE),
    re.compile(r"\barXiv:\s*\d{4}\.\d{4,5}\b", re.IGNORECASE),
    re.compile(
        r"\b(?:published\s+in|appeared\s+in)\s+(?:the\s+)?[A-Z][\w&]*(?:\s+(?:of|on|for|and|[A-Z][\w&]*))*"
    ),
]
_QUOTE_RE = re.compile(r"[\"\u201c]([^\"\u201d]{3,})[\"\u201d]")
_TEMPORAL_RE = re.compile(
    r"\b(?:currently|current|now|nowadays|today|at present|presently|as of|latest|"
    r"most recent|recently|this year|so far|to date|still|record)\b",
    re.IGNORECASE,
)
_ENTITY_RE = re.compile(
    r"\b[A-Z][\w'\u2019\-]*(?:\s+(?:of|the|de|du|von|van|da|del)\s+[A-Z][\w'\u2019\-]*"
    r"|\s+[A-Z][\w'\u2019\-]*)*"
)
_SENTENCE_INITIAL_STOPWORDS = frozenset(
    {
        "The", "A", "An", "It", "Its", "This", "That", "These", "Those", "In", "On", "At",
        "He", "She", "They", "We", "I", "You", "Their", "His", "Her", "There", "However",
        "Also", "Additionally", "Furthermore", "Moreover", "For", "As", "By", "According",
        "After", "Before", "During", "Since", "While", "When", "Although", "Some", "Many",
        "Most", "Both", "Each", "If", "Note", "Today", "Currently", "Yes", "No", "What",
        "Why", "How", "Where", "Who", "Which", "Would", "Could", "Can", "Do", "Does", "Is",
        "Are", "Was", "Were", "Did",
    }
)  # fmt: skip
_PRONOUNS = frozenset(
    {"it", "its", "they", "their", "them", "he", "she", "his", "her", "this", "these", "those"}
)


def _unique(items: list[str]) -> list[str]:
    seen: dict[str, None] = {}
    for item in items:
        item = item.strip().rstrip(".,;:")
        if item:
            seen.setdefault(item, None)
    return list(seen)


def _named_entities(text: str) -> list[str]:
    found = []
    for match in _ENTITY_RE.finditer(text):
        words = match.group().split()
        # Drop function words that are only capitalised because they start a sentence.
        while words and words[0] in _SENTENCE_INITIAL_STOPWORDS:
            words.pop(0)
        # Single capital letters are usually units or initials ("°C"), not entities.
        if words and not (len(words) == 1 and len(words[0]) == 1):
            found.append(" ".join(words))
    return found


def detect_features(text: str) -> ClaimFeatures:
    dates: list[str] = []
    for regex in _DATE_RES:
        for match in regex.finditer(text):
            if not any(match.group() in d for d in dates):
                dates.append(match.group())
    first_word = text.split(maxsplit=1)[0].lower().strip(",") if text.strip() else ""
    return ClaimFeatures(
        numbers=_unique([m.group() for m in _NUMBER_RE.finditer(text)]),
        dates=_unique(dates),
        measurements=_unique([m.group() for m in _MEASUREMENT_RE.finditer(text)]),
        named_entities=_unique(_named_entities(text)),
        citations=_unique([m.group() for r in _CITATION_RES for m in r.finditer(text)]),
        quotations=_unique([m.group(1) for m in _QUOTE_RE.finditer(text)]),
        temporal_markers=_unique([m.group().lower() for m in _TEMPORAL_RE.finditer(text)]),
        starts_with_pronoun=first_word in _PRONOUNS,
    )


def temporal_context(features: ClaimFeatures) -> str | None:
    """Human-readable note about time-dependence, or None."""
    parts = []
    if features.temporal_markers:
        parts.append("time-sensitive wording: " + ", ".join(features.temporal_markers))
    if features.dates:
        parts.append("refers to: " + ", ".join(features.dates))
    return "; ".join(parts) or None
