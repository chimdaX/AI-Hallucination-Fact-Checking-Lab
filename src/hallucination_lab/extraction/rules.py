"""Transparent rules that flag segments which are not checkable factual assertions.

Each rule returns *which* pattern fired so a reviewer can see why a segment was
flagged. Flags are suggestions: a reviewer can override ``requires_verification``.

Known limitations: a sentence like "I think Paris is the capital of France"
contains a checkable fact but is flagged as opinion; conditional scientific
statements ("If water is heated to 100 °C at sea level, it boils") are flagged
as hypothetical. Both are deliberate, conservative choices that the reviewer
can override.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from hallucination_lab.models import NonCheckableReason

_DISCOURSE_MARKERS = re.compile(
    r"^(?:(?:additionally|also|in addition|furthermore|moreover|in fact|interestingly|notably|"
    r"however|of course|actually|importantly|indeed|for example|for instance|similarly|"
    r"finally|first|second|third|lastly|overall|in summary|in short)\s*,\s*"
    r"|note that\s+|it is worth noting that\s+|it'?s worth noting that\s+)+",
    re.IGNORECASE,
)

_NO_ASSERTION = re.compile(
    r"^(?:sure|certainly|absolutely|of course|great question|good question|happy to help|"
    r"here (?:is|are)|hope this helps|i hope this helps|let me know|thanks|thank you)\b",
    re.IGNORECASE,
)
_IMPERATIVE_VERBS = frozenset(
    {
        "make", "try", "consider", "remember", "ensure", "check", "visit", "click", "ask",
        "avoid", "keep", "follow", "see", "read", "please", "let's", "lets", "use", "contact",
        "consult", "be", "do", "don't", "never", "always", "go", "take", "look", "verify",
    }
)  # fmt: skip
_INSTRUCTION_RE = re.compile(
    r"^you\s+(?:should|must|need to|have to|can|could|may want to)\b", re.IGNORECASE
)
_HYPOTHETICAL_RE = re.compile(
    r"^(?:if|suppose|supposing|imagine|assuming|assume|hypothetically|what if|in theory|"
    r"had|were)\b|\bwere to\b|\bhypothetically\b",
    re.IGNORECASE,
)
_OPINION_RES = [
    re.compile(r"\bI\s+(?:think|believe|feel|guess|suspect|reckon|personally)\b"),
    re.compile(r"\bin my (?:opinion|view|experience)\b", re.I),
    re.compile(r"\b(?:personally|arguably|subjectively)\b", re.I),
    re.compile(r"\bit seems\b", re.I),
    re.compile(r"\b(?:should|ought to)\b", re.I),
    re.compile(r"\b(?:best|worst)\b(?![- ]known|-selling)", re.I),
    re.compile(
        r"\b(?:beautiful|amazing|wonderful|fascinating|terrible|awful|boring|exciting|"
        r"incredible|delicious|overrated|underrated|stunning|breathtaking)\b",
        re.I,
    ),
]


@dataclass(frozen=True)
class RuleResult:
    reason: NonCheckableReason | None
    rule: str | None = None
    matched: str | None = None

    @property
    def is_checkable(self) -> bool:
        return self.reason is None

    def describe(self) -> str:
        if self.reason is None:
            return "no non-checkable rule matched"
        detail = f" (matched {self.matched!r})" if self.matched else ""
        return f"flagged as {self.reason.value} by rule {self.rule}{detail}"


def normalize_claim_text(text: str) -> str:
    """Collapse whitespace and drop leading discourse markers. Meaning is unchanged."""
    collapsed = " ".join(text.split())
    stripped = _DISCOURSE_MARKERS.sub("", collapsed).strip()
    if not stripped:
        return collapsed
    return stripped[0].upper() + stripped[1:]


def classify(text: str, min_words: int = 3) -> RuleResult:
    """Apply rules in a fixed order; the first match wins."""
    stripped = text.strip()
    tail = stripped.rstrip("\"'\u201d\u2019)] ")
    if tail.endswith("?"):
        return RuleResult(NonCheckableReason.QUESTION, "ends_with_question_mark", "?")

    words = re.findall(r"[A-Za-z0-9][\w'\u2019\-]*", stripped)
    if len(words) < min_words:
        return RuleResult(NonCheckableReason.NO_ASSERTION, f"fewer_than_{min_words}_words")
    if tail.endswith(":"):
        return RuleResult(NonCheckableReason.NO_ASSERTION, "lead_in_ending_with_colon", ":")
    if m := _NO_ASSERTION.match(stripped):
        return RuleResult(NonCheckableReason.NO_ASSERTION, "conversational_filler", m.group())

    first, second = words[0].lower(), (words[1].lower() if len(words) > 1 else "")
    if first in _IMPERATIVE_VERBS and second not in {"of", "is", "was", "are", "were"}:
        return RuleResult(NonCheckableReason.INSTRUCTION, "starts_with_imperative_verb", words[0])
    if m := _INSTRUCTION_RE.match(stripped):
        return RuleResult(NonCheckableReason.INSTRUCTION, "second_person_directive", m.group())

    if m := _HYPOTHETICAL_RE.search(stripped):
        return RuleResult(NonCheckableReason.HYPOTHETICAL, "conditional_or_hypothetical", m.group())

    for regex in _OPINION_RES:
        if m := regex.search(stripped):
            return RuleResult(NonCheckableReason.OPINION, "opinion_or_evaluative_marker", m.group())

    return RuleResult(None)
