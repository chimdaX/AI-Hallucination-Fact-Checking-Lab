"""Lightweight quality checks for written verification rationales.

Two levels:

* **Errors** reject text that cannot be a rationale: empty text, placeholders
  ("n/a", "tbd"), a bare score ("4/5"), no letters, or a single word that just
  restates a verdict ("correct"). The schema enforces these, so they cannot be
  bypassed.
* **Warnings** flag text that is probably too generic ("Looks fine.", "The AI
  is wrong.") or that does not refer to the claim or evidence. The reviewer can
  keep it after seeing the warning.

There is deliberately no minimum length. "Source dates it 1887-1889, not
1899." is short and specific. These heuristics catch obvious problems only.
They cannot judge whether a human's reasoning is actually sound.

This module has no project imports, so the schema can use it without
circular imports.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

PLACEHOLDERS = frozenset(
    {
        "n/a", "na", "n.a.", "n.a", "tbd", "tba", "todo", "to do", "none", "nothing", "null",
        "nil", "-", "--", "---", "?", "??", "???", "..", "...", "x", "xx", "xxx", "idk",
        "dunno", "test", "testing", "asdf", "qwerty", "placeholder", "lorem ipsum", "same",
        "see above", "same as above", "blah", "no comment",
    }
)  # fmt: skip
# A single word that only restates a verdict is not an explanation.
VERDICT_WORDS = frozenset(
    [
        "ok",
        "okay",
        "yes",
        "no",
        "fine",
        "correct",
        "incorrect",
        "wrong",
        "right",
        "true",
        "false",
        "agree",
        "disagree",
        "supported",
        "contradicted",
        "insufficient",
        "unverifiable",
        "good",
        "bad",
    ]
)
GENERIC_PHRASES = (
    "the claim is correct", "the claim is true", "the claim is right", "claim is correct",
    "the claim is false", "the claim is wrong", "the claim is incorrect", "looks fine",
    "looks good", "looks right", "looks ok", "seems fine", "seems right", "seems correct",
    "the ai is wrong", "the ai is right", "the ai is correct", "ai is wrong", "good evidence",
    "bad evidence", "strong evidence", "obviously true", "obviously false", "everyone knows",
    "common knowledge", "as expected", "self explanatory", "self-explanatory", "verified",
    "checked", "confirmed", "it is true", "it is false", "this is correct", "this is wrong",
)  # fmt: skip
_SCORE_RE = re.compile(
    r"^(?:score|rating|confidence|grade)?\s*[:=]?\s*[-+]?\d+(?:[.,]\d+)?\s*"
    r"(?:/\s*\d+|%|percent|out of \d+|stars?|points?)?$",
    re.IGNORECASE,
)
_MISSING_WORDS = re.compile(
    r"\b(?:no|not|none|missing|lack\w*|insufficient|unclear|ambiguous|inconclusive|"
    r"conflict\w*|only|unable|cannot|can't|couldn't|doesn't|does not|did not|unknown|"
    r"irrelevant|outdated|unverified|weak|partial\w*|neither|nor|without)\b",
    re.IGNORECASE,
)
_NOT_CHECKABLE_WORDS = re.compile(
    r"\b(?:opinion|subjective|preference|taste|vague|ambiguous|undefined|define\w*|"
    r"meaning|hypothetical|prediction|future|normative|value judg\w*|rhetorical|"
    r"question|instruction|unfalsifiable|not a factual|no objective|no factual|"
    r"evaluative|depends on)\b",
    re.IGNORECASE,
)
_RESEARCH_DIFFICULTY = re.compile(
    r"\b(?:hard to (?:find|research|verify)|difficult to (?:find|research|verify)|"
    r"could(?:n't| not) find|no evidence|not found|no sources?|nothing (?:was )?found)\b",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"[^\W\d_]{3,}|\d+(?:[.,]\d+)?", re.UNICODE)
_STOP = frozenset(
    [
        "the",
        "and",
        "for",
        "that",
        "this",
        "with",
        "from",
        "are",
        "was",
        "were",
        "has",
        "have",
        "had",
        "not",
        "but",
        "its",
        "into",
        "over",
        "claim",
        "claims",
        "evidence",
        "passage",
        "source",
        "sources",
        "says",
        "said",
        "states",
        "stated",
        "because",
        "which",
        "than",
        "then",
        "also",
        "they",
        "them",
        "there",
        "their",
        "what",
        "when",
        "where",
        "who",
        "why",
        "how",
        "does",
        "did",
        "done",
    ]
)


def _normalise(text: str) -> str:
    return re.sub(r"[^\w\s/%-]", "", " ".join(text.lower().split())).strip()


@dataclass
class RationaleCheck:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def rationale_errors(text: str | None) -> list[str]:
    """Problems severe enough that the text cannot count as an explanation."""
    if text is None or not text.strip():
        return ["the explanation is empty"]
    norm = _normalise(text)
    if norm in PLACEHOLDERS or text.strip().lower() in PLACEHOLDERS:
        return [f"{text.strip()!r} is a placeholder, not an explanation"]
    if _SCORE_RE.fullmatch(text.strip()):
        return ["a score or number alone is not an explanation"]
    if not re.search(r"[^\W\d_]", text):
        return ["the explanation contains no words"]
    letters = re.sub(r"[\W\d_]", "", text.lower())
    if len(letters) > 2 and len(set(letters)) <= 2:
        return ["the explanation looks like filler (repeated characters)"]
    if norm in VERDICT_WORDS:
        return [f"{text.strip()!r} restates a verdict without explaining it"]
    return []


def _content_words(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text) if w.lower() not in _STOP}


def check_rationale(
    text: str | None,
    verdict: str | None = None,
    claim_text: str = "",
    cited_passages: list[str] | None = None,
) -> RationaleCheck:
    """Return errors (must fix) and warnings (reviewer may override)."""
    result = RationaleCheck(errors=rationale_errors(text))
    if result.errors:
        return result
    assert text is not None
    norm = _normalise(text)
    words = norm.split()
    if any(norm == p or (len(words) <= 8 and p in norm) for p in GENERIC_PHRASES):
        result.warnings.append(
            "this reads as a generic statement; say what the evidence shows and how it "
            "relates to the claim (e.g. which date, number or name agrees or conflicts)"
        )
    if verdict in {"SUPPORTED", "CONTRADICTED"}:
        reference = _content_words(claim_text) | _content_words(" ".join(cited_passages or []))
        if reference and not (_content_words(text) & reference):
            result.warnings.append(
                "the explanation does not mention anything from the claim or the cited "
                "passages; consider naming the specific fact that agrees or conflicts"
            )
    elif verdict == "INSUFFICIENT_EVIDENCE" and not _MISSING_WORDS.search(text):
        result.warnings.append(
            "for INSUFFICIENT_EVIDENCE, explain what is missing, weak, ambiguous or "
            "conflicting in the available evidence"
        )
    elif verdict == "NOT_CHECKABLE":
        if _RESEARCH_DIFFICULTY.search(text):
            result.warnings.append(
                "NOT_CHECKABLE is for opinions or statements without an objective factual "
                "proposition, not for claims that are hard to research; consider "
                "INSUFFICIENT_EVIDENCE"
            )
        elif not _NOT_CHECKABLE_WORDS.search(text):
            result.warnings.append(
                "for NOT_CHECKABLE, explain why the statement is not objectively checkable "
                "(e.g. opinion, subjective, too vague, a prediction)"
            )
    return result
