"""Rule-based sentence and clause segmentation with exact character offsets.

Every segment carries ``start``/``end`` offsets into the original text so a
claim can always be traced back to the exact wording the AI produced.

Known limitations (deliberately simple and explainable, not perfect):

* Abbreviations are handled with a fixed list. An abbreviation that really ends
  a sentence ("... in the U.S. It was ...") is not split.
* Line handling: blank lines, bullets, headings and lines ending in ``.!?:``
  start a new block; any other single newline is treated as a soft wrap.
* Clause splitting only fires on ``;`` and on ``, and|but|while|whereas`` when
  both sides look like full clauses (3+ words and a verb from a small list or a
  lowercase ``-ed`` word). Coordinated noun phrases are left intact, and many
  genuine multi-fact sentences are not split. Reviewers can split manually.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

ABBREVIATIONS = frozenset(
    {
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "ft", "vs", "cf", "approx",
        "ca", "c", "inc", "ltd", "co", "corp", "dept", "est", "al", "e.g", "i.e", "u.s",
        "u.k", "u.n", "e.u", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept",
        "oct", "nov", "dec", "gen", "gov", "sen", "rep", "lt", "col", "capt", "sgt", "ph.d",
    }
)  # fmt: skip
# Abbreviations that commonly end a sentence: split if the next word is capitalised.
SENTENCE_FINAL_ABBREVIATIONS = frozenset({"etc", "a.m", "p.m"})
# Abbreviations that are only abbreviations when a number follows ("No. 5", "p. 12").
NUMERIC_ABBREVIATIONS = frozenset({"no", "nos", "vol", "pp", "p", "fig", "figs"})

_BULLET_RE = re.compile(r"[ \t]*(?:[-*\u2022+]|\d{1,3}[.)]|#{1,6})[ \t]+")
_HEADING_RE = re.compile(r"[ \t]*#{1,6}[ \t]+")
_BOUNDARY_RE = re.compile(r"[.!?]+[\"'\u201d\u2019)\]]*(?=\s|$)")
_CLAUSE_SEP_RE = re.compile(r";\s+|,\s+(?:and|but|while|whereas)\s+")
_VERB_RE = re.compile(
    r"\b(?:is|are|was|were|has|have|had|will|did|does|do|can|could|may|might|must|"
    r"became|become|becomes|remains|remained|contains|includes|lies|sits|stands|stood|"
    r"won|wrote|made|took|began|led|holds|held|lasts|takes|orbits|boils|melts|freezes|"
    r"covers|flows|runs|grew|rose|fell|gave|built|died|lived|means|equals)\b"
    r"|\b(?!hundred\b|red\b|bed\b|need\b|speed\b|seed\b|feed\b)[a-z]{2,}ed\b"
)


@dataclass(frozen=True)
class Segment:
    text: str
    start: int
    end: int


def _trimmed(text: str, start: int, end: int) -> Segment | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start == end:
        return None
    return Segment(text[start:end], start, end)


def _blocks(text: str) -> list[tuple[int, int]]:
    """Group lines into blocks; returns (start, end) offsets, bullet markers excluded."""
    blocks: list[tuple[int, int]] = []
    current: list[int] | None = None  # [start, end]
    line_start = 0
    for line in text.split("\n"):
        line_end = line_start + len(line)
        next_start = line_end + 1
        if not line.strip():
            if current:
                blocks.append((current[0], current[1]))
            current = None
            line_start = next_start
            continue
        bullet = _BULLET_RE.match(line)
        body_start = line_start + (bullet.end() if bullet else 0)
        if current is None or bullet:
            if current:
                blocks.append((current[0], current[1]))
            current = [body_start, line_end]
        else:
            current[1] = line_end
        stripped = line.rstrip()
        if _HEADING_RE.match(line) or stripped.endswith((".", "!", "?", ":")):
            blocks.append((current[0], current[1]))
            current = None
        line_start = next_start
    if current:
        blocks.append((current[0], current[1]))
    return blocks


def _is_sentence_boundary(text: str, match: re.Match[str], block_end: int) -> bool:
    if any(ch in match.group() for ch in "!?"):
        return True
    token_match = re.search(r"(\S+)$", text[: match.start()])
    token = token_match.group(1).lstrip("(\"'\u201c\u2018") if token_match else ""
    following = text[match.end() : block_end].lstrip()
    nxt = following[:1]
    lowered = token.lower()
    if lowered in SENTENCE_FINAL_ABBREVIATIONS:
        return nxt.isupper()
    if lowered in NUMERIC_ABBREVIATIONS:
        return not nxt.isdigit()
    if lowered in ABBREVIATIONS:
        return False
    if re.fullmatch(r"[A-Z]", token):  # an initial, e.g. "J. R. R. Tolkien"
        return False
    return not (nxt and nxt.islower())


def split_sentences(text: str) -> list[Segment]:
    """Split text into sentence segments with offsets into ``text``."""
    segments: list[Segment] = []
    for block_start, block_end in _blocks(text):
        pos = block_start
        for match in _BOUNDARY_RE.finditer(text, block_start, block_end):
            if not _is_sentence_boundary(text, match, block_end):
                continue
            seg = _trimmed(text, pos, match.end())
            if seg:
                segments.append(seg)
            pos = match.end()
        seg = _trimmed(text, pos, block_end)
        if seg:
            segments.append(seg)
    return segments


def _looks_like_clause(fragment: str) -> bool:
    return len(fragment.split()) >= 3 and bool(_VERB_RE.search(fragment))


def split_compound(text: str, segment: Segment) -> list[Segment]:
    """Split one sentence into independent clauses where the rules are confident."""
    pieces: list[tuple[int, int]] = []
    pos = segment.start
    for match in _CLAUSE_SEP_RE.finditer(text, segment.start, segment.end):
        pieces.append((pos, match.start()))
        pos = match.end()
    pieces.append((pos, segment.end))
    if len(pieces) == 1:
        return [segment]

    merged: list[tuple[int, int]] = [pieces[0]]
    for start, end in pieces[1:]:
        prev_start, prev_end = merged[-1]
        if _looks_like_clause(text[prev_start:prev_end]) and _looks_like_clause(text[start:end]):
            merged.append((start, end))
        else:
            # Re-join, keeping the separator, so the span is still verbatim text.
            merged[-1] = (prev_start, end)
    result = [_trimmed(text, s, e) for s, e in merged]
    return [seg for seg in result if seg]
