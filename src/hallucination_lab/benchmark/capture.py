"""Capture verbatim evidence passages from source text, reproducibly.

* ``html_to_text`` (version ``html-text-v1``) turns an HTML page into plain text:
  scripts and styles are dropped, block elements become line breaks, and
  whitespace within a line is collapsed. Nothing else is changed.
* ``extract_sentence`` returns the sentence containing an anchor phrase, using
  the project's sentence segmenter on the anchor's line.
* ``extract_span`` returns the exact text from one anchor to another.

The SHA-256 of the extracted passage is stored with the evidence. Re-running
the same extraction on the same source must give the same hash; if it does
not, the source has changed (see :func:`recheck_html_evidence`).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from html.parser import HTMLParser

from hallucination_lab.extraction.segmenter import split_sentences
from hallucination_lab.ids import content_sha256

HTML_TEXT_VERSION = "html-text-v1"
_BLOCK = {
    "p",
    "br",
    "tr",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "div",
    "li",
    "table",
    "section",
    "article",
    "td",
    "th",
    "dt",
    "dd",
    "header",
    "footer",
    "nav",
    "blockquote",
}
_SKIP = {"script", "style", "noscript", "svg", "template"}


class CaptureError(ValueError):
    """The anchor could not be found, so no passage was captured."""


class _TextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP:
            self.skip += 1
        elif tag in _BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in _SKIP:
            self.skip = max(0, self.skip - 1)
        elif tag in _BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    parser = _TextParser()
    parser.feed(html)
    raw = "".join(parser.parts).replace("\xa0", " ")
    return "\n".join(" ".join(line.split()) for line in raw.splitlines() if line.strip())


@dataclass(frozen=True)
class Captured:
    text: str
    sha256: str
    start: int
    end: int
    line: int  # 1-based line of the passage start

    def locator(self, how: str) -> str:
        return f"{how}; line {self.line}, characters {self.start}-{self.end} of the extracted text"


def _line_no(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def extract_sentence(text: str, anchor: str) -> Captured:
    index = text.find(anchor)
    if index < 0:
        raise CaptureError(f"anchor not found: {anchor!r}")
    line_start = text.rfind("\n", 0, index) + 1
    line_end = text.find("\n", index)
    line_end = len(text) if line_end < 0 else line_end
    line = text[line_start:line_end]
    rel = index - line_start
    for seg in split_sentences(line):
        if seg.start <= rel < seg.end:
            start, end = line_start + seg.start, line_start + seg.end
            if end < index + len(anchor):  # anchor crosses a sentence boundary
                end = index + len(anchor)
            passage = text[start:end]
            return Captured(passage, content_sha256(passage), start, end, _line_no(text, start))
    raise CaptureError(f"could not delimit a sentence around {anchor!r}")


def extract_span(text: str, start_anchor: str, end_anchor: str) -> Captured:
    start = text.find(start_anchor)
    if start < 0:
        raise CaptureError(f"start anchor not found: {start_anchor!r}")
    end_at = text.find(end_anchor, start)
    if end_at < 0:
        raise CaptureError(f"end anchor not found after start: {end_anchor!r}")
    end = end_at + len(end_anchor)
    passage = text[start:end]
    return Captured(passage, content_sha256(passage), start, end, _line_no(text, start))


@dataclass(frozen=True)
class RecheckResult:
    status: str  # unchanged | changed | anchor_missing | unavailable
    detail: str


def recheck_html_evidence(
    url: str, anchor: str, end_anchor: str | None, expected_sha256: str, fetch: Callable[[str], str]
) -> RecheckResult:
    """Re-fetch a page and re-extract the passage; compare with the stored hash."""
    try:
        html = fetch(url)
    except Exception as exc:  # network errors, HTTP errors
        return RecheckResult("unavailable", f"could not fetch: {exc}")
    text = html_to_text(html)
    try:
        captured = (
            extract_span(text, anchor, end_anchor) if end_anchor else extract_sentence(text, anchor)
        )
    except CaptureError as exc:
        return RecheckResult("anchor_missing", str(exc))
    if captured.sha256 == expected_sha256:
        return RecheckResult("unchanged", "passage hash matches the stored capture")
    return RecheckResult("changed", "passage text differs from the stored capture")
