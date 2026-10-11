"""Tokenizer used for BM25 indexing and querying.

The same function processes documents and queries, so normalisation only has
to be consistent. It does not need to be linguistically perfect. Steps:

1. Unicode NFKC, casefold, and accent folding (``São`` -> ``sao``).
2. Symbol rewriting: ``°C`` -> ``celsius``, ``°F`` -> ``fahrenheit``, ``%`` -> ``percent``.
3. Numbers are kept as single tokens. Thousands separators are removed
   (``8,848.86`` -> ``8848.86``), and a decimal also emits its integer part
   (``8848``), so "about 8,848 m" can still match "8,848.86 m".
4. Possessive ``'s`` is dropped. Hyphenated words split into their parts.
5. A small English stopword list is removed (numbers are never removed).
6. Unit abbreviations after numbers become words (``8,848 m`` -> ``8848 metre``),
   plus light plural stemming and US/UK spellings (``meters`` -> ``metre``).

Limitations: no synonyms ("year" will not match "orbital period"), no phrase or
proximity matching, and multi-word names are matched word by word.
"""

from __future__ import annotations

import re
import unicodedata

TOKENIZER_VERSION = "lexical-1"  # bump whenever tokenization changes

# "may" is deliberately NOT a stopword: it is also a month ("3 May 1921").
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "then",
        "than",
        "so",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "from",
        "by",
        "with",
        "about",
        "as",
        "into",
        "over",
        "under",
        "after",
        "before",
        "between",
        "during",
        "through",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "am",
        "has",
        "have",
        "had",
        "do",
        "does",
        "did",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "there",
        "their",
        "they",
        "them",
        "he",
        "she",
        "his",
        "her",
        "him",
        "we",
        "our",
        "you",
        "your",
        "i",
        "me",
        "my",
        "which",
        "who",
        "whom",
        "whose",
        "what",
        "when",
        "where",
        "why",
        "how",
        "not",
        "no",
        "nor",
        "can",
        "could",
        "would",
        "should",
        "might",
        "must",
        "will",
        "shall",
        "also",
        "very",
        "just",
        "such",
        "some",
        "any",
        "all",
        "each",
        "both",
        "more",
        "most",
        "other",
        "own",
        "same",
        "only",
        "too",
    ]
)

_SPELLING = {
    "meter": "metre",
    "kilometer": "kilometre",
    "centimeter": "centimetre",
    "millimeter": "millimetre",
    "liter": "litre",
    "center": "centre",
    "color": "colour",
}

_SYMBOLS = [
    (re.compile(r"°\s?c\b", re.IGNORECASE), " celsius "),
    (re.compile(r"°\s?f\b", re.IGNORECASE), " fahrenheit "),
    (re.compile(r"%"), " percent "),
]
# Unit abbreviations directly after a number ("8,848.86 m") -> full unit word.
_UNIT_ABBREVIATIONS = {
    "m": "metre",
    "km": "kilometre",
    "cm": "centimetre",
    "mm": "millimetre",
    "kg": "kilogram",
    "ft": "foot",
}
_UNIT_RE = re.compile(r"(\d)\s?(km|cm|mm|kg|ft|m)\b")
_IRREGULAR = {"feet": "foot"}
_TOKEN_RE = re.compile(r"\d+(?:[.,]\d+)*|[^\W\d_]+(?:'[^\W\d_]+)*")
_THOUSANDS_RE = re.compile(r"^\d{1,3}(?:,\d{3})+(?:\.\d+)?$")


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


# Words ending in "s" that are not plurals (stripping would merge "mars" with "mar").
_NO_STEM = frozenset(
    [
        "mars",
        "species",
        "series",
        "news",
        "physics",
        "mathematics",
        "economics",
        "always",
        "perhaps",
        "chaos",
        "atlas",
        "lens",
        "gas",
        "has",
        "was",
        "does",
        "its",
        "towards",
        "whereas",
    ]
)


def _stem(word: str) -> str:
    if word in _NO_STEM:
        return word
    if len(word) > 4 and word.endswith("ies"):
        word = word[:-3] + "y"
    elif len(word) > 4 and word.endswith(("ches", "shes", "sses", "xes", "zes")):
        word = word[:-2]
    elif len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        word = word[:-1]
    word = _IRREGULAR.get(word, word)
    return _SPELLING.get(word, word)


def _number_tokens(raw: str) -> list[str]:
    if _THOUSANDS_RE.match(raw):
        value = raw.replace(",", "")
    else:
        # A comma not in thousands position separates numbers ("1,2") - keep the parts.
        parts = raw.split(",")
        if len(parts) > 1:
            return [t for p in parts for t in _number_tokens(p)]
        value = raw
    tokens = [value]
    if "." in value:
        integer = value.split(".", 1)[0]
        if integer:
            tokens.append(integer)
    return tokens


def tokenize(text: str) -> list[str]:
    """Return index/query tokens for ``text`` (order preserved, duplicates kept)."""
    text = _fold(text)
    for pattern, replacement in _SYMBOLS:
        text = pattern.sub(replacement, text)
    text = _UNIT_RE.sub(lambda m: f"{m.group(1)} {_UNIT_ABBREVIATIONS[m.group(2)]} ", text)
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(text):
        raw = match.group()
        if raw[0].isdigit():
            tokens.extend(_number_tokens(raw))
            continue
        if raw.endswith("'s"):
            raw = raw[:-2]
        for word in raw.split("'"):
            if len(word) < 2 or word in STOPWORDS:
                continue
            tokens.append(_stem(word))
    return tokens
