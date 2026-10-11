"""Document loading, validation and passage segmentation.

Canonical text: documents are decoded, a leading byte-order mark is removed,
and line endings are normalised to ``\\n``. Nothing else changes: no case
changes, no Unicode rewriting, no whitespace collapsing. Passage offsets and
line numbers refer to this canonical text, so ``text[char_start:char_end]``
is the exact passage. The raw file bytes are hashed separately so any change
to the stored file can be detected.

Passage segmentation:

1. Split the document into sections at headings. MediaWiki ``== Heading ==``
   and Markdown ``# Heading`` are recognised. Each passage records its section
   path, e.g. ``History > Construction``.
2. Split each section into sentences with the same segmenter used for claims.
3. Pack consecutive sentences into passages of up to ``max_chars`` characters.
   The next passage repeats the last ``overlap_sentences`` sentences, so a fact
   stated across a passage boundary is still retrievable. Passages never cross
   section boundaries.
4. A single "sentence" longer than ``max_chars`` (for example a table without
   punctuation) is cut at a line break or space, and a warning is recorded.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from hallucination_lab import __version__
from hallucination_lab.errors import DocumentIngestionError, DocumentModifiedError
from hallucination_lab.extraction.segmenter import split_sentences
from hallucination_lab.ids import PASSAGE, content_sha256, make_id
from hallucination_lab.models import SourceOrigin, utc_now
from hallucination_lab.retrieval.models import (
    SUPPORTED_ENCODINGS,
    ChunkingConfig,
    CollectionManifest,
    DocumentEntry,
    IngestionWarning,
    Passage,
    ProcessedCollection,
    ProcessedDocument,
    SourceMetadata,
    SystemFacts,
    source_id_for,
)
from hallucination_lab.retrieval.tokenizer import TOKENIZER_VERSION

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = frozenset({".txt", ".md", ".markdown"})
DEFAULT_MAX_BYTES = 5_000_000
SHORT_DOCUMENT_CHARS = 200

_WIKI_HEADING = re.compile(r"[ \t]*(={2,6})[ \t]*(.+?)[ \t]*\1[ \t]*")
_MD_HEADING = re.compile(r"[ \t]{0,3}(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*")


@dataclass(frozen=True)
class LoadedDocument:
    raw_bytes: bytes
    text: str  # canonical text
    system: SystemFacts
    warnings: list[IngestionWarning]


# ----------------------------------------------------------------------- loading


def _decode(raw: bytes, encoding: str, name: str) -> tuple[str, str]:
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw[3:].decode("utf-8"), "utf-8"
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16"), "utf-16"
    if encoding not in SUPPORTED_ENCODINGS:
        raise DocumentIngestionError(
            f"{name}: unsupported encoding {encoding!r}; "
            f"use one of {', '.join(SUPPORTED_ENCODINGS)}"
        )
    try:
        return raw.decode(encoding), encoding
    except UnicodeDecodeError as exc:
        raise DocumentIngestionError(
            f"{name}: not valid {encoding} text (byte {exc.start}). Re-save the file as UTF-8, "
            "or declare its encoding in the metadata."
        ) from exc


def canonicalize(text: str) -> str:
    """Remove a BOM and normalise line endings. Factual content is untouched."""
    return text.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")


def load_document(
    path: Path, encoding: str = "utf-8", max_bytes: int = DEFAULT_MAX_BYTES
) -> LoadedDocument:
    """Read and validate one reference document."""
    name = path.name
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise DocumentIngestionError(
            f"{name}: unsupported file type {path.suffix or '(none)'!r}; supported: "
            f"{', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    if not path.is_file():
        raise DocumentIngestionError(f"{name}: file not found at {path}")
    size = path.stat().st_size
    if size > max_bytes:
        raise DocumentIngestionError(
            f"{name}: {size:,} bytes exceeds the {max_bytes:,}-byte limit; split the document"
        )
    raw = path.read_bytes()
    if not raw.strip():
        raise DocumentIngestionError(f"{name}: document is empty")
    decoded, used = _decode(raw, encoding, name)
    text = canonicalize(decoded)
    if "\x00" in text:
        raise DocumentIngestionError(f"{name}: contains NUL bytes; this looks like a binary file")
    controls = sum(1 for ch in text if ord(ch) < 32 and ch not in "\n\t")
    if controls > max(5, len(text) // 100):
        raise DocumentIngestionError(f"{name}: too many control characters for a text document")
    if not text.strip():
        raise DocumentIngestionError(f"{name}: document contains only whitespace")

    warnings = []
    if len(text.strip()) < SHORT_DOCUMENT_CHARS:
        warnings.append(
            IngestionWarning(
                code="short_document",
                message=f"only {len(text.strip())} characters of text",
                local_filename=name,
            )
        )
    if "�" in text:
        warnings.append(
            IngestionWarning(
                code="replacement_characters",
                message="contains U+FFFD replacement characters; the file may be mis-encoded",
                local_filename=name,
            )
        )
    system = SystemFacts(
        content_sha256=hashlib.sha256(raw).hexdigest(),
        text_sha256=content_sha256(text),
        byte_size=len(raw),
        char_count=len(text),
        line_count=text.count("\n") + 1,
        encoding=used,
    )
    return LoadedDocument(raw, text, system, warnings)


# ------------------------------------------------------------------ segmentation


@dataclass(frozen=True)
class _Section:
    path: str | None
    start: int
    end: int


def _sections(text: str) -> list[_Section]:
    sections: list[_Section] = []
    stack: list[tuple[int, str]] = []  # (level, title)
    body_start = 0
    pos = 0
    for line in text.split("\n"):
        line_end = pos + len(line)
        match = _WIKI_HEADING.fullmatch(line) or _MD_HEADING.fullmatch(line)
        if match:
            current = " > ".join(t for _, t in stack) or None
            sections.append(_Section(current, body_start, pos))
            level = len(match.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, match.group(2).strip()))
            body_start = line_end + 1
        pos = line_end + 1
    sections.append(_Section(" > ".join(t for _, t in stack) or None, body_start, len(text)))
    return [s for s in sections if s.end > s.start and text[s.start : s.end].strip()]


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _hard_split(text: str, start: int, end: int, max_chars: int) -> list[tuple[int, int]]:
    pieces = []
    while end - start > max_chars:
        window = text[start : start + max_chars]
        cut = window.rfind("\n")
        if cut < max_chars // 2:
            cut = window.rfind(" ")
        if cut <= 0:
            cut = max_chars
        s, e = _trim(text, start, start + cut)
        if e > s:
            pieces.append((s, e))
        start, _ = _trim(text, start + cut, end)
    s, e = _trim(text, start, end)
    if e > s:
        pieces.append((s, e))
    return pieces


def segment_passages(
    text: str, source_id: str, config: ChunkingConfig, filename: str | None = None
) -> tuple[list[Passage], list[IngestionWarning]]:
    """Divide canonical text into passages with exact offsets."""
    line_starts = [0] + [i + 1 for i, ch in enumerate(text) if ch == "\n"]

    def line_of(offset: int) -> int:
        return bisect.bisect_right(line_starts, offset)

    passages: list[Passage] = []
    oversized = 0
    for section in _sections(text):
        units: list[tuple[int, int]] = []
        for seg in split_sentences(text[section.start : section.end]):
            s, e = section.start + seg.start, section.start + seg.end
            if e - s > config.max_chars:
                oversized += 1
                units.extend(_hard_split(text, s, e, config.max_chars))
            else:
                units.append((s, e))
        i = 0
        while i < len(units):
            j = i + 1
            while j < len(units) and units[j][1] - units[i][0] <= config.max_chars:
                j += 1
            start, end = units[i][0], units[j - 1][1]
            body = text[start:end]
            passages.append(
                Passage(
                    passage_id=make_id(PASSAGE, source_id, start, end),
                    source_id=source_id,
                    passage_index=len(passages),
                    char_start=start,
                    char_end=end,
                    line_start=line_of(start),
                    line_end=line_of(end - 1),
                    section=section.path,
                    text=body,
                    text_sha256=content_sha256(body),
                )
            )
            if j >= len(units):
                break
            i = max(j - config.overlap_sentences, i + 1)

    warnings = []
    if oversized:
        warnings.append(
            IngestionWarning(
                code="long_unpunctuated_text",
                message=f"{oversized} segment(s) longer than {config.max_chars} characters were "
                "cut at line breaks or spaces",
                local_filename=filename,
            )
        )
    return passages, warnings


# -------------------------------------------------------------------- collection


def collection_fingerprint(manifest: CollectionManifest, chunking: ChunkingConfig) -> str:
    """Changes whenever documents, their metadata, chunking or tokenizer change."""
    payload = {
        "name": manifest.name,
        "documents": [
            [d.source_id, d.local_filename, d.content_sha256, d.metadata.model_dump(mode="json")]
            for d in manifest.documents
        ],
        "chunking": chunking.model_dump(),
        "tokenizer": TOKENIZER_VERSION,
    }
    return content_sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def check_documents(manifest: CollectionManifest, documents_dir: Path) -> dict[str, str]:
    """Return {local_filename: problem} for missing or modified documents."""
    problems: dict[str, str] = {}
    for entry in manifest.documents:
        path = documents_dir / entry.local_filename
        if not path.is_file():
            problems[entry.local_filename] = "missing"
        elif hashlib.sha256(path.read_bytes()).hexdigest() != entry.content_sha256:
            problems[entry.local_filename] = "modified since it was registered"
    return problems


def ingest_collection(
    manifest: CollectionManifest,
    documents_dir: Path,
    chunking: ChunkingConfig | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    now: datetime | None = None,
) -> ProcessedCollection:
    chunking = chunking or ChunkingConfig()
    if not manifest.documents:
        raise DocumentIngestionError(f"collection {manifest.name!r} has no documents to index")
    problems = check_documents(manifest, documents_dir)
    if problems:
        detail = "; ".join(f"{name}: {issue}" for name, issue in sorted(problems.items()))
        raise DocumentModifiedError(
            f"collection {manifest.name!r} failed integrity check ({detail}). Restore the "
            "original files, or re-register changed files with `hallab sources add --replace`."
        )

    documents: list[ProcessedDocument] = []
    passages: list[Passage] = []
    warnings: list[IngestionWarning] = []
    for entry in manifest.documents:
        loaded = load_document(documents_dir / entry.local_filename, entry.encoding, max_bytes)
        doc_passages, doc_warnings = segment_passages(
            loaded.text, entry.source_id, chunking, entry.local_filename
        )
        if not doc_passages:
            raise DocumentIngestionError(f"{entry.local_filename}: produced no passages")
        documents.append(
            ProcessedDocument(entry=entry, system=loaded.system, passage_count=len(doc_passages))
        )
        passages.extend(doc_passages)
        warnings.extend(loaded.warnings + doc_warnings)
    logger.info(
        "indexed collection %s: %d documents, %d passages, %d warnings",
        manifest.name,
        len(documents),
        len(passages),
        len(warnings),
    )
    return ProcessedCollection(
        collection_name=manifest.name,
        origin=manifest.origin,
        fingerprint=collection_fingerprint(manifest, chunking),
        built_at=now or utc_now(),
        tool_version=__version__,
        tokenizer_version=TOKENIZER_VERSION,
        chunking=chunking,
        documents=documents,
        passages=passages,
        warnings=warnings,
    )


def new_document_entry(
    loaded: LoadedDocument,
    local_filename: str,
    metadata: SourceMetadata,
    origin: SourceOrigin,
    now: datetime | None = None,
) -> DocumentEntry:
    """Register a loaded document. Hash, size and encoding come from the file, not the user."""
    return DocumentEntry(
        source_id=source_id_for(loaded.system.content_sha256),
        local_filename=local_filename,
        origin=origin,
        metadata=metadata,
        content_sha256=loaded.system.content_sha256,
        byte_size=loaded.system.byte_size,
        encoding=loaded.system.encoding,
        added_at=now or utc_now(),
    )
