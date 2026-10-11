"""Reference collection models: source metadata, documents, passages.

Metadata is split into two parts on purpose:

* :class:`SourceMetadata` holds what someone *declared* about a document
  (title, author, URL, dates, licence). The system cannot verify these
  offline. Each record carries a :class:`MetadataStatus` saying where the
  values came from.
* :class:`SystemFacts` holds what the system *computed* from the file itself
  (hashes, byte size, encoding, character count). These are never taken from
  user input.
"""

from __future__ import annotations

import re
from datetime import date
from enum import Enum
from typing import Annotated, Literal

from pydantic import Field, StringConstraints, field_validator, model_validator

from hallucination_lab.ids import SOURCE, content_sha256, make_id
from hallucination_lab.models import (
    AwareDatetime,
    Identifier,
    NonBlankStr,
    OptionalUrl,
    Sha256Hex,
    SourceOrigin,
    StrictModel,
)

MANIFEST_VERSION = "1.0.0"
PROCESSED_FORMAT_VERSION = "1.0.0"
SUPPORTED_ENCODINGS = ("utf-8", "utf-16", "latin-1", "cp1252")

CollectionName = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")]


class MetadataStatus(str, Enum):
    """Where the declared metadata came from. None of these mean the content is true."""

    USER_SUPPLIED = "user_supplied"  # typed in by whoever added the document; unchecked
    CAPTURED_AT_DOWNLOAD = "captured_at_download"  # recorded from the publisher's API/page
    MANUALLY_CONFIRMED = "manually_confirmed"  # a named person checked it against the source


class DocumentType(str, Enum):
    ENCYCLOPEDIA_ARTICLE = "encyclopedia_article"
    QUOTATION_COMPILATION = "quotation_compilation"
    FACT_SHEET = "fact_sheet"
    OFFICIAL_DOCUMENTATION = "official_documentation"
    SCIENTIFIC_PUBLICATION = "scientific_publication"
    NEWS_ARTICLE = "news_article"
    BOOK = "book"
    DATASET = "dataset"
    SYNTHETIC_FIXTURE = "synthetic_fixture"
    OTHER = "other"


class SourceMetadata(StrictModel):
    """Declared bibliographic metadata. Leave a field empty rather than guess."""

    title: NonBlankStr
    author_or_organization: str | None = None
    publisher: str | None = None
    publication_date: date | None = None
    last_updated: date | None = None
    url: OptionalUrl = None
    accessed_on: date | None = None
    document_type: DocumentType = DocumentType.OTHER
    license: str | None = None
    license_url: OptionalUrl = None
    source_version: str | None = None
    status: MetadataStatus = MetadataStatus.USER_SUPPLIED
    status_note: str | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def _consistent(self) -> SourceMetadata:
        if (
            self.publication_date
            and self.last_updated
            and self.last_updated < self.publication_date
        ):
            raise ValueError("last_updated cannot be earlier than publication_date")
        if self.status is not MetadataStatus.USER_SUPPLIED and not self.status_note:
            raise ValueError(
                f"metadata status {self.status.value!r} requires status_note explaining "
                "how and by whom the metadata was captured or confirmed"
            )
        return self


class SystemFacts(StrictModel):
    """Facts computed by the system from the document bytes during ingestion."""

    content_sha256: Sha256Hex  # hash of the raw file bytes
    text_sha256: Sha256Hex  # hash of the canonical decoded text that offsets refer to
    byte_size: int = Field(ge=1)
    char_count: int = Field(ge=1)
    line_count: int = Field(ge=1)
    encoding: str


def _check_filename(value: str) -> str:
    if (
        value in {".", ".."}
        or "/" in value
        or "\\" in value
        or not re.fullmatch(r"[\w.\-]+", value)
    ):
        raise ValueError(
            f"local_filename must be a plain file name (letters, digits, . _ -), got {value!r}"
        )
    return value


class DocumentEntry(StrictModel):
    """One registered document in a collection manifest."""

    source_id: Identifier
    local_filename: str
    origin: SourceOrigin
    metadata: SourceMetadata
    content_sha256: Sha256Hex  # registered when the document was added
    byte_size: int = Field(ge=1)
    encoding: Literal["utf-8", "utf-16", "latin-1", "cp1252"] = "utf-8"
    added_at: AwareDatetime

    @field_validator("local_filename")
    @classmethod
    def _filename(cls, value: str) -> str:
        return _check_filename(value)

    @model_validator(mode="after")
    def _content_addressed(self) -> DocumentEntry:
        if self.source_id != source_id_for(self.content_sha256):
            raise ValueError("source_id must be derived from content_sha256")
        return self


def source_id_for(content_hash: str) -> str:
    """Source IDs are content-addressed: same bytes, same ID; edited bytes, new ID."""
    return make_id(SOURCE, content_hash)


class CollectionManifest(StrictModel):
    """``collection.json``: the registry of documents in a collection."""

    manifest_version: Literal["1.0.0"] = MANIFEST_VERSION
    name: CollectionName
    description: str | None = None
    origin: SourceOrigin
    created_at: AwareDatetime
    documents: list[DocumentEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique(self) -> CollectionManifest:
        for label, values in (
            ("source_id", [d.source_id for d in self.documents]),
            ("local_filename", [d.local_filename.lower() for d in self.documents]),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"duplicate {label} in collection {self.name}")
        return self

    def get(self, source_id: str) -> DocumentEntry:
        for doc in self.documents:
            if doc.source_id == source_id:
                return doc
        raise KeyError(source_id)


class ChunkingConfig(StrictModel):
    """How documents are divided into passages."""

    max_chars: int = Field(default=800, ge=100, le=10_000)
    overlap_sentences: int = Field(default=1, ge=0, le=5)


class Passage(StrictModel):
    """A retrievable span of a document. ``text`` is an exact slice of the canonical text."""

    passage_id: Identifier
    source_id: Identifier
    passage_index: int = Field(ge=0)
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    section: str | None = None
    text: str = Field(min_length=1)
    text_sha256: Sha256Hex

    @model_validator(mode="after")
    def _integrity(self) -> Passage:
        if self.char_end - self.char_start != len(self.text):
            raise ValueError(f"passage {self.passage_id}: offsets do not match text length")
        if content_sha256(self.text) != self.text_sha256:
            raise ValueError(f"passage {self.passage_id}: text does not match its hash")
        if self.line_end < self.line_start:
            raise ValueError("line_end must not be before line_start")
        return self


class IngestionWarning(StrictModel):
    code: NonBlankStr
    message: NonBlankStr
    local_filename: str | None = None


class ProcessedDocument(StrictModel):
    entry: DocumentEntry
    system: SystemFacts
    passage_count: int = Field(ge=0)


class ProcessedCollection(StrictModel):
    """Saved output of ``hallab index build``: documents, passages and build settings.

    BM25 statistics are recomputed from the passages when loaded (cheap and
    deterministic), so they do not need to be stored.
    """

    format_version: Literal["1.0.0"] = PROCESSED_FORMAT_VERSION
    collection_name: CollectionName
    origin: SourceOrigin
    fingerprint: Sha256Hex
    built_at: AwareDatetime
    tool_version: str
    tokenizer_version: str
    chunking: ChunkingConfig
    documents: list[ProcessedDocument]
    passages: list[Passage]
    warnings: list[IngestionWarning] = Field(default_factory=list)

    @model_validator(mode="after")
    def _references(self) -> ProcessedCollection:
        source_ids = {d.entry.source_id for d in self.documents}
        passage_ids = [p.passage_id for p in self.passages]
        if len(passage_ids) != len(set(passage_ids)):
            raise ValueError("duplicate passage_id in processed collection")
        for passage in self.passages:
            if passage.source_id not in source_ids:
                raise ValueError(f"passage {passage.passage_id} references unknown source")
        return self

    def document(self, source_id: str) -> ProcessedDocument:
        for doc in self.documents:
            if doc.entry.source_id == source_id:
                return doc
        raise KeyError(source_id)
