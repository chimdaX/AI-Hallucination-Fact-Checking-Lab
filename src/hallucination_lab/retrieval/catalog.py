"""Locating, modifying and indexing reference collections on disk.

Layout of a collection directory::

    <collection>/
      collection.json        manifest (CollectionManifest)
      documents/<file>.txt   original document files, never rewritten by indexing

* The packaged ``demo`` collection lives in ``data/references/demo`` in the
  repository and is read-only through the tool.
* User collections live in ``<data_dir>/collections/<name>``.
* Processed indexes live in ``<data_dir>/indexes/<name>.json`` and can always
  be regenerated.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ValidationError

from hallucination_lab.config import Settings
from hallucination_lab.errors import (
    CollectionNotFoundError,
    DocumentIngestionError,
    DuplicateDocumentError,
    IndexNotFoundError,
    ReadOnlyCollectionError,
    ReferenceCollectionError,
    StaleIndexError,
)
from hallucination_lab.models import SourceOrigin, utc_now
from hallucination_lab.retrieval.ingest import (
    DEFAULT_MAX_BYTES,
    check_documents,
    collection_fingerprint,
    ingest_collection,
    load_document,
    new_document_entry,
    segment_passages,
)
from hallucination_lab.retrieval.models import (
    ChunkingConfig,
    CollectionManifest,
    DocumentEntry,
    IngestionWarning,
    ProcessedCollection,
    SourceMetadata,
)
from hallucination_lab.retrieval.search import CollectionSearcher
from hallucination_lab.storage import summarize_validation_error, write_json_atomic

logger = logging.getLogger(__name__)

PACKAGED_REFERENCES_DIR = Path(__file__).resolve().parents[3] / "data" / "references"
MANIFEST_NAME = "collection.json"
DOCUMENTS_DIR = "documents"


@dataclass(frozen=True)
class CollectionInfo:
    name: str
    origin: SourceOrigin
    path: Path
    document_count: int
    indexed: bool


def _write_bytes_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _read_model(path: Path, model: type[BaseModel], what: str) -> BaseModel:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReferenceCollectionError(f"{what} {path} is not valid JSON: {exc}") from exc
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise ReferenceCollectionError(
            f"{what} {path} failed validation: {summarize_validation_error(exc)}"
        ) from exc


class CollectionCatalog:
    def __init__(self, settings: Settings, packaged_dir: Path = PACKAGED_REFERENCES_DIR) -> None:
        self.settings = settings
        self.packaged_dir = packaged_dir

    # --- locating ------------------------------------------------------------------

    def is_packaged(self, name: str) -> bool:
        return (self.packaged_dir / name / MANIFEST_NAME).is_file()

    def collection_dir(self, name: str) -> Path:
        if self.is_packaged(name):
            return self.packaged_dir / name
        return self.settings.collections_dir / name

    def exists(self, name: str) -> bool:
        return (self.collection_dir(name) / MANIFEST_NAME).is_file()

    def documents_dir(self, name: str) -> Path:
        return self.collection_dir(name) / DOCUMENTS_DIR

    def index_path(self, name: str) -> Path:
        return self.settings.indexes_dir / f"{name}.json"

    def load_manifest(self, name: str) -> CollectionManifest:
        path = self.collection_dir(name) / MANIFEST_NAME
        if not path.is_file():
            known = ", ".join(c.name for c in self.list_collections()) or "none"
            raise CollectionNotFoundError(
                f"no reference collection named {name!r} (available: {known})"
            )
        manifest = _read_model(path, CollectionManifest, "collection manifest")
        if manifest.name != name:
            raise ReferenceCollectionError(f"{path} declares name {manifest.name!r}, not {name!r}")
        return manifest

    def list_collections(self) -> list[CollectionInfo]:
        infos: dict[str, CollectionInfo] = {}
        for root in (self.packaged_dir, self.settings.collections_dir):
            if not root.is_dir():
                continue
            for manifest_path in sorted(root.glob(f"*/{MANIFEST_NAME}")):
                name = manifest_path.parent.name
                if name in infos:
                    continue  # packaged collections take precedence
                try:
                    manifest = self.load_manifest(name)
                except ReferenceCollectionError as exc:
                    logger.warning("skipping unreadable collection %s: %s", name, exc)
                    continue
                infos[name] = CollectionInfo(
                    name=name,
                    origin=manifest.origin,
                    path=manifest_path.parent,
                    document_count=len(manifest.documents),
                    indexed=self.index_path(name).is_file(),
                )
        return list(infos.values())

    # --- modifying ------------------------------------------------------------------

    def create_collection(self, name: str, description: str | None = None) -> CollectionManifest:
        if self.is_packaged(name):
            raise ReadOnlyCollectionError(f"{name!r} is a packaged collection name")
        if self.exists(name):
            raise ReferenceCollectionError(f"collection {name!r} already exists")
        try:
            manifest = CollectionManifest(
                name=name,
                description=description,
                origin=SourceOrigin.USER_ADDED,
                created_at=utc_now(),
            )
        except ValidationError as exc:
            raise ReferenceCollectionError(
                f"invalid collection name {name!r}: use lowercase letters, digits, - and _"
            ) from exc
        self._save_manifest(manifest)
        return manifest

    def _save_manifest(self, manifest: CollectionManifest) -> None:
        write_json_atomic(self.collection_dir(manifest.name) / MANIFEST_NAME, manifest)

    def add_document(
        self,
        name: str,
        file: Path,
        metadata: SourceMetadata,
        *,
        encoding: str = "utf-8",
        replace: bool = False,
        create: bool = True,
        max_bytes: int = DEFAULT_MAX_BYTES,
        now: datetime | None = None,
    ) -> tuple[DocumentEntry, list[IngestionWarning]]:
        """Validate a document, copy it into the collection and register it."""
        if self.is_packaged(name):
            raise ReadOnlyCollectionError(
                f"{name!r} is the packaged demonstration collection and is read-only; "
                "add documents to your own collection, e.g. --collection my-refs"
            )
        if not self.exists(name):
            if not create:
                raise CollectionNotFoundError(f"no reference collection named {name!r}")
            self.create_collection(name)
        manifest = self.load_manifest(name)

        loaded = load_document(Path(file), encoding, max_bytes)
        entry = new_document_entry(
            loaded, Path(file).name, metadata, SourceOrigin.USER_ADDED, now=now
        )
        passages, seg_warnings = segment_passages(
            loaded.text, entry.source_id, ChunkingConfig(), entry.local_filename
        )
        if not passages:
            raise DocumentIngestionError(f"{entry.local_filename}: no retrievable text found")

        for existing in manifest.documents:
            if existing.content_sha256 == entry.content_sha256:
                raise DuplicateDocumentError(
                    f"{entry.local_filename} has identical content to already registered "
                    f"{existing.local_filename} ({existing.source_id})"
                )
        clash = [
            d
            for d in manifest.documents
            if d.local_filename.lower() == entry.local_filename.lower()
        ]
        if clash and not replace:
            raise ReferenceCollectionError(
                f"a document named {entry.local_filename} is already registered; pass "
                "--replace to register the new content (the old entry is removed)"
            )
        remaining = [d for d in manifest.documents if d not in clash]
        _write_bytes_atomic(self.documents_dir(name) / entry.local_filename, loaded.raw_bytes)
        updated = manifest.model_copy(update={"documents": [*remaining, entry]})
        self._save_manifest(CollectionManifest.model_validate(updated.model_dump()))
        logger.info("registered %s in collection %s", entry.local_filename, name)
        return entry, loaded.warnings + seg_warnings

    # --- indexing -------------------------------------------------------------------

    def verify(self, name: str) -> dict[str, str]:
        """{filename: problem} for missing/modified documents; empty means all intact."""
        return check_documents(self.load_manifest(name), self.documents_dir(name))

    def build_index(self, name: str, chunking: ChunkingConfig | None = None) -> ProcessedCollection:
        manifest = self.load_manifest(name)
        processed = ingest_collection(manifest, self.documents_dir(name), chunking)
        write_json_atomic(self.index_path(name), processed)
        return processed

    def load_index(self, name: str) -> ProcessedCollection:
        """Load a saved index and confirm it still matches the collection on disk."""
        manifest = self.load_manifest(name)
        path = self.index_path(name)
        if not path.is_file():
            raise IndexNotFoundError(
                f"collection {name!r} is not indexed; run `hallab index build --collection {name}`"
            )
        processed = _read_model(path, ProcessedCollection, "index")
        problems = check_documents(manifest, self.documents_dir(name))
        if problems:
            ingest_collection(manifest, self.documents_dir(name))  # raises a detailed error
        if processed.fingerprint != collection_fingerprint(manifest, processed.chunking):
            raise StaleIndexError(
                f"the index for {name!r} is out of date (documents, metadata or settings "
                f"changed); run `hallab index build --collection {name}`"
            )
        return processed

    def searcher(
        self, name: str, *, auto_build: bool = True, chunking: ChunkingConfig | None = None
    ) -> tuple[CollectionSearcher, bool]:
        """Return (searcher, rebuilt). Missing/stale indexes are rebuilt when allowed.

        Rebuilding re-runs the document integrity check, so modified documents
        still raise an error instead of being silently re-indexed.
        """
        try:
            processed = self.load_index(name)
            if chunking is not None and processed.chunking != chunking:
                raise StaleIndexError("chunking settings differ from the saved index")
            return CollectionSearcher(processed), False
        except (IndexNotFoundError, StaleIndexError):
            if not auto_build:
                raise
        return CollectionSearcher(self.build_index(name, chunking)), True
