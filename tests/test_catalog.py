"""Reference collection management, integrity checks and index lifecycle."""

import json

import pytest

from hallucination_lab.errors import (
    CollectionNotFoundError,
    DocumentIngestionError,
    DocumentModifiedError,
    DuplicateDocumentError,
    IndexNotFoundError,
    ReadOnlyCollectionError,
    ReferenceCollectionError,
    StaleIndexError,
)
from hallucination_lab.models import SourceOrigin
from hallucination_lab.retrieval.models import ChunkingConfig
from tests.conftest import synthetic_metadata


def test_add_document_registers_and_copies(catalog, synthetic_files):
    entry, _ = catalog.add_document("mine", synthetic_files["lake"], synthetic_metadata("Lake"))
    manifest = catalog.load_manifest("mine")
    assert manifest.origin is SourceOrigin.USER_ADDED
    assert entry.origin is SourceOrigin.USER_ADDED  # distinguished from packaged demo docs
    copied = catalog.documents_dir("mine") / "lake_verenna.md"
    assert copied.read_bytes() == synthetic_files["lake"].read_bytes()
    assert catalog.verify("mine") == {}


def test_duplicate_content_detected_even_under_another_name(catalog, synthetic_files, tmp_path):
    catalog.add_document("mine", synthetic_files["lake"], synthetic_metadata("Lake"))
    copy = tmp_path / "renamed_copy.md"
    copy.write_bytes(synthetic_files["lake"].read_bytes())
    with pytest.raises(DuplicateDocumentError, match="identical content"):
        catalog.add_document("mine", copy, synthetic_metadata("Copy"))


def test_same_filename_requires_replace(catalog, synthetic_files):
    catalog.add_document("mine", synthetic_files["lake"], synthetic_metadata("Lake"))
    synthetic_files["lake"].write_text("Revised text about Lake Verenna for tests.\n")
    with pytest.raises(ReferenceCollectionError, match="--replace"):
        catalog.add_document("mine", synthetic_files["lake"], synthetic_metadata("Lake v2"))
    entry, _ = catalog.add_document(
        "mine", synthetic_files["lake"], synthetic_metadata("Lake v2"), replace=True
    )
    docs = catalog.load_manifest("mine").documents
    assert [d.source_id for d in docs] == [entry.source_id]  # new content -> new source ID


def test_packaged_demo_is_read_only(catalog, synthetic_files):
    assert catalog.is_packaged("demo")
    with pytest.raises(ReadOnlyCollectionError):
        catalog.add_document("demo", synthetic_files["lake"], synthetic_metadata("Lake"))
    with pytest.raises(ReadOnlyCollectionError):
        catalog.create_collection("demo")


def test_unsupported_input_rejected_before_registration(catalog, tmp_path):
    bad = tmp_path / "paper.pdf"
    bad.write_bytes(b"%PDF-1.7")
    with pytest.raises(DocumentIngestionError):
        catalog.add_document("mine", bad, synthetic_metadata("Paper"))
    assert catalog.load_manifest("mine").documents == []


def test_missing_collection(catalog):
    with pytest.raises(CollectionNotFoundError, match="available"):
        catalog.load_manifest("nope")


def test_invalid_collection_name(catalog):
    with pytest.raises(ReferenceCollectionError, match="invalid collection name"):
        catalog.create_collection("Bad Name")


def test_list_collections_includes_demo_and_user(catalog, test_collection):
    infos = {c.name: c for c in catalog.list_collections()}
    assert infos["demo"].origin is SourceOrigin.PACKAGED_DEMO
    assert infos[test_collection].origin is SourceOrigin.USER_ADDED
    assert infos[test_collection].document_count == 2


def test_index_lifecycle(catalog, test_collection, synthetic_files, tmp_path):
    with pytest.raises(IndexNotFoundError, match="hallab index build"):
        catalog.load_index(test_collection)
    built = catalog.build_index(test_collection)
    assert catalog.load_index(test_collection) == built

    # Adding a document makes the saved index stale (never silently used).
    extra = tmp_path / "extra.txt"
    extra.write_text("An extra synthetic note about Brindle harbour and its ferries.\n")
    catalog.add_document(test_collection, extra, synthetic_metadata("Extra"))
    with pytest.raises(StaleIndexError):
        catalog.load_index(test_collection)
    searcher, rebuilt = catalog.searcher(test_collection)
    assert rebuilt and len(searcher.collection.documents) == 3


def test_reindex_with_new_chunking(catalog, test_collection):
    catalog.build_index(test_collection)
    searcher, rebuilt = catalog.searcher(test_collection, chunking=ChunkingConfig(max_chars=150))
    assert rebuilt and searcher.collection.chunking.max_chars == 150
    with pytest.raises(StaleIndexError):
        catalog.searcher(test_collection, auto_build=False, chunking=ChunkingConfig())


def test_modified_document_blocks_search_even_with_auto_build(catalog, test_collection):
    catalog.build_index(test_collection)
    doc = catalog.documents_dir(test_collection) / "lake_verenna.md"
    doc.write_text(doc.read_text().replace("312 m", "999 m"))
    assert catalog.verify(test_collection) == {
        "lake_verenna.md": "modified since it was registered"
    }
    with pytest.raises(DocumentModifiedError):
        catalog.searcher(test_collection)


def test_corrupt_index_reported(catalog, test_collection):
    catalog.build_index(test_collection)
    catalog.index_path(test_collection).write_text("{broken")
    with pytest.raises(ReferenceCollectionError, match="not valid JSON"):
        catalog.load_index(test_collection)


def test_tampered_index_passage_rejected(catalog, test_collection):
    catalog.build_index(test_collection)
    path = catalog.index_path(test_collection)
    data = json.loads(path.read_text())
    text = data["passages"][0]["text"]
    data["passages"][0]["text"] = text[:-1] + ("?" if text[-1] != "?" else "!")  # same length
    path.write_text(json.dumps(data))
    with pytest.raises(ReferenceCollectionError, match="failed validation"):
        catalog.load_index(test_collection)
