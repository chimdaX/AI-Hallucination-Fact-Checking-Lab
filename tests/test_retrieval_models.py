"""Source schema validation for reference collections."""

from datetime import date

import pytest
from pydantic import ValidationError

from hallucination_lab.ids import content_sha256
from hallucination_lab.models import SourceOrigin
from hallucination_lab.retrieval.models import (
    ChunkingConfig,
    CollectionManifest,
    DocumentEntry,
    MetadataStatus,
    Passage,
    SourceMetadata,
    source_id_for,
)
from tests.conftest import FIXED_TIME

HASH = "a" * 64


def _entry(**overrides) -> DocumentEntry:
    data = dict(
        source_id=source_id_for(HASH),
        local_filename="doc.txt",
        origin=SourceOrigin.USER_ADDED,
        metadata=SourceMetadata(title="Doc"),
        content_sha256=HASH,
        byte_size=10,
        added_at=FIXED_TIME,
    )
    data.update(overrides)
    return DocumentEntry(**data)


def test_metadata_defaults_to_user_supplied_and_unverified():
    meta = SourceMetadata(title="Some report")
    assert meta.status is MetadataStatus.USER_SUPPLIED
    assert meta.publication_date is None and meta.url is None  # never invented


@pytest.mark.parametrize(
    "status", [MetadataStatus.CAPTURED_AT_DOWNLOAD, MetadataStatus.MANUALLY_CONFIRMED]
)
def test_non_user_status_requires_explanation(status):
    with pytest.raises(ValidationError, match="status_note"):
        SourceMetadata(title="X", status=status)
    assert SourceMetadata(title="X", status=status, status_note="checked by reviewer-1")


def test_last_updated_not_before_publication():
    with pytest.raises(ValidationError, match="last_updated"):
        SourceMetadata(title="X", publication_date=date(2020, 1, 2), last_updated=date(2020, 1, 1))


@pytest.mark.parametrize("url", ["ftp://x.org/a", "www.example.org", "http://has space"])
def test_metadata_rejects_bad_urls(url):
    with pytest.raises(ValidationError):
        SourceMetadata(title="X", url=url)


def test_metadata_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        SourceMetadata(title="X", confidence=0.9)


def test_source_id_is_content_addressed():
    assert _entry().source_id == source_id_for(HASH)
    with pytest.raises(ValidationError, match="derived from content_sha256"):
        _entry(source_id=source_id_for("b" * 64))


@pytest.mark.parametrize("name", ["../evil.txt", "a/b.txt", "a\\b.txt", "..", "bad name.txt"])
def test_filenames_cannot_escape_collection(name):
    with pytest.raises(ValidationError):
        _entry(local_filename=name)


def test_manifest_rejects_duplicate_documents():
    with pytest.raises(ValidationError, match="duplicate"):
        CollectionManifest(
            name="x",
            origin=SourceOrigin.USER_ADDED,
            created_at=FIXED_TIME,
            documents=[_entry(), _entry(local_filename="other.txt")],
        )
    with pytest.raises(ValidationError, match="local_filename"):
        CollectionManifest(
            name="x",
            origin=SourceOrigin.USER_ADDED,
            created_at=FIXED_TIME,
            documents=[
                _entry(),
                _entry(
                    local_filename="DOC.TXT",
                    content_sha256="b" * 64,
                    source_id=source_id_for("b" * 64),
                ),
            ],
        )


@pytest.mark.parametrize("name", ["Demo", "has space", "", "-start", "x" * 65])
def test_collection_name_rules(name):
    with pytest.raises(ValidationError):
        CollectionManifest(name=name, origin=SourceOrigin.USER_ADDED, created_at=FIXED_TIME)


def _passage(text="Some passage.", start=0, **kw) -> Passage:
    data = dict(
        passage_id="psg_0000000000000001",
        source_id=source_id_for(HASH),
        passage_index=0,
        char_start=start,
        char_end=start + len(text),
        line_start=1,
        line_end=1,
        text=text,
        text_sha256=content_sha256(text),
    )
    data.update(kw)
    return Passage(**data)


def test_passage_hash_detects_alteration():
    assert _passage()
    with pytest.raises(ValidationError, match="hash"):
        _passage(text_sha256=content_sha256("Different text."))


def test_passage_offsets_must_match_text():
    with pytest.raises(ValidationError, match="offsets"):
        _passage(char_end=99)


def test_chunking_bounds():
    with pytest.raises(ValidationError):
        ChunkingConfig(max_chars=10)
    with pytest.raises(ValidationError):
        ChunkingConfig(overlap_sentences=-1)
