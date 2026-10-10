import json

import pytest

from hallucination_lab.errors import (
    InvalidIdentifierError,
    SessionLoadError,
    SessionNotFoundError,
    StorageError,
)
from hallucination_lab.extraction.editing import remove_claim
from hallucination_lab.storage import JsonSessionStore


@pytest.fixture
def store(tmp_path):
    return JsonSessionStore(tmp_path / "sessions")


def test_save_and_load_round_trip(store, session):
    path = store.save(session)
    assert path.exists() and path.name == f"{session.session_id}.json"
    assert store.load(session.session_id) == session


def test_resume_after_edits(store, session):
    """Save, edit, save again, reload: the human edit and audit trail survive."""
    store.save(session)
    target = session.claims[0].claim_id
    edited = session.model_copy(
        update={"claims": remove_claim(session.claims, target, "reviewer-1", "duplicate")}
    ).touch()
    store.save(edited)
    resumed = store.load(session.session_id)
    claim = resumed.get_claim(target)
    assert not claim.is_active
    assert claim.reviewer_edits[-1].reviewer_id == "reviewer-1"


def test_no_temp_files_left_behind(store, session):
    store.save(session)
    store.save(session)
    assert [p.name for p in store.root.iterdir()] == [f"{session.session_id}.json"]


def test_load_missing_session(store):
    with pytest.raises(SessionNotFoundError):
        store.load("ses_0000000000000000")


@pytest.mark.parametrize("bad_id", ["../outside", "ses_../../x", "", "SES_0000000000000000"])
def test_path_traversal_rejected(store, bad_id):
    with pytest.raises(InvalidIdentifierError):
        store.load(bad_id)


def test_corrupt_json_reported(store, session):
    path = store.save(session)
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(SessionLoadError, match="not valid JSON"):
        store.load(session.session_id)


def test_schema_violation_reported(store, session):
    path = store.save(session)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["response"]["text"] = "Tampered text."
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(SessionLoadError, match="schema validation"):
        store.load(session.session_id)


def test_mismatched_session_id_in_file(store, session):
    path = store.save(session)
    other = store.root / "ses_ffffffffffffffff.json"
    other.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(SessionLoadError, match="expected"):
        store.load("ses_ffffffffffffffff")


def test_refuses_to_save_invalid_session(store, session):
    session.claims[0].source_response_id = "rsp_ffffffffffffffff"  # in-place mutation
    with pytest.raises(StorageError, match="refusing to save"):
        store.save(session)
    assert not store.root.exists() or not any(store.root.iterdir())


def test_list_sessions_skips_corrupt_files(store, session):
    store.save(session)
    (store.root / "ses_ffffffffffffffff.json").write_text("garbage", encoding="utf-8")
    summaries, problems = store.list_sessions()
    assert [s.session_id for s in summaries] == [session.session_id]
    assert summaries[0].pending_reviews == len(session.pending_review())
    assert len(problems) == 1


def test_list_sessions_on_missing_directory(tmp_path):
    assert JsonSessionStore(tmp_path / "nope").list_sessions() == ([], [])


def test_delete(store, session):
    store.save(session)
    store.delete(session.session_id)
    assert not store.exists(session.session_id)
    with pytest.raises(SessionNotFoundError):
        store.delete(session.session_id)
