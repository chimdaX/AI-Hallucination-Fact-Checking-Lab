"""Storage guarantees for verification work: conflicts and append-only records."""

import pytest

from hallucination_lab.errors import AuditTrailError, SessionConflictError
from hallucination_lab.models import Verdict, utc_now
from hallucination_lab.verification.workflow import record_verification
from tests.conftest import make_evidence, make_verification


def test_concurrent_modification_detected(verification_env):
    store, sid = verification_env["store"], verification_env["session_id"]
    session, version = store.load_versioned(sid)
    other, other_version = store.load_versioned(sid)
    store.save_versioned(
        other.model_copy(update={"notes": "saved elsewhere"}), expected_version=other_version
    )
    with pytest.raises(SessionConflictError, match="changed on disk"):
        store.save(session.model_copy(update={"notes": "mine"}), expected_version=version)


def test_versioned_saves_chain(verification_env):
    store, sid = verification_env["store"], verification_env["session_id"]
    session, version = store.load_versioned(sid)
    version = store.save_versioned(
        session.model_copy(update={"notes": "a"}), expected_version=version
    )
    store.save_versioned(session.model_copy(update={"notes": "b"}), expected_version=version)
    assert store.load(sid).notes == "b"


@pytest.fixture
def saved_with_verification(store_session):
    store, session = store_session
    cid = session.claims[0].claim_id
    ev = make_evidence(cid)
    v = make_verification(cid, Verdict.SUPPORTED, supporting=[ev.evidence_id])
    session = session.model_copy(update={"evidence": [ev]})
    session = record_verification(session, v)
    store.save(session)
    return store, session, v


@pytest.fixture
def store_session(tmp_path, session):
    from hallucination_lab.storage import JsonSessionStore

    return JsonSessionStore(tmp_path / "sessions"), session.model_copy(
        update={"updated_at": utc_now()}
    )


def test_saved_verification_cannot_be_modified(saved_with_verification):
    store, session, v = saved_with_verification
    edited = v.model_copy(update={"reasoning": "Quietly changed reasoning text here."})
    with pytest.raises(AuditTrailError, match="would be modified"):
        store.save(session.model_copy(update={"verifications": [edited]}))


def test_saved_verification_cannot_be_removed(saved_with_verification):
    store, session, _ = saved_with_verification
    with pytest.raises(AuditTrailError, match="would be removed"):
        store.save(session.model_copy(update={"verifications": []}))


def test_saved_evidence_cannot_be_removed(store_session):
    store, session = store_session
    cid = session.claims[0].claim_id
    store.save(session.model_copy(update={"evidence": [make_evidence(cid)]}))
    with pytest.raises(AuditTrailError, match="evidence"):
        store.save(session)


def test_unreadable_existing_file_is_not_overwritten(store_session):
    store, session = store_session
    path = store.save(session)
    path.write_text("{corrupt")
    with pytest.raises(AuditTrailError, match="move the unreadable file aside"):
        store.save(session)
    assert path.read_text() == "{corrupt"
