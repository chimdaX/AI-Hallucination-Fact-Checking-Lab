"""Streamlit interface tests using streamlit.testing.v1.AppTest (no browser needed).

Each page is rendered against a temporary data folder; workflows are driven
through the real widgets, so these tests exercise UI + services + engines.
"""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from hallucination_lab.services.workspace import Workspace
from hallucination_lab.ui.page_collections import safe_filename
from tests.conftest import evidence_containing

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
TIMEOUT = 60


def page(name: str) -> AppTest:
    return AppTest.from_string(
        f"from hallucination_lab.ui import {name}\n{name}.render()\n", default_timeout=TIMEOUT
    )


def button(at: AppTest, label: str):
    matches = [b for b in at.button if b.label == label]
    assert matches, f"no button {label!r}; have {[b.label for b in at.button]}"
    return matches[0]


def no_errors(at: AppTest) -> None:
    assert not at.exception, [e.message for e in at.exception]


@pytest.fixture
def data_dir(settings, monkeypatch):
    monkeypatch.setenv("HALLAB_DATA_DIR", str(settings.data_dir))
    return settings.data_dir


@pytest.fixture
def env(verification_env, data_dir):
    return verification_env


# ----------------------------------------------------------------------- startup


def test_full_app_starts_with_empty_data(data_dir):
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT).run()
    no_errors(at)
    assert at.title[0].value == "Dashboard"
    assert any("No sessions yet" in i.value for i in at.info)


@pytest.mark.parametrize(
    "name",
    [
        "page_dashboard",
        "page_analyze",
        "page_verify",
        "page_reports",
        "page_collections",
        "page_guide",
    ],
)
def test_every_page_renders_without_data(data_dir, name):
    at = page(name).run()
    no_errors(at)
    assert at.title


@pytest.mark.parametrize("name", ["page_dashboard", "page_analyze", "page_verify", "page_reports"])
def test_every_page_renders_with_a_session(env, name):
    at = page(name)
    at.session_state["current_session_id"] = env["session_id"]
    at.session_state["reviewer_id"] = "r1"
    no_errors(at.run())


def test_dashboard_numbers_come_from_files(env):
    at = page("page_dashboard").run()
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Sessions"] == "1" and metrics["Candidate claims"] == "3"
    assert metrics["Human-verified claims"] == "0" and metrics["Pending review"] == "3"


def test_corrupted_session_file_is_reported_not_fatal(env, data_dir):
    (data_dir / "sessions" / "ses_ffffffffffffffff.json").write_text("{broken")
    at = page("page_dashboard").run()
    no_errors(at)
    assert any("Unreadable session file" in w.value for w in at.warning)


def test_invalid_session_in_state_is_handled(data_dir):
    at = page("page_verify")
    at.session_state["current_session_id"] = "ses_0000000000000000"
    at.run()
    no_errors(at)
    assert any("Could not open session" in e.value for e in at.error)


# --------------------------------------------------------------- analyze workflow


def test_extract_then_retrieve(data_dir, test_collection):
    at = page("page_analyze").run()
    at.text_area[0].input("The Osterland Railway is 48 km long. Do you like trains?")
    button(at, "Extract claims").click().run()
    no_errors(at)
    sid = at.session_state["current_session_id"]
    assert sid and any("Saved session" in s.value for s in at.success)
    assert any("rule-based" in i.value for i in at.info)
    session = Workspace.from_env().load(sid).session
    assert session.response.text == "The Osterland Railway is 48 km long. Do you like trains?"
    assert [c.requires_verification for c in session.claims] == [True, False]

    collection_box = next(s for s in at.selectbox if s.label == "Reference collection")
    collection_box.set_value("testrefs")
    button(at, "Retrieve evidence").click().run()
    no_errors(at)
    assert any("Searched 1 claim(s)" in s.value for s in at.success)
    assert Workspace.from_env().load(sid).session.evidence


def test_empty_submission_shows_error(data_dir):
    at = page("page_analyze").run()
    button(at, "Extract claims").click().run()
    assert any("empty" in e.value for e in at.error)


def test_repeated_extraction_does_not_duplicate(data_dir):
    at = page("page_analyze").run()
    at.text_area[0].input("Brindle sits on the northern shore of Lake Verenna.")
    button(at, "Extract claims").click().run()
    button(at, "Extract claims").click().run()  # same text again
    assert any("already created session" in w.value for w in at.warning)
    assert len(Workspace.from_env().list_sessions()[0]) == 1


def test_stale_view_blocks_edit(env):
    at = page("page_analyze")
    at.session_state["current_session_id"] = env["session_id"]
    at.session_state["reviewer_id"] = "r1"
    at.run()
    ws = Workspace.from_env()  # another "tab" changes the session meanwhile
    ws.add_claim(ws.load(env["session_id"]), "Brindle has a lighthouse.", "someone-else")
    form_input = next(t for t in at.text_input if t.label == "Add a claim the extractor missed")
    form_input.input("A claim typed in the old view.")
    button(at, "Add claim").click().run()
    assert any("changed since the page was displayed" in w.value for w in at.warning)
    claims = Workspace.from_env().load(env["session_id"]).session.claims
    assert not any(c.normalized_claim == "A claim typed in the old view." for c in claims)


# ----------------------------------------------------------- verification workflow


def _verify_page(env, reviewer="r1") -> AppTest:
    at = page("page_verify")
    at.session_state["current_session_id"] = env["session_id"]
    at.session_state["reviewer_id"] = reviewer
    at.run()
    claim_box = next(s for s in at.selectbox if s.label == "Claim")
    session = Workspace.from_env().load(env["session_id"]).session
    index = [c.claim_id for c in session.active_claims].index(env["bridge"])
    claim_box.set_value(index).run()
    return at


def _fill_bridge(at: AppTest, env, prefix: str, rationale: str):
    session = Workspace.from_env().load(env["session_id"]).session
    eid = evidence_containing(session, env["bridge"], "completed in 1874")
    for other in session.latest_retrieval(env["bridge"]).evidence_ids:
        at.radio(key=f"lbl_{prefix}_{other}").set_value(
            "CONTRADICTING" if other == eid else "IRRELEVANT"
        )
    at.selectbox(key=f"verdict_{prefix}").set_value("CONTRADICTED")
    at.multiselect(key=f"cited_{prefix}").set_value([eid])
    at.text_area(key=f"rat_{prefix}").input(rationale)
    return eid


def test_score_alone_cannot_produce_supported(env):
    at = _verify_page(env)
    prefix = f"{env['bridge']}:r1:new"
    session = Workspace.from_env().load(env["session_id"]).session
    top = session.latest_retrieval(env["bridge"]).evidence_ids[0]  # highest BM25, not assessed
    at.selectbox(key=f"verdict_{prefix}").set_value("SUPPORTED")
    at.multiselect(key=f"cited_{prefix}").set_value([top])
    at.text_area(key=f"rat_{prefix}").input("The top passage mentions the bridge and 1874.")
    button(at, "Check and review").click().run()
    no_errors(at)
    assert any("Cannot record this verification yet" in e.value for e in at.error)
    saved = Workspace.from_env().load(env["session_id"]).session
    assert saved.verifications == [] and saved.get_draft(env["bridge"], "r1")  # draft kept


def test_full_verification_requires_confirmation(env):
    at = _verify_page(env)
    prefix = f"{env['bridge']}:r1:new"
    _fill_bridge(at, env, prefix, "The passage dates the first bridge to 1874, not 1880.")
    button(at, "Check and review").click().run()
    no_errors(at)
    assert Workspace.from_env().load(env["session_id"]).session.verifications == []
    save = button(at, "Save verification")
    assert save.disabled  # not before explicit confirmation
    at.checkbox(key=f"confirm_{prefix}").check().run()
    button(at, "Save verification").click().run()
    no_errors(at)
    session = Workspace.from_env().load(env["session_id"]).session
    assert len(session.verifications) == 1
    assert session.human_verdict(env["bridge"]).reviewer_id == "r1"
    assert session.verification_drafts == []


def test_draft_saved_and_recovered(env):
    at = _verify_page(env)
    prefix = f"{env['bridge']}:r1:new"
    session = Workspace.from_env().load(env["session_id"]).session
    eid = evidence_containing(session, env["bridge"], "completed in 1874")
    at.radio(key=f"lbl_{prefix}_{eid}").set_value("CONTRADICTING")
    at.text_area(key=f"rat_{prefix}").input("Half-written thought about 1874")
    button(at, "Save draft").click().run()
    assert any("Draft saved" in s.value for s in at.success)

    fresh = _verify_page(env)  # new browser session: state comes from the file
    assert any("Resuming your saved draft" in i.value for i in fresh.info)
    assert fresh.radio(key=f"lbl_{prefix}_{eid}").value == "CONTRADICTING"
    assert fresh.text_area(key=f"rat_{prefix}").value == "Half-written thought about 1874"


def test_rationale_warning_requires_acknowledgement(env):
    at = _verify_page(env)
    prefix = f"{env['bridge']}:r1:new"
    _fill_bridge(at, env, prefix, "The AI is wrong.")
    button(at, "Check and review").click().run()
    assert any("Rationale warning" in w.value for w in at.warning)
    at.checkbox(key=f"confirm_{prefix}").check().run()
    assert button(at, "Save verification").disabled
    at.checkbox(key=f"ack_{prefix}").check().run()
    assert not button(at, "Save verification").disabled


def test_placeholder_rationale_rejected(env):
    at = _verify_page(env)
    prefix = f"{env['bridge']}:r1:new"
    _fill_bridge(at, env, prefix, "n/a")
    button(at, "Check and review").click().run()
    assert any("placeholder" in e.value for e in at.error)


def test_revision_via_ui_keeps_history(env):
    at = _verify_page(env)
    prefix = f"{env['bridge']}:r1:new"
    _fill_bridge(at, env, prefix, "The passage dates the first bridge to 1874, not 1880.")
    button(at, "Check and review").click().run()
    at.checkbox(key=f"confirm_{prefix}").check().run()
    button(at, "Save verification").click().run()
    first = Workspace.from_env().load(env["session_id"]).session.verifications[0]

    at = _verify_page(env)
    button(at, "Correct this verdict (creates a new revision)").click().run()
    rprefix = f"{env['bridge']}:r1:{first.verification_id}"
    reason = next(t for t in at.text_area if t.label == "Correction reason (required)")
    reason.input("Clarified that the passage refers to the first bridge.")
    at.text_area(key=f"rat_{rprefix}").input(
        "Passage: first bridge completed in 1874; the claim's 1880 is wrong."
    )
    at.selectbox(key=f"verdict_{rprefix}").set_value("CONTRADICTED")
    at.multiselect(key=f"cited_{rprefix}").set_value(first.contradicting_evidence_ids)
    button(at, "Check and review").click().run()
    no_errors(at)
    at.checkbox(key=f"confirm_{rprefix}").check().run()
    button(at, "Save verification").click().run()
    no_errors(at)
    session = Workspace.from_env().load(env["session_id"]).session
    assert len(session.verifications) == 2
    assert session.get_verification(first.verification_id) == first
    assert session.human_verdict(env["bridge"]).revision_of == first.verification_id


def test_source_integrity_warning_shown(env, catalog):
    doc = catalog.documents_dir("testrefs") / "lake_verenna.md"
    doc.write_text(doc.read_text().replace("1874", "1875"))
    at = _verify_page(env)
    no_errors(at)
    assert any(
        "cannot be cited" in e.value.lower() or "Cannot be cited" in e.value for e in at.error
    )
    coll = page("page_collections").run()
    next(s for s in coll.selectbox if s.label == "Collection").set_value("testrefs").run()
    assert any("modified since it was registered" in e.value for e in coll.error)


def test_reports_page_offers_three_downloads(env):
    at = page("page_reports")
    at.session_state["current_session_id"] = env["session_id"]
    at.run()
    no_errors(at)
    assert any("Incomplete" in w.value for w in at.warning)
    labels = [b.proto.label for b in at.get("download_button")]
    assert labels == ["Download JSON", "Download CSV", "Download MD"]


# ------------------------------------------------------------------- file names


@pytest.mark.parametrize(
    "name,expected",
    [
        ("notes.md", "notes.md"),
        ("My Notes (v2).TXT", "My_Notes_v2.txt"),
        ("C:\\Users\\me\\Desktop\\report.md", "report.md"),
        ("/home/me/../../etc/passwd.txt", "passwd.txt"),
        ("..md", "document.md"),
        ("résumé.md", "r_sum.md"),
    ],
)
def test_safe_filename_is_portable(name, expected):
    assert safe_filename(name) == expected
