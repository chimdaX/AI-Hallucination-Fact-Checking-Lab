import json

import pytest

from hallucination_lab.cli import main
from hallucination_lab.examples import load_examples
from hallucination_lab.models import ResponseProvenance
from hallucination_lab.sessions import new_session


def test_example_dataset_validates_and_is_labelled_synthetic():
    dataset = load_examples()
    assert len(dataset.items) >= 5
    for item in dataset.items:
        assert item.response.provenance is ResponseProvenance.SYNTHETIC_FIXTURE
        assert item.response.model_name is None  # fixtures never pose as model outputs
        session = new_session(item.response)
        assert session.claims


def test_example_dataset_rejects_non_synthetic(tmp_path):
    from hallucination_lab.examples import DEFAULT_EXAMPLES_PATH

    data = json.loads(DEFAULT_EXAMPLES_PATH.read_text(encoding="utf-8"))
    data["items"][0]["response"]["provenance"] = "user_submitted"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="synthetic_fixture"):
        load_examples(bad)


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HALLAB_DATA_DIR", str(tmp_path))
    return tmp_path


def test_cli_extract_json(capsys, data_dir):
    assert main(["extract", "--text", "Paris is in France. Is it big?", "--json"]) == 0
    claims = json.loads(capsys.readouterr().out)
    assert [c["requires_verification"] for c in claims] == [True, False]


def test_cli_demo_save_then_list_and_show(capsys, data_dir):
    assert main(["demo", "--save"]) == 0
    files = list((data_dir / "sessions").glob("*.json"))
    assert len(files) == len(load_examples().items)
    capsys.readouterr()
    assert main(["sessions", "list"]) == 0
    listing = capsys.readouterr().out
    assert "[fixture]" in listing
    session_id = listing.split()[0]
    assert main(["sessions", "show", session_id]) == 0


def test_cli_reports_errors_cleanly(capsys, data_dir):
    assert main(["sessions", "show", "../etc"]) == 2
    assert "invalid session id" in capsys.readouterr().err
    assert main(["extract", "--text", "   "]) == 2
