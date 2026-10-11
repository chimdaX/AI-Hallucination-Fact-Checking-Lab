"""Approved exports, versioning, prediction validation and metric calculations."""

import json
from datetime import datetime, timezone

import pytest

from hallucination_lab.benchmark import workflow as wf
from hallucination_lab.benchmark.evaluation import (
    PredictionFile,
    evaluate_predictions,
    load_predictions,
    run_evaluation,
)
from hallucination_lab.benchmark.ids import dataset_hash
from hallucination_lab.benchmark.models import (
    ApprovalPolicy,
    ApprovedExport,
    ExportedItem,
    ReviewStatus,
)
from hallucination_lab.benchmark.store import BenchmarkError

NOW = datetime(2026, 2, 1, tzinfo=timezone.utc)
H = "b" * 64


def _approve_one(bench, text, verdict="CONTRADICTED", relation="CONTRADICTING"):
    item = bench.item(text)
    bench.review(
        item,
        verdict=verdict,
        relation=relation,
        rationale="The passage dates the first bridge to 1874; this decides it.",
    )
    return wf.approve(bench.store, item.item_id, approver_id="alice", confirmed_own_judgement=True)


def test_export_requires_approved_items(bench):
    bench.item("The first bridge across Lake Verenna was completed in 1880.")
    with pytest.raises(BenchmarkError, match="no APPROVED items"):
        wf.export_approved(bench.store, "1.0.0", "alice")


def test_approved_only_export_and_immutable_versions(bench):
    approved = _approve_one(bench, "The first bridge across Lake Verenna was completed in 1880.")
    pending = bench.item(
        "The first Verenna bridge was designed by Mara Quill in 1874.",
        passages=(("Mara Quill", "SUPPORTING"),),
    )
    with pytest.raises(BenchmarkError, match="human reviewer"):
        wf.export_approved(bench.store, "1.0.0", "claude-assistant")
    export, path = wf.export_approved(bench.store, "1.0.0", "alice")
    assert [i.item_id for i in export.items] == [approved.item_id]
    assert pending.item_id not in path.read_text()
    assert export.items[0].review_level.value == "single_review"
    assert bench.store.load_export(path).dataset_sha256 == export.dataset_sha256
    with pytest.raises(BenchmarkError, match="immutable"):
        wf.export_approved(bench.store, "1.0.0", "alice")
    data = json.loads(path.read_text())
    data["items"][0]["label"] = "SUPPORTED"
    path.write_text(json.dumps(data))
    with pytest.raises(BenchmarkError, match="dataset_sha256"):
        bench.store.load_export(path)


def test_provisional_export_is_marked(bench):
    bench.item("The first bridge across Lake Verenna was completed in 1880.", proposed="SUPPORTED")
    data = wf.provisional_export(bench.store)
    assert data["format"] == "hallab-benchmark-provisional-candidates"
    assert "Not ground truth" in data["notice"]
    assert data["items"][0]["approved_label"] is None
    assert data["items"][0]["proposed_label"]["notice"].startswith("proposed label")


# ------------------------------------------------------------------- evaluation


def _export(labels):
    """labels: list of (item suffix, gold label, partition)."""
    items = [
        ExportedItem(
            item_id=f"bmk_{n:016x}",
            claim_text=f"claim {n} text",
            domain="history",
            difficulty="easy",
            claim_types=["incorrect_date"],
            partition=part,
            label=gold,
            review_level="single_review",
            approved_by="alice",
            approved_at=NOW,
            approval_id=f"apr_{n:016x}",
            item_content_sha256=H,
        )
        for n, gold, part in labels
    ]
    return ApprovedExport(
        benchmark_name="b",
        version="1.0.0",
        created_at=NOW,
        created_by="alice",
        policy=ApprovalPolicy(),
        items=items,
        notice="n",
        dataset_sha256=dataset_hash(items),
    )


def _predictions(export, preds, version=None):
    return PredictionFile(
        benchmark_version=version or export.version,
        dataset_sha256=export.dataset_sha256,
        system={"name": "baseline", "version": "0.1"},
        created_at=NOW,
        predictions=[{"item_id": f"bmk_{n:016x}", **p} for n, p in preds],
    )


GOLD = [
    (1, "SUPPORTED", "dev"),
    (2, "SUPPORTED", "dev"),
    (3, "CONTRADICTED", "dev"),
    (4, "CONTRADICTED", "test"),
    (5, "INSUFFICIENT_EVIDENCE", "test"),
]
PREDS = [
    (1, {"verdict": "SUPPORTED"}),
    (2, {"verdict": "CONTRADICTED"}),
    (3, {"verdict": "CONTRADICTED"}),
    (4, {"abstained": True}),
]  # item 5 missing


def test_metrics_hand_computed():
    export = _export(GOLD)
    r = evaluate_predictions(export, _predictions(export, PREDS), "all")
    assert (r.n, r.answered, r.abstained, r.missing, r.correct) == (5, 3, 1, 1, 2)
    assert r.coverage == 0.6 and r.abstention_rate == 0.2 and r.missing_rate == 0.2
    assert r.conditional_accuracy == round(2 / 3, 4) and r.accuracy == 0.4
    s, c, i = (r.per_class[k] for k in ("SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"))
    assert (s.support, s.predicted, s.true_positive, s.precision, s.recall) == (2, 1, 1, 1.0, 0.5)
    assert s.f1 == round(2 * 1 * 0.5 / 1.5, 4)
    assert (c.support, c.predicted, c.true_positive, c.precision, c.recall) == (2, 2, 1, 0.5, 0.5)
    assert (i.support, i.predicted, i.precision, i.recall, i.f1) == (1, 0, None, 0.0, 0.0)
    assert r.per_class["NOT_CHECKABLE"].support == 0
    assert r.classes_without_support == ["NOT_CHECKABLE"]
    assert r.macro_f1 == round((s.f1 + c.f1 + 0.0) / 3, 4)
    assert r.confusion["CONTRADICTED"]["ABSTAIN"] == 1
    assert r.confusion["INSUFFICIENT_EVIDENCE"]["MISSING"] == 1
    assert r.confusion["SUPPORTED"]["CONTRADICTED"] == 1


def test_partition_filter():
    export = _export(GOLD)
    dev = evaluate_predictions(export, _predictions(export, PREDS), "dev")
    assert dev.n == 3 and dev.predictions_outside_partition == 1 and dev.accuracy == round(2 / 3, 4)


def test_prediction_validation():
    export = _export(GOLD)
    with pytest.raises(BenchmarkError, match="same version"):
        evaluate_predictions(export, _predictions(export, PREDS, version="9.9.9"))
    with pytest.raises(BenchmarkError, match="not in the benchmark"):
        evaluate_predictions(export, _predictions(export, [(99, {"verdict": "SUPPORTED"})]))
    with pytest.raises(ValueError, match="duplicate"):
        _predictions(export, [(1, {"verdict": "SUPPORTED"}), (1, {"verdict": "SUPPORTED"})])
    with pytest.raises(ValueError, match="abstained"):
        _predictions(export, [(1, {})])
    with pytest.raises(ValueError):
        _predictions(export, [(1, {"verdict": "SUPPORTED", "abstained": True})])


def test_no_items_in_partition():
    export = _export([(1, "SUPPORTED", "dev")])
    with pytest.raises(BenchmarkError, match="nothing to evaluate"):
        evaluate_predictions(export, _predictions(export, []), "test")


def test_load_predictions_errors(tmp_path):
    bad = tmp_path / "p.json"
    bad.write_text("{oops")
    with pytest.raises(BenchmarkError, match="not valid JSON"):
        load_predictions(bad)
    with pytest.raises(BenchmarkError, match="not found"):
        load_predictions(tmp_path / "missing.json")
    bad.write_text(json.dumps({"format": "hallab-benchmark-predictions"}))
    with pytest.raises(BenchmarkError, match="invalid predictions"):
        load_predictions(bad)


def test_test_partition_use_is_logged(bench):
    export = _export(GOLD)
    preds = _predictions(export, PREDS)
    _, first = run_evaluation(bench.store, export, preds, "test")
    result, second = run_evaluation(bench.store, export, preds, "test")
    assert first["previous_test_runs_for_system"] == 0
    assert second["previous_test_runs_for_system"] == 1
    assert any("tuning to the test set" in n for n in result.notes)
    log = bench.store.evaluation_dir / "test_partition_log.jsonl"
    assert len(log.read_text().splitlines()) == 2
    _, dev = run_evaluation(bench.store, export, preds, "dev")
    assert len(log.read_text().splitlines()) == 2  # dev runs are not test accesses
    assert (bench.store.evaluation_dir / "runs" / f"{dev['run_id']}.json").is_file()


def test_end_to_end_with_real_workflow(bench):
    item = _approve_one(bench, "The first bridge across Lake Verenna was completed in 1880.")
    assert item.status is ReviewStatus.APPROVED
    export, _ = wf.export_approved(bench.store, "0.1.0", "alice")
    preds = PredictionFile(
        benchmark_version="0.1.0",
        dataset_sha256=export.dataset_sha256,
        system={"name": "x", "version": "1"},
        created_at=NOW,
        predictions=[{"item_id": item.item_id, "verdict": "CONTRADICTED"}],
    )
    result = evaluate_predictions(export, preds)
    assert result.accuracy == 1.0 and result.n == 1
    assert any("small sample" in n for n in result.notes)
