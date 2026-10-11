"""Evaluate automated verdict predictions against an approved benchmark export.

Rules:

* Only an **approved export** (human-approved labels, fixed version and
  ``dataset_sha256``) can be evaluated against. Predictions must name the same
  version and hash, otherwise evaluation is refused.
* Every approved item in the chosen partition counts. Each is one of:
  answered (a verdict), **abstained** (the system explicitly declined) or
  **missing** (no prediction submitted).
* ``coverage = answered / N``. ``conditional_accuracy = correct / answered``.
  ``accuracy = correct / N`` (strict: abstentions and missing items count as
  errors). Coverage and conditional accuracy are always reported together.
* Per class: ``precision = TP / predicted_as_class`` (undefined when the class
  was never predicted), ``recall = TP / support`` (support = gold items of that
  class in the partition, so abstentions lower recall), and F1. Macro-F1
  averages classes with support > 0. Classes without support are listed, not
  averaged.
* Evaluations touching the held-out ``test`` partition are appended to a log.
  Repeated test evaluations of the same system are reported, because tuning on
  the test set invalidates it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from hallucination_lab.benchmark.ids import EVAL_RUN
from hallucination_lab.benchmark.models import ApprovedExport, Partition
from hallucination_lab.benchmark.store import BenchmarkError, BenchmarkStore
from hallucination_lab.ids import content_sha256, make_id
from hallucination_lab.models import (
    AwareDatetime,
    Identifier,
    NonBlankStr,
    Sha256Hex,
    StrictModel,
    Verdict,
    utc_now,
)

ABSTAIN = "ABSTAIN"
MISSING = "MISSING"
PREDICTIONS_FORMAT = "hallab-benchmark-predictions"


class SystemInfo(StrictModel):
    name: NonBlankStr
    version: NonBlankStr
    description: str | None = None


class Prediction(StrictModel):
    item_id: Identifier
    verdict: Verdict | None = None
    abstained: bool = False

    @model_validator(mode="after")
    def _one(self) -> Prediction:
        if (self.verdict is None) != self.abstained:
            raise ValueError("give a verdict, or set abstained: true with no verdict")
        return self


class PredictionFile(StrictModel):
    format: Literal["hallab-benchmark-predictions"] = PREDICTIONS_FORMAT
    benchmark_version: NonBlankStr
    dataset_sha256: Sha256Hex
    system: SystemInfo
    created_at: AwareDatetime
    predictions: list[Prediction] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique(self) -> PredictionFile:
        ids = [p.item_id for p in self.predictions]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate item_id in predictions")
        return self


@dataclass
class ClassMetrics:
    support: int
    predicted: int
    true_positive: int
    precision: float | None
    recall: float | None
    f1: float | None


@dataclass
class EvaluationResult:
    benchmark_version: str
    dataset_sha256: str
    partition: str
    system: dict
    n: int
    answered: int
    abstained: int
    missing: int
    correct: int
    coverage: float
    abstention_rate: float
    missing_rate: float
    conditional_accuracy: float | None
    accuracy: float
    per_class: dict[str, ClassMetrics]
    macro_f1: float | None
    classes_without_support: list[str]
    confusion: dict[str, dict[str, int]]
    predictions_outside_partition: int
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _ratio(a: int, b: int) -> float | None:
    return round(a / b, 4) if b else None


def evaluate_predictions(
    export: ApprovedExport,
    predictions: PredictionFile,
    partition: Literal["dev", "test", "all"] = "all",
) -> EvaluationResult:
    if predictions.benchmark_version != export.version or (
        predictions.dataset_sha256 != export.dataset_sha256
    ):
        raise BenchmarkError(
            f"predictions target benchmark {predictions.benchmark_version} "
            f"({predictions.dataset_sha256[:12]}...) but the export is {export.version} "
            f"({export.dataset_sha256[:12]}...). Evaluate against the same version."
        )
    gold_all = {i.item_id: i for i in export.items}
    unknown = [p.item_id for p in predictions.predictions if p.item_id not in gold_all]
    if unknown:
        raise BenchmarkError(f"predictions for items not in the benchmark: {unknown[:5]}")
    gold = {
        iid: item
        for iid, item in gold_all.items()
        if partition == "all" or item.partition is Partition(partition)
    }
    if not gold:
        raise BenchmarkError(f"no approved items in partition {partition!r}: nothing to evaluate")
    by_id = {p.item_id: p for p in predictions.predictions}
    outside = sum(1 for p in predictions.predictions if p.item_id not in gold)

    labels = [v.value for v in Verdict]
    confusion = {g: {c: 0 for c in [*labels, ABSTAIN, MISSING]} for g in labels}
    answered = abstained = missing = correct = 0
    for iid, item in gold.items():
        p = by_id.get(iid)
        if p is None:
            column = MISSING
            missing += 1
        elif p.abstained:
            column = ABSTAIN
            abstained += 1
        else:
            column = p.verdict.value
            answered += 1
            correct += column == item.label.value
        confusion[item.label.value][column] += 1

    per_class: dict[str, ClassMetrics] = {}
    for label in labels:
        support = sum(confusion[label].values())
        predicted = sum(confusion[g][label] for g in labels)
        tp = confusion[label][label]
        precision = _ratio(tp, predicted)
        recall = _ratio(tp, support)
        if precision is None or recall is None:
            f1 = None if support == 0 else 0.0
        else:
            f1 = round(2 * precision * recall / (precision + recall), 4) if tp else 0.0
        per_class[label] = ClassMetrics(support, predicted, tp, precision, recall, f1)
    supported = [m.f1 for m in per_class.values() if m.support > 0]
    n = len(gold)
    notes = [
        "accuracy counts abstentions and missing predictions as errors; "
        "conditional_accuracy covers answered items only",
        "metrics describe this benchmark version and partition only",
    ]
    if n < 30:
        notes.append(f"small sample (n={n}): differences of a few items change metrics a lot")
    return EvaluationResult(
        benchmark_version=export.version,
        dataset_sha256=export.dataset_sha256,
        partition=partition,
        system=predictions.system.model_dump(),
        n=n,
        answered=answered,
        abstained=abstained,
        missing=missing,
        correct=correct,
        coverage=round(answered / n, 4),
        abstention_rate=round(abstained / n, 4),
        missing_rate=round(missing / n, 4),
        conditional_accuracy=_ratio(correct, answered),
        accuracy=round(correct / n, 4),
        per_class=per_class,
        macro_f1=round(sum(supported) / len(supported), 4) if supported else None,
        classes_without_support=[k for k, m in per_class.items() if m.support == 0],
        confusion=confusion,
        predictions_outside_partition=outside,
        notes=notes,
    )


def run_evaluation(
    store: BenchmarkStore,
    export: ApprovedExport,
    predictions: PredictionFile,
    partition: Literal["dev", "test", "all"] = "all",
    now: datetime | None = None,
) -> tuple[EvaluationResult, dict]:
    """Evaluate, save the run record, and log any use of the held-out test partition."""
    now = now or utc_now()
    result = evaluate_predictions(export, predictions, partition)
    predictions_sha = content_sha256(predictions.model_dump_json())
    run_id = make_id(EVAL_RUN, export.dataset_sha256, predictions_sha, partition, now.isoformat())
    record = {
        "run_id": run_id,
        "evaluated_at": now.isoformat(),
        "predictions_sha256": predictions_sha,
        **result.to_dict(),
    }
    runs = store.evaluation_dir / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    (runs / f"{run_id}.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    info: dict = {"run_id": run_id, "previous_test_runs_for_system": 0}
    if partition in {"test", "all"}:
        log = store.evaluation_dir / "test_partition_log.jsonl"
        previous = []
        if log.is_file():
            previous = [
                json.loads(line)
                for line in log.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        same = [
            r
            for r in previous
            if r["system"] == predictions.system.name
            and r["dataset_sha256"] == export.dataset_sha256
        ]
        info["previous_test_runs_for_system"] = len(same)
        with log.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "run_id": run_id,
                        "at": now.isoformat(),
                        "system": predictions.system.name,
                        "system_version": predictions.system.version,
                        "partition": partition,
                        "benchmark_version": export.version,
                        "dataset_sha256": export.dataset_sha256,
                        "predictions_sha256": predictions_sha,
                    }
                )
                + "\n"
            )
        if same:
            result.notes.append(
                f"this system was already evaluated on the test partition {len(same)} time(s); "
                "repeated test evaluation risks tuning to the test set. Use the dev partition "
                "for development."
            )
    return result, info


def load_predictions(path) -> PredictionFile:
    from pathlib import Path

    from pydantic import ValidationError

    from hallucination_lab.storage import summarize_validation_error

    try:
        return PredictionFile.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
    except FileNotFoundError as exc:
        raise BenchmarkError(f"predictions file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise BenchmarkError(f"predictions file is not valid JSON: {exc}") from exc
    except ValidationError as exc:
        raise BenchmarkError(
            f"invalid predictions file: {summarize_validation_error(exc)}"
        ) from exc
