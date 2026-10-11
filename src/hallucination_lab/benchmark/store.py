"""File storage for the benchmark, with write-once and append-only guarantees.

Layout (default ``data/benchmark/``, override with ``HALLAB_BENCHMARK_DIR``)::

    benchmark.json                       manifest (policy, partition rule, targets)
    candidates/<item_id>.json            one item each; histories append-only
    references/<reference_id>.json       source metadata; write-once
    annotations/<item_id>/<ann_id>.json  human reviews; write-once
    annotations/<item_id>/<adj_id>.json  adjudications; write-once
    approved/benchmark-v<version>.json   approved exports; write-once
    evaluation/runs/<run_id>.json        evaluation results
    evaluation/test_partition_log.jsonl  every evaluation that touched the test split

All files are validated on read and written atomically.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from hallucination_lab.benchmark.models import (
    Adjudication,
    Annotation,
    ApprovedExport,
    BenchmarkItem,
    BenchmarkManifest,
    ReferenceSource,
)
from hallucination_lab.errors import HallucinationLabError
from hallucination_lab.ids import is_valid_id
from hallucination_lab.storage import summarize_validation_error, write_json_atomic

DEFAULT_BENCHMARK_DIR = Path(__file__).resolve().parents[3] / "data" / "benchmark"
M = TypeVar("M", bound=BaseModel)


class BenchmarkError(HallucinationLabError):
    """A benchmark operation is invalid or would break the audit trail."""


def benchmark_dir() -> Path:
    return Path(os.environ.get("HALLAB_BENCHMARK_DIR", DEFAULT_BENCHMARK_DIR))


def _read(path: Path, model: type[M]) -> M:
    try:
        return model.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except FileNotFoundError as exc:
        raise BenchmarkError(f"not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise BenchmarkError(f"{path} is not valid JSON: {exc}") from exc
    except ValidationError as exc:
        raise BenchmarkError(
            f"{path} failed validation: {summarize_validation_error(exc)}"
        ) from exc


def _check_id(value: str) -> str:
    if not is_valid_id(value):
        raise BenchmarkError(f"invalid identifier {value!r}")
    return value


class BenchmarkStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root else benchmark_dir()

    # --- paths -------------------------------------------------------------------

    @property
    def manifest_path(self) -> Path:
        return self.root / "benchmark.json"

    def item_path(self, item_id: str) -> Path:
        return self.root / "candidates" / f"{_check_id(item_id)}.json"

    def reference_path(self, reference_id: str) -> Path:
        return self.root / "references" / f"{_check_id(reference_id)}.json"

    def review_dir(self, item_id: str) -> Path:
        return self.root / "annotations" / _check_id(item_id)

    @property
    def approved_dir(self) -> Path:
        return self.root / "approved"

    @property
    def evaluation_dir(self) -> Path:
        return self.root / "evaluation"

    def exists(self) -> bool:
        return self.manifest_path.is_file()

    # --- manifest ----------------------------------------------------------------

    def load_manifest(self) -> BenchmarkManifest:
        if not self.exists():
            raise BenchmarkError(f"no benchmark at {self.root} (benchmark.json missing)")
        return _read(self.manifest_path, BenchmarkManifest)

    def save_manifest(self, manifest: BenchmarkManifest) -> None:
        write_json_atomic(self.manifest_path, manifest)

    # --- items -------------------------------------------------------------------

    def item_ids(self) -> list[str]:
        folder = self.root / "candidates"
        return sorted(p.stem for p in folder.glob("*.json")) if folder.is_dir() else []

    def load_item(self, item_id: str) -> BenchmarkItem:
        path = self.item_path(item_id)
        if not path.is_file():
            raise BenchmarkError(f"no benchmark item {item_id}")
        return _read(path, BenchmarkItem)

    def load_items(self) -> tuple[list[BenchmarkItem], list[str]]:
        items, problems = [], []
        for item_id in self.item_ids():
            try:
                items.append(self.load_item(item_id))
            except BenchmarkError as exc:
                problems.append(str(exc))
        return items, problems

    def create_item(self, item: BenchmarkItem) -> None:
        path = self.item_path(item.item_id)
        if path.exists():
            raise BenchmarkError(f"item {item.item_id} already exists")
        write_json_atomic(path, BenchmarkItem.model_validate(item.model_dump()))

    def save_item(self, item: BenchmarkItem) -> None:
        """Save an updated item; histories may only be extended, identity never changed."""
        item = BenchmarkItem.model_validate(item.model_dump())
        old = self.load_item(item.item_id)
        for field in ("original_claim_text", "candidate_provenance", "partition", "schema_version"):
            if getattr(old, field) != getattr(item, field):
                raise BenchmarkError(f"refusing to change {field} of {item.item_id}")
        if old.proposed_label is not None and old.proposed_label != item.proposed_label:
            raise BenchmarkError("a proposed label cannot be changed once recorded")
        for field in (
            "status_events",
            "claim_edits",
            "approvals",
            "evidence",
            "research_log",
            "research_tasks",
        ):
            before, after = getattr(old, field), getattr(item, field)
            if field == "research_tasks":  # tasks may be marked resolved, never removed
                before = [t.model_copy(update={"resolved": False}) for t in before]
                after = [t.model_copy(update={"resolved": False}) for t in after[: len(before)]]
                if after != before:
                    raise BenchmarkError(f"refusing to rewrite {field} of {item.item_id}")
                continue
            if after[: len(before)] != before:
                raise BenchmarkError(
                    f"refusing to rewrite {field} of {item.item_id}: history is append-only"
                )
        write_json_atomic(self.item_path(item.item_id), item)

    # --- references --------------------------------------------------------------

    def save_reference(self, ref: ReferenceSource) -> None:
        path = self.reference_path(ref.reference_id)
        if path.exists():
            if _read(path, ReferenceSource) != ref:
                raise BenchmarkError(f"reference {ref.reference_id} exists with other content")
            return
        write_json_atomic(path, ref)

    def load_reference(self, reference_id: str) -> ReferenceSource:
        return _read(self.reference_path(reference_id), ReferenceSource)

    def references(self) -> dict[str, ReferenceSource]:
        folder = self.root / "references"
        if not folder.is_dir():
            return {}
        return {p.stem: _read(p, ReferenceSource) for p in sorted(folder.glob("*.json"))}

    # --- reviews (write-once) ----------------------------------------------------

    def _write_once(self, path: Path, model: BaseModel) -> None:
        if path.exists():
            raise BenchmarkError(f"{path.name} already exists; records are write-once")
        write_json_atomic(path, model)

    def save_annotation(self, annotation: Annotation) -> None:
        path = self.review_dir(annotation.item_id) / f"{annotation.annotation_id}.json"
        self._write_once(path, annotation)

    def save_adjudication(self, adjudication: Adjudication) -> None:
        path = self.review_dir(adjudication.item_id) / f"{adjudication.adjudication_id}.json"
        self._write_once(path, adjudication)

    def annotations(self, item_id: str) -> list[Annotation]:
        folder = self.review_dir(item_id)
        found = [_read(p, Annotation) for p in sorted(folder.glob("ann_*.json"))]
        for a in found:
            if a.item_id != item_id:
                raise BenchmarkError(f"annotation {a.annotation_id} filed under wrong item")
        return sorted(found, key=lambda a: a.created_at)

    def adjudications(self, item_id: str) -> list[Adjudication]:
        folder = self.review_dir(item_id)
        found = [_read(p, Adjudication) for p in sorted(folder.glob("adj_*.json"))]
        return sorted(found, key=lambda a: a.created_at)

    # --- exports -----------------------------------------------------------------

    def export_path(self, version: str) -> Path:
        return self.approved_dir / f"benchmark-v{version}.json"

    def save_export(self, export: ApprovedExport) -> Path:
        path = self.export_path(export.version)
        if path.exists():
            raise BenchmarkError(
                f"version {export.version} already exported; exports are immutable. "
                "Choose a new version number."
            )
        write_json_atomic(path, export)
        return path

    def exports(self) -> list[Path]:
        return (
            sorted(self.approved_dir.glob("benchmark-v*.json"))
            if self.approved_dir.is_dir()
            else []
        )

    @staticmethod
    def load_export(path: Path) -> ApprovedExport:
        return _read(Path(path), ApprovedExport)
