"""Loader for the bundled synthetic example responses.

These are hand-written illustrative fixtures, *not* outputs captured from any
AI model. The ``author_notes`` describe what the fixture author intended to
include; they are not verified ground-truth labels. A benchmark with verified
labels and cited sources is a separate deliverable (Phase 6).
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import Field, model_validator

from hallucination_lab.models import AIResponse, NonBlankStr, ResponseProvenance, StrictModel

DEFAULT_EXAMPLES_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "examples" / "synthetic_responses.json"
)


class ExampleResponse(StrictModel):
    topic: NonBlankStr
    response: AIResponse
    author_notes: NonBlankStr


class ExampleDataset(StrictModel):
    dataset: NonBlankStr
    schema_version: NonBlankStr
    description: NonBlankStr
    items: list[ExampleResponse] = Field(min_length=1)

    @model_validator(mode="after")
    def _all_synthetic(self) -> ExampleDataset:
        for item in self.items:
            if item.response.provenance is not ResponseProvenance.SYNTHETIC_FIXTURE:
                raise ValueError(
                    f"example {item.response.response_id} must be labelled synthetic_fixture"
                )
        return self


def load_examples(path: Path = DEFAULT_EXAMPLES_PATH) -> ExampleDataset:
    return ExampleDataset.model_validate(json.loads(path.read_text(encoding="utf-8")))
