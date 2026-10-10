"""Regenerate data/examples/synthetic_responses.json.

Run:  python scripts/build_examples.py

The texts below are hand-written illustrative fixtures. They were NOT produced
by an AI model and must never be presented as real model outputs. Some contain
deliberate errors so the pipeline has something to catch. ``author_notes``
record the author's intent and are not verified ground-truth labels.
"""

from __future__ import annotations

import json
from pathlib import Path

from hallucination_lab.models import SCHEMA_VERSION, AIResponse, ResponseProvenance

FIXTURES = [
    {
        "topic": "Geography: Mount Everest",
        "text": (
            "Mount Everest is the highest mountain above sea level, with an elevation of about "
            "9,848 metres. It sits on the border between Nepal and China. The first confirmed "
            "ascent was made by Edmund Hillary and Tenzing Norgay in 1953."
        ),
        "author_notes": (
            "Deliberate numeric error: the elevation is written as 9,848 m; the 2020 joint "
            "China-Nepal survey figure is 8,848.86 m. Border and 1953 first-ascent sentences "
            "are intended to be correct. To be confirmed against a cited reference in Phase 6."
        ),
    },
    {
        "topic": "Science: water and light",
        "text": (
            "Water boils at 100 °C at standard atmospheric pressure. The chemical formula of "
            "water is H2O, and each molecule contains two hydrogen atoms and one oxygen atom. "
            "Light from the Sun takes about 8 minutes to reach Earth. I think chemistry is the "
            "most fascinating science."
        ),
        "author_notes": (
            "Intended correct scientific statements plus one opinion sentence that should not "
            "be treated as a checkable fact. The second sentence is a compound with two claims."
        ),
    },
    {
        "topic": "History: Eiffel Tower",
        "text": (
            "The Eiffel Tower was completed in 1899 for the World's Fair in Paris. It was "
            "designed and built by the engineering company of Gustave Eiffel. Would you like "
            "to know more about its construction?"
        ),
        "author_notes": (
            "Deliberate date error: 1899 instead of 1889 (Exposition Universelle of 1889). "
            "Second sentence starts with a pronoun and needs context. Final sentence is a "
            "question and is not a claim."
        ),
    },
    {
        "topic": "Fabricated reference and unsupported statistic",
        "text": (
            "According to a 2019 study by Hartwell et al. published in the Journal of Coastal "
            "Futures, 73% of coastal cities will be underwater by 2050. You should consult "
            "local authorities for official flood maps."
        ),
        "author_notes": (
            "The author name, journal title and statistic were invented for this fixture to "
            "illustrate a fabricated citation and an unsupported statistic. Any resemblance to "
            "a real publication is unintended. The second sentence is an instruction."
        ),
    },
    {
        "topic": "Quotation, time-sensitive and hypothetical statements",
        "text": (
            'Albert Einstein once said, "The definition of insanity is doing the same thing '
            'over and over and expecting different results." Currently, the tallest building '
            "in the world is the Burj Khalifa in Dubai. If humans lived on Mars, a year would "
            "last about 687 Earth days."
        ),
        "author_notes": (
            "The quotation is commonly attributed to Einstein without a reliable primary "
            "source; it requires verification against a quotation reference. The Burj Khalifa "
            "claim is time-sensitive. The Mars sentence is framed hypothetically even though "
            "it embeds a checkable fact (Mars orbital period), so a reviewer may rewrite it."
        ),
    },
    {
        "topic": "Mathematics and calendar facts",
        "text": (
            "The square root of 144 is 12. A leap year has 365 days. Pi is approximately "
            "3.14159, and it is an irrational number."
        ),
        "author_notes": (
            "Deliberate error: a leap year has 366 days. The square root and pi statements are "
            "intended to be correct."
        ),
    },
]


def main() -> None:
    items = []
    for fx in FIXTURES:
        response = AIResponse.from_text(
            fx["text"],
            ResponseProvenance.SYNTHETIC_FIXTURE,
            notes="Hand-written illustrative fixture; not an AI model output.",
        )
        items.append(
            {
                "topic": fx["topic"],
                "response": response.model_dump(mode="json", exclude_none=True),
                "author_notes": fx["author_notes"],
            }
        )
    dataset = {
        "dataset": "synthetic_response_fixtures",
        "schema_version": SCHEMA_VERSION,
        "description": (
            "Hand-written illustrative responses for demos and tests. None of these texts "
            "were generated by an AI model. author_notes describe the fixture author's intent "
            "and are not verified labels."
        ),
        "items": items,
    }
    out = Path(__file__).resolve().parents[1] / "data" / "examples" / "synthetic_responses.json"
    out.write_text(json.dumps(dataset, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(items)} fixtures to {out}")


if __name__ == "__main__":
    main()
