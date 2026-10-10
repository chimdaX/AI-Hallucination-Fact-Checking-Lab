# AI Hallucination Detection & Fact-Checking Lab

**Claim-level factual verification for AI-generated text, with evidence you can trace.**

This project breaks an AI response into individual factual claims, retrieves evidence
for each claim from curated reference documents, and records evidence-backed verdicts
(`SUPPORTED`, `CONTRADICTED`, `INSUFFICIENT_EVIDENCE`, `NOT_CHECKABLE`). The output is
a transparent fact-checking report.

The tool does not decide what is true on its own authority. Every verdict must cite
evidence and record how it was produced. Human judgments and automated suggestions are
kept apart at the schema level.

> **Status: pre-alpha (milestone 1 of 8).** The data model, persistence, and a rule-based
> claim-extraction prototype work and are tested. Evidence retrieval, verification, the
> Streamlit UI, reports, and the benchmark have **not been built yet**. See
> [What works today](#what-works-today) and the [roadmap](docs/ROADMAP.md).

---

## Why this project exists

General response-quality evaluation (helpfulness, tone, rubric scores) asks *"is this a
good answer?"*. This lab asks a narrower question: *"which specific statements in this
answer are factually supported, and by what evidence?"* It works at the level of single
claims instead of whole responses.

## What works today

| Capability | Status | Where |
|---|---|---|
| Pydantic schemas for responses, claims, sources, evidence, verifications, sessions | ✅ implemented, tested | `src/hallucination_lab/models.py` |
| Integrity rules enforced by validation (see below) | ✅ implemented, tested | `models.py` |
| Deterministic, content-hashed identifiers | ✅ implemented, tested | `ids.py` |
| Atomic, validated JSON session storage (save / load / list / delete) | ✅ implemented, tested | `storage.py` |
| Rule-based sentence segmentation with exact character offsets | ✅ prototype, tested | `extraction/segmenter.py` |
| Compound-sentence splitting (conservative) | ✅ prototype, tested | `extraction/segmenter.py` |
| Flags for questions, opinions, instructions, hypotheticals, filler | ✅ prototype, tested | `extraction/rules.py` |
| Detection of numbers, dates, measurements, entities, citations, quotes, time-sensitive wording | ✅ prototype (heuristic), tested | `extraction/features.py` |
| Manual claim edit / split / merge / add / remove / restore with audit trail | ✅ implemented, tested | `extraction/editing.py` |
| Synthetic example responses (clearly labelled, not real model outputs) | ✅ 6 fixtures | `data/examples/` |
| Command-line interface (`hallab`) | ✅ implemented, tested | `cli.py` |
| Evidence retrieval (BM25) | ⏳ Phase 3 | — |
| Automated baseline verification and human verification flow | ⏳ Phase 4 | — |
| Streamlit interface | ⏳ Phase 5 | — |
| Reports (JSON / CSV / Markdown) | ⏳ Phase 5 | — |
| Labelled benchmark and metrics | ⏳ Phase 6 | — |
| Docker, CI, release | ⏳ Phase 8 | — |

## Integrity rules enforced in code

These rules are checked by schema validation, so invalid records cannot be created or
saved. Each one has a test.

- A `SUPPORTED` verdict must cite supporting evidence, and a `CONTRADICTED` verdict must
  cite contradicting evidence. Missing evidence can never turn into a contradiction.
- A human verification must carry a `reviewer_id`. Automated or model verifications
  must not carry one, so they can never pass as human review.
- A model-generated suggestion must name the model in its provenance.
- `extraction_confidence` can only be stored together with a description of how it was
  calibrated. The rule-based extractor stores no confidence at all.
- An AI response's ID is a SHA-256 hash of its exact text. If a stored response is
  altered, loading it fails.
- A session is referentially consistent. Evidence and verdicts can only point at claims
  and evidence in the same session, and a verdict cannot cite evidence that was
  retrieved for a different claim.
- Evidence counts as retrieved until a human verdict cites it. `human_cited_evidence_ids()`
  tracks the difference.
- Removed, split, and merged claims are deactivated, never deleted. Every change is
  logged in `reviewer_edits`.

## Quick start

Requirements: Python 3.10 or newer and git. No API keys, no paid services, no network
access at runtime.

```bash
git clone <your-fork-url> ai-hallucination-verification-lab
cd ai-hallucination-verification-lab

# Option A: uv (fast)
uv venv .venv
uv pip install -e ".[dev]"

# Option B: standard library venv + pip
python -m venv .venv
# Linux/macOS:   source .venv/bin/activate
# Windows (PowerShell):   .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

Try it:

```bash
# Extract candidate claims from text
hallab extract --text "The Eiffel Tower was completed in 1889. Is it in Paris? I think it is beautiful."

# Same, as JSON (shows spans, features, and which rule fired)
hallab extract --file my_response.txt --json

# Run extraction over the bundled synthetic fixtures and save them as sessions
hallab demo --save
hallab sessions list
hallab sessions show <session_id>
```

Sessions are saved as JSON under `./lab_data/sessions/`. Set `HALLAB_DATA_DIR` to change
the location and `HALLAB_LOG_LEVEL=INFO` for more logging.

Example output (from `hallab demo`):

```
== Mathematics and calendar facts (4 candidates, 4 to check)
  1. [CHECK] The square root of 144 is 12.
       numbers: 144, 12
  2. [CHECK] A leap year has 365 days.
       numbers: 365 | measurements: 365 days
  3. [CHECK] Pi is approximately 3.14159
       numbers: 3.14159 | entities: Pi
  4. [CHECK] It is an irrational number.
```

Extraction only produces *candidates*. Claim 4 starts with a pronoun, and the claim
record notes that it needs context. Claim 2 is wrong, but deciding that is the job of
the verification phases, not extraction.

## Development

```bash
pytest            # run the test suite
ruff check .      # lint
ruff format .     # format
```

Platform note: development and testing so far have been on Linux (WSL2, Python 3.12).
Windows and macOS instructions above use standard Python tooling but **have not yet been
verified**. Cross-platform CI is planned for Phase 8.

## How claim extraction works (and its limits)

1. **Segmentation.** Text is split into sentences with transparent rules: a fixed
   abbreviation list ("Dr.", "e.g.", "U.S."), decimal handling, bullet and heading
   detection, and soft-wrapped line joining. Each segment keeps exact character offsets
   into the original response.
2. **Compound splitting.** A sentence is split at `;` or `, and / but / while / whereas`
   only when both sides look like full clauses (at least three words and a verb).
3. **Classification.** Ordered rules flag questions, filler, instructions,
   hypotheticals, and opinion markers. The rule that fired is stored in
   `extraction_notes`.
4. **Features.** Regexes detect numbers, dates, measurements, capitalised-name runs,
   citation patterns, quotations, and time-sensitive wording.

**This is sentence segmentation, not true factual claim extraction.** Known limitations:

- A sentence may contain several facts that are not split, for example "X, born in 1879
  in Ulm, developed relativity".
- Claims that start with a pronoun ("It opened in 1889") need context. They are flagged
  but not resolved.
- "I think Paris is the capital of France" is flagged as opinion even though it contains a
  checkable fact. Reviewers can override this.
- Entity detection is just runs of capitalised words, and spelled-out numbers are not
  detected.
- The rules are tuned for English.

The design response to these limits is human review. Reviewers can edit, split, merge,
add, and remove claims before verification, and every change is recorded. An optional
LLM-assisted extractor may be added later as a separate, clearly labelled module.

## Project layout

```
src/hallucination_lab/
  models.py          Pydantic schemas + integrity rules
  ids.py             deterministic content-hashed IDs
  storage.py         atomic, validated JSON session store
  sessions.py        session creation
  examples.py        loader for the synthetic fixtures
  cli.py             `hallab` command-line interface
  config.py          settings from environment variables
  logging_utils.py   logging setup
  errors.py          exception hierarchy
  extraction/
    segmenter.py     sentence + clause segmentation with offsets
    rules.py         non-checkable rules, normalisation
    features.py      numbers/dates/entities/citations/... detection
    extractor.py     pipeline producing candidate Claim records
    editing.py       human edit/split/merge/add/remove operations
data/examples/       synthetic fixtures (hand-written, labelled as such)
scripts/             dataset build scripts
tests/               pytest suite
docs/                architecture notes and roadmap
```

More detail: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) ·
[docs/ROADMAP.md](docs/ROADMAP.md)

## Data and content policy

- `data/examples/synthetic_responses.json` contains **hand-written fixtures**. None of
  them were produced by an AI model. Some contain deliberate errors. Their
  `author_notes` describe intent and are **not** verified ground-truth labels.
- Reference documents (Phase 3) will be original or openly licensed. Each will be
  recorded with its licence. No copyrighted articles and no private data.

## Licence

MIT. See [LICENSE](LICENSE).
