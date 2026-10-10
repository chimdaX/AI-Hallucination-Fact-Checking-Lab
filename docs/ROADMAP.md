# Development roadmap

Each phase ends with a working, tested milestone and a review before the next begins.
Nothing is pushed, tagged, or released without explicit approval.

## ✅ Milestone 1: Foundation + extraction prototype (Phases 1 and 2, initial)

- `pyproject.toml` (hatchling, src layout), Pytest, Ruff
- Pydantic schemas: response, claim, source, evidence, verification, session
- Validation-enforced integrity rules; schema versioning; deterministic IDs
- Atomic validated JSON store; error hierarchy; logging setup
- Rule-based segmentation, compound splitting, non-checkable rules, feature detection
- Human claim editing operations with audit trail
- 6 synthetic fixtures (labelled), `hallab` CLI
- Test suite: IDs, schema rules, storage and recovery, segmentation, extraction,
  editing, fixtures, and CLI

## Phase 2 (remaining)

- Evaluate the extractor on the fixtures and record its failure cases
- Optional, separately documented LLM-assisted extractor interface (no default provider)

## Phase 3: Evidence retrieval

- Reference collection format: a folder of `.txt`/`.md` files plus `collection.json`
  manifest (title, url/reference, licence, publication date, version)
- Ingestion with validation (encoding, size, empty files, duplicate content via SHA-256)
- Passage chunking with character offsets and paragraph index
- BM25 index (`rank-bm25`), configurable `top_k`, `k1`, `b`; reindex command
- Retrieval parameters stored on every `Evidence` record
- A small curated, openly licensed reference collection (original writing or
  public-domain/CC sources, licence recorded per document)
- Tests: ingestion, malformed docs, ranking order on fixtures, provenance, reindex

## Phase 4: Verification

- Human verification API: record verdict, rationale, cited evidence IDs
- Automated baseline for deterministic cases only:
  - numeric comparison between a claim and a curated reference record (with tolerance
    and unit normalisation)
  - exact match against curated fact records
  - anything else returns `INSUFFICIENT_EVIDENCE`, never a guess
- Tests: missing evidence ≠ contradiction, numeric edge cases, human override

## Phase 5: Streamlit UI and reports

- Pages: Dashboard, Analyze Response, Human Verification, Reports, Settings
- Reports: JSON, CSV, Markdown, with citations, limitations, and unchecked claims listed
- Every control wired to real functionality

## Phase 6: Benchmark

- 30–50 claim-level examples across the specified categories
- Each label backed by a cited reference; items needing manual verification marked
- Metrics (per-class counts, confusion matrix, precision/recall/F1) computed only on
  the labelled set, with the evaluation script included

## Phase 7: Testing and reproducibility hardening

- Coverage review against the spec's test list; offline-run check

## Phase 8: Packaging and release

- Dockerfile; GitHub Actions (tests, lint, build) on Linux/Windows/macOS
- Package the example data properly (currently resolved from the source tree)
- Verified installation docs per OS; quick-start guide; versioned release (on approval)
