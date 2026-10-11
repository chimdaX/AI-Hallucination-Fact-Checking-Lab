# Architecture and design decisions

This document explains *why* the system is built the way it is. Each decision below
can be discussed in an interview.

## Pipeline

```
AI response (stored verbatim, hash-identified)
   │
   ▼
Claim extraction  ── rule-based candidates ──►  Human claim review
   (Phase 2)                                     edit / split / merge / add / remove
   │                                                    │
   ▼                                                    ▼
Evidence retrieval (Phase 3) ── BM25 over curated reference passages
   │        records: source, passage offsets, rank, raw score, parameters
   ▼
Verification (Phase 4)
   ├─ automated baseline: deterministic checks only, always cites evidence
   ├─ (optional, later) model suggestion: labelled as a suggestion
   └─ human verdict: authoritative, requires reviewer_id + reasoning + evidence
   │
   ▼
Report (Phase 5): per-claim findings, citations, limitations, unchecked items
```

Everything for one investigated response lives in a single `VerificationSession` JSON
file.

## Key decisions

### 1. Integrity rules live in the schema, not in the UI

Rules such as "SUPPORTED must cite evidence" and "human verdicts need a reviewer" are
Pydantic validators. They therefore apply everywhere: the CLI, the future Streamlit UI,
tests, and hand-edited JSON files. A UI bug cannot produce an unsupported verdict,
because the model will not validate and the store will not save it.

### 2. Retrieved ≠ verified

An `Evidence` record means "a ranking algorithm thought this passage was relevant". It
says nothing about truth. Evidence only becomes *relied-upon* when a human verification
cites it. Retrieval scores are stored as raw ranking scores (for example BM25), never as
percentages or probabilities.

### 3. No uncalibrated confidence numbers

A number like "87% confident" implies calibration that rule-based or untested systems
don't have. The schema lets `extraction_confidence` be stored only together with a
`confidence_calibration` description. The rule-based extractor leaves both empty.

### 4. Human judgments and automated outputs are structurally distinct

`VerificationMethod` is `human`, `automated_rule`, or `model_suggestion`. Only `human`
may carry a `reviewer_id`, and only human verdicts count in `human_verdict()` and
`pending_review()`. A claim with only an automated verdict therefore still shows as
pending human review.

### 5. Deterministic, content-derived identifiers

IDs are `prefix_` plus the first 16 hex characters of a SHA-256 hash of the defining
content. For a response this is its text. For a claim it is the response ID, span, and
extractor version. Re-running extraction gives identical IDs, which makes runs
reproducible and lets tests assert exact values. The response ID also works as a
tamper check. Sessions use random IDs because two sessions on the same text are
legitimately different investigations. 64 bits of hash makes collisions negligible at
this scale.

### 6. Original wording is never lost

`AIResponse.text` is never stripped or normalised. Each rule-based claim stores its
exact `original_text` and character `span`, and the test suite asserts that
`response.text[span.start:span.end] == original_text`. Reviewer edits only change
`normalized_claim`, and each edit is appended to `reviewer_edits`.

### 7. Soft deletion and an audit trail for claim edits

Removing, splitting, or merging a claim deactivates the old record and links the new
ones through `related_claim_ids`. The path from AI wording to the final verified claim
can always be reconstructed.

### 8. Atomic, validated persistence

Sessions are written to a temporary file and then moved into place with `os.replace`,
so a crash cannot leave a half-written file. Every save re-validates the whole session.
Every load validates against the schema and reports a readable error. Session IDs must
match a strict pattern before they are used as file names, which prevents path
traversal.

### 9. Schema versioning

Each session carries `schema_version` (semver). Loading rejects an unknown major
version instead of guessing. Minor versions can add optional fields without breaking
older files.

### 10. Rule-based extraction first, LLM optional later

Rules are offline, free, deterministic, fast, and explainable: each claim records which
rule fired. Their limits are documented rather than hidden. An LLM extractor could find
more claims, but it would add cost and non-determinism, so it will be an optional module
whose outputs are tagged `llm_assisted`.

### 11. Minimal dependencies

The core, including retrieval, needs only Pydantic. Streamlit will be added in Phase 5.

## Milestone 2 decisions: evidence retrieval

### 12. BM25 implemented directly instead of using `rank-bm25`

BM25 is about 60 lines. Writing it directly:

- adds no dependency (`rank-bm25` pulls in numpy and has not been released since 2022)
- uses the Lucene IDF variant, which never goes negative. The classic formula can
  penalise a passage for matching a very common word.
- returns each query term's contribution, so the CLI can show *why* a passage ranked
  where it did

A unit test checks the score against a hand calculation.

### 13. Content-addressed sources, hash-checked documents

`source_id = hash(file bytes)`. Two consequences:

- the same document gets the same ID everywhere, so duplicates are detected for free
- an edited document gets a *new* ID, so evidence saved in an old session can never be
  silently re-pointed at changed text

Indexing and search refuse to run while a document differs from its registered hash. A
deliberate change requires `--replace`, which is visible and recorded.

### 14. Canonical text and exact offsets

The only normalisations applied to stored text are BOM removal and newline unification.
Case folding, accent folding, and unit rewriting happen only inside the tokenizer. Every
passage is therefore an exact slice of the canonical text, with character offsets and
line numbers a reviewer can open. Tokenizer normalisation affects ranking, never the
evidence text.

### 15. Declared vs computed metadata

Title, author, URL, and dates are declarations that cannot be checked offline, so they
carry a status (`user_supplied`, `captured_at_download`, or `manually_confirmed`).
Hashes, size, and encoding are computed from the file and never accepted from input.
Nothing is ever filled in with guessed values.

### 16. "Searched, nothing found" is recorded explicitly

A `RetrievalRun` stores an outcome for every searched claim: `candidates_found`,
`no_candidates`, or `no_searchable_terms`. Later phases need this to tell
`INSUFFICIENT_EVIDENCE` (looked, found nothing) apart from an unchecked claim. Missing
candidates never become a contradiction.

### 17. Additive, idempotent session updates

Evidence IDs are derived from (claim, passage, passage hash). Re-running retrieval
reuses existing records instead of duplicating or overwriting them. A new run only
appends. Schema 1.1.0 adds only optional fields, and `migrations.py` upgrades 1.0.0
files explicitly. The backward-compatibility test uses a session file generated by the
actual Milestone 1 code.

### 18. Evaluation labels independent of the system under test

Relevance is defined as (file, exact phrase), not as passage IDs, and the labels were
written before the retriever was run. The labels therefore survive re-chunking and are
not biased toward what the retriever happened to return. Defaults were not tuned on the
evaluation set.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Rule-based extraction misses or mis-splits claims | Human claim editing; documented limitations; tests pin known behaviour |
| Users read retrieval relevance as truth | Separate concepts in schema and UI; raw scores only; explicit labels |
| Automated verdicts mistaken for human review | Schema-level separation; pending-review logic ignores automated verdicts |
| Benchmark labels fabricated or unverified | Phase 6 labels require cited sources; fixtures are labelled as fixtures |
| Copyright issues in reference collections | Only original or openly licensed documents, with licence recorded per source |
| Session file corruption | Atomic writes; validation on load; corrupt files reported, not fatal |
| Time-sensitive facts going stale | `temporal_context` captured at extraction; publication dates on evidence |
| Schema evolution breaks saved sessions | `schema_version` with major-version gate |
| Examples, demo collection, and evaluation paths only resolve in a source checkout | Package data properly in Phase 8 (known gap) |
| Reviewers treat a high BM25 score as support | Banner on every result; no score-to-confidence conversion; docs show a contradicting top hit |
| Reference document silently edited | Content-addressed IDs, registration hashes, passage hashes, refusal to index |
| Licence violations in reference text | Only CC BY-SA and U.S. Government sources, attribution file, per-document licence fields |
