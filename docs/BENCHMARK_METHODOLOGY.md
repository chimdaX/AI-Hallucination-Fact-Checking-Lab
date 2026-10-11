# Benchmark methodology

> **Status: under development.** The benchmark currently contains **45 candidate claims
> and 0 human-approved items**. The candidates, their proposed labels and the evidence
> choices were written by an AI coding assistant and are **provisional**. No accuracy
> figure may be reported until items have been reviewed and approved by humans and
> exported as a versioned dataset.

## 1. Objective

Provide a small, carefully documented set of **claim-level factual verification items**
with **traceable evidence** and **human-approved labels**
(SUPPORTED / CONTRADICTED / INSUFFICIENT_EVIDENCE / NOT_CHECKABLE). Automated
verification systems are evaluated against it reproducibly, including a future baseline
in this project.

## 2. Three things that must not be confused

| What | Measured by | What it says | What it does **not** say |
|---|---|---|---|
| Software correctness | `pytest` (unit/integration tests) | The code behaves as specified | Anything about factual accuracy |
| Retrieval quality | `hallab eval retrieval`, `hallab benchmark retrieval` (Recall@k, MRR) | Whether the evidence passage is found among top-ranked passages | Whether the claim is true. Contradicting evidence counts as relevant. |
| Verification accuracy | `hallab benchmark evaluate` against an **approved export** | How often a system's verdicts match human-approved labels on this benchmark version | General LLM hallucination rates, or performance on other data |

Passing tests are never reported as verification accuracy. Retrieval metrics are never
reported as verification accuracy.

## 3. Intended uses

- Evaluate automated claim-verification systems (the planned deterministic baseline,
  model-based verifiers) on a fixed, versioned set.
- Practice and audit human fact-checking: evidence assessment, rationales,
  disagreement handling.
- Demonstrate methodology. With about 45 items, this is a **diagnostic** benchmark, not
  a statistically powerful one.

**Not intended for:** estimating hallucination rates of any model, ranking commercial
models, or any claim about performance outside these items.

## 4. Scope and composition

**Sampling targets** (by *proposed* label; not pre-approved): 15 SUPPORTED, 15
CONTRADICTED, 10 INSUFFICIENT_EVIDENCE, 5 NOT_CHECKABLE. These are the current proposed
counts.

- **Domains:** history 4, geography 4, mathematics 4, physical sciences 8, technology 3,
  general knowledge 8, environmental science 2, public health 6, economics 3,
  quotations 3.
- **Difficulty:** 26 easy, 12 medium, 7 hard (author's estimate).
- **Claim structures covered:** correct assertions, incorrect numbers, incorrect dates,
  fabricated references, unsupported statistics, misattributed quotations, ambiguous
  claims, time-sensitive claims, a compound claim (flagged *split recommended*),
  misleading context (unit confusion), common myths, opinions, predictions, and a
  normative statement.
- **Original context:** 9 items are adapted from the project's synthetic response
  fixtures and keep that response as `original_context`. The rest were written
  directly.
- **No personal data.** Names belong to well-known public figures, and invented names
  (the "Hartwell et al." study, the "Global Ocean Restoration Council") are labelled as
  invented in the item notes.

## 5. Claim selection

Candidates were chosen to (a) cover the verdict categories and claim structures above,
(b) span domains and difficulty, (c) have checkable evidence available under a usable
licence where a factual answer exists, and (d) include deliberate **boundary cases**:

- Einstein "insanity" quote, Marie Antoinette "cake": absence of evidence vs
  contradiction
- the 2040 discovery: INSUFFICIENT_EVIDENCE vs NOT_CHECKABLE
- the Nile: conflicting sources
- the Mars year: vocabulary mismatch between claim and source

**Selection bias:** an AI assistant authored the candidates. They may favour facts that
are well covered online and phrasing that suits language models, and they are likely
present in model training data (see §12).

## 6. Verdict definitions

The definitions are those of `docs/HUMAN_VERIFICATION_GUIDE.md`, applied to the
benchmark's evidence:

- **SUPPORTED:** reliable evidence attached to the item adequately establishes the
  claim. At least one cited passage must be judged SUPPORTING.
- **CONTRADICTED:** reliable evidence establishes that the claim is false or materially
  incorrect. At least one cited passage must be judged CONTRADICTING.
- **INSUFFICIENT_EVIDENCE:** the attached evidence and documented searches do not settle
  the claim: nothing found, weak, ambiguous, or unresolved conflict.
  *Benchmark-specific:* this label is **relative to the documented evidence and
  searches**, which are part of the item, and absence of evidence is never a
  contradiction.
- **NOT_CHECKABLE:** not an objective factual proposition (opinion, prediction,
  normative or too vague). Not for claims that are merely hard to research.

## 7. Evidence standards

**Source priority:** (1) official institutions, (2) primary documents, (3) authoritative
technical documentation, (4) scientific or academic references, (5) high-quality
secondary sources (encyclopedias) when nothing better is available.

**Capture rules** (`scripts/build_benchmark_candidates.py`, `benchmark/capture.py`):

- Passages are captured **verbatim** from the source and stored with SHA-256, locator,
  anchor phrases, access date and capture method.
  - Wikipedia text comes from the MediaWiki API for a **pinned revision** (`oldid`).
  - Web pages are converted with `html-text-v1`.
  - Packaged demo documents are referenced by file and content hash.
- **Licences:** passage text is stored only when the licence allows redistribution
  (public domain or open licence). Otherwise only a **locator and hash** are kept, and
  the reviewer opens the URL. Currently: 4 public-domain passages, 29 open-licence
  passages, and 6 locator-only passages (WHO, Federal Reserve, UN).
- **Access refusals** (HTTP 403 from cdc.gov and toureiffel.paris) were not bypassed.
  They became **research tasks**.
- **Searches for possibly fabricated sources** are stored as **search records** (6
  Crossref queries with result summaries), not as proof of non-existence.
- **No URL, date, author or passage was written from memory.** Every value comes from
  the captured source or its API response.

**Current coverage (honest assessment):**

| | Items |
|---|---|
| Evidence includes an official, primary or technical source | 11 |
| Evidence from encyclopedias only | 23 |
| No passages; documented searches only | 4 |
| No passages, no searches (opinions, prediction, future-dated claim, one research item) | 7 |
| Items with open research tasks | 11 |

**Reproducibility check:** on the capture date, `hallab benchmark check-sources --online`
re-fetched every web page and reproduced all 39 passage hashes (pinned Wikipedia
revisions and local documents verified offline).

## 8. Source reliability assessment

Reviewers rate each passage's **reliability** (high / medium / low / unknown) separately
from its **relation** to the claim (supporting / contradicting / irrelevant /
uncertain). Each reference records its source type, licence and reliability notes; for
example, "tertiary, community-edited source; used as supplementary evidence". The tool
does not decide reliability. A verdict resting only on low-reliability evidence should
normally be INSUFFICIENT_EVIDENCE.

## 9. Annotation procedure

1. Open a READY_FOR_REVIEW item (`hallab benchmark review ITEM --reviewer-id ID` or the
   Benchmark page).
2. **Blind review:** the AI-proposed label is hidden. If you reveal it before reviewing,
   that fact is recorded on your review (`proposed_label_seen_before_review`).
3. Read the claim, its context, temporal scope, any split recommendation, and every
   passage (or open the source for locator-only evidence).
4. Judge each passage's relation and reliability, and add notes.
5. Choose a verdict, cite the passages that justify SUPPORTED or CONTRADICTED, and write
   a rationale. Placeholders are rejected and generic text is flagged.
6. Record uncertainty and flag *research needed* where appropriate. Flagging research
   moves the item to RESEARCH_NEEDED.
7. Confirm the judgement is your own. Reviewer IDs reserved for automated authors
   (`claude`, `ai`, `assistant`, …) are rejected.

Reviews are **write-once**. A correction is a new review with `revision_of` and a
reason, and only the original reviewer can revise their own review.

## 10. Independent review and adjudication

| Review kind | Rule | Meaning |
|---|---|---|
| `first_review` | No current review exists | One person's judgement |
| `independent_second_review` | Different reviewer from existing reviews | Inter-annotator check |
| `repeat_review_same_reviewer` | Same reviewer, at least 24 h later (configurable) | Self-consistency check, **not** independent agreement |

- Disagreeing current reviews put the item in **DISPUTED**. A human **adjudicator**
  records the final verdict and rationale. The original reviews are preserved.
- **Approval** requires a human approver, at least `policy.min_reviews` current reviews,
  agreement or adjudication, and captured evidence for SUPPORTED/CONTRADICTED. The
  approval records the **review level** (`single_review`,
  `repeat_review_same_reviewer`, `independent_double_review`, `adjudicated`), so
  consumers can filter by it.
- The default policy allows single-review approval, which suits a one-person project.
  Set `require_independent_second_review` in `data/benchmark/benchmark.json` for
  stricter versions. Exports state the policy used.
- Changing an approved label requires reopening it (APPROVED → DISPUTED) with a
  documented reason by a human. Earlier approvals and exported versions stay unchanged.

## 11. Integrity and leakage prevention

- **Stable IDs:** item IDs come from the normalised original claim text, evidence IDs
  from source plus passage hash, and reference IDs from URL plus version.
- **Append-only histories:** status events, claim edits, approvals, evidence, and
  research logs can only be extended. Identity fields and proposed labels cannot change.
- **Content fingerprint:** reviews and approvals record the item's content hash. Adding
  evidence after review makes earlier reviews non-current, sending the item back to
  READY_FOR_REVIEW, and approvals must match the current content.
- **Versioned exports:** `approved/benchmark-vX.Y.Z.json` is write-once and carries a
  `dataset_sha256`. Evaluations must name the same version and hash.
- **Partitions:** dev if `int(item_id) % 3 == 0`, otherwise test. The rule is
  deterministic, ignores labels, and gives 16 dev / 29 test. There is no training stage.
  The split separates development (dev: inspect freely) from final evaluation (test).
  **Every test-partition evaluation is logged**, and repeated runs of the same system are
  flagged as a tuning risk.
  - *Known imbalance:* by proposed label, dev contains only 3 CONTRADICTED vs 12 in
    test.
- **Near-duplicates:** token-overlap similarity of at least 0.75 is flagged on creation
  and by `validate`.
- **No contamination claim:** these are well-known facts and myths that are almost
  certainly in language-model training data. The benchmark is **not** claimed to be
  contamination-free.

## 12. Limitations

- **AI-authored candidates and proposed labels:** possible selection and framing bias.
  Human review is the safeguard, and proposed labels are hidden during review.
- **Small size:** 45 items, 29 in test. One item changes test accuracy by about 3.4
  points, and per-class supports are about 2 to 12. Report counts alongside rates.
- **Evidence depth:** 23 items rely on encyclopedia evidence only. Several items have
  open research tasks for primary sources.
- **Single-reviewer reality:** without a second person, labels are single or repeat
  reviews. That is disclosed per item in exports but is weaker than inter-annotator
  agreement.
- **English only; time-sensitive items** are judged against evidence as of the access
  date (2026-10-11).
- **Licence statuses** marked "requires manual confirmation" have not been
  legally reviewed.
- INSUFFICIENT_EVIDENCE is relative to the documented evidence and searches.

## 13. Evaluation metrics

`hallab benchmark evaluate --predictions FILE [--partition dev|test|all]`:

- **N:** approved items in the partition. Each is *answered*, *abstained* (explicit
  `"abstained": true`) or *missing* (no prediction).
- **Coverage** = answered / N. **Abstention rate** and **missing rate** are reported
  separately.
- **Conditional accuracy** = correct / answered. **Strict accuracy** = correct / N, where
  abstentions and missing items count as errors. Coverage and conditional accuracy are
  always reported together.
- **Per class:** support, predicted count, TP, precision (TP / predicted; *undefined*
  when never predicted), recall (TP / support), F1.
  - **Macro-F1** averages classes with support > 0. Classes without support are listed.
- **Confusion matrix:** gold rows × predicted columns plus ABSTAIN and MISSING.
- Evaluation is **refused** when no approved export exists, when the version or hash
  does not match, or when predictions name unknown items.

## 14. Versioning and reproducibility

- `benchmark.json`: name, schema version, approval policy, partition rule, composition
  targets.
- Candidate set: `scripts/build_benchmark_candidates.py --download DIR` re-captures
  sources. The committed files are the canonical snapshot. Re-running never modifies
  existing items, so human reviews are safe.
- Approved versions follow semantic versioning:
  - **MAJOR:** label changes or removed items
  - **MINOR:** added items
  - **PATCH:** metadata fixes that leave labels unchanged
- Every evaluation run is saved under `evaluation/runs/` with the benchmark version,
  dataset hash and predictions hash.
