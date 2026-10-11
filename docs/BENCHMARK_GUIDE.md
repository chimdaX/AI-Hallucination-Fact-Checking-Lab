# Benchmark user guide

How to maintain and use the ground-truth benchmark. Read
[BENCHMARK_METHODOLOGY.md](BENCHMARK_METHODOLOGY.md) for the reasoning behind each rule.

> The benchmark is **under development**: 45 AI-authored candidates, **0 approved**.
> Nothing here is ground truth until you review and approve it.

Data lives in `data/benchmark/`. To work on a copy, set `HALLAB_BENCHMARK_DIR`. Analysis
sessions (`lab_data/`) are separate and unaffected.

## Quick reference

```bash
hallab benchmark status                          # counts; approved labels; pending work
hallab benchmark list --status READY_FOR_REVIEW  # filters: --domain --difficulty --partition
hallab benchmark show ITEM_ID                    # claim, context, evidence, reviews (proposed label hidden)
hallab benchmark review ITEM_ID --reviewer-id YOU          # interactive, blind review
hallab benchmark review ITEM_ID --reviewer-id YOU --revise ANN_ID
hallab benchmark adjudicate ITEM_ID --adjudicator-id YOU   # for DISPUTED items
hallab benchmark approve ITEM_ID --approver-id YOU --confirm-own-judgement
hallab benchmark mark ITEM_ID --status RESEARCH_NEEDED --actor YOU --reason "..."
hallab benchmark validate                        # schema, references, labels, near-duplicates
hallab benchmark check-sources [--online]        # changed or missing evidence sources
hallab benchmark export --version 1.0.0 --created-by YOU    # approved items only, immutable
hallab benchmark export --provisional --output candidates.json   # NOT ground truth
hallab benchmark evaluate --predictions preds.json --partition dev
hallab benchmark retrieval [--approved-only]     # retrieval metrics, separate from verdicts
```

In the web interface, the **Benchmark** page does the same: filter candidates, inspect
sources, review blind, adjudicate, approve, and export or download.

## Adding candidates

```bash
hallab benchmark add --claim "The Danube flows through ten countries." \
  --domain geography --difficulty medium --type correct_assertion \
  --author-id YOU --method "from a news article, 2026-10-12"
```

- Exact duplicates are refused (case, accent and whitespace-insensitive), and
  near-duplicates are warned about.
- New candidates start as CANDIDATE. Attach evidence before marking them
  READY_FOR_REVIEW.
- Large batches belong in a script like `scripts/build_benchmark_candidates.py`, which
  captures passages with hashes and documents searches.

## Researching references

- Prefer official, primary or technical sources, and use encyclopedias as a supplement.
- Capture the passage verbatim, with its URL, access date, version and licence. Store the
  text only if the licence allows. Otherwise store a locator and hash.
- **Never write a URL, date or quotation from memory.** If a source cannot be reached,
  add a research task instead.
- Possibly fabricated studies: document the searches you ran (service, query, result
  summary).
- `check-sources --online` re-fetches captured web pages and confirms each passage still
  hashes the same.

## Reviewing

1. `hallab benchmark review ITEM --reviewer-id YOU`. The review type is chosen
   automatically:
   - first review
   - independent second review (when a different person already reviewed)
   - delayed repeat review (when you already reviewed, at least 24 h later)
2. You are asked whether you have seen the AI-proposed label. Answer honestly; it is
   recorded.
3. For each passage, give the relation (s/c/i/u) and reliability (h/m/l/?).
4. Give a verdict and a rationale. Passages you judged accordingly are cited
   automatically.
5. Add an optional uncertainty note, and flag research if needed.
6. Confirm the judgement is yours. Nothing is saved without that confirmation, and `q`
   quits without saving.

To correct your own review, use `--revise ANN_ID` with a reason. The original stays in
the file.

## Resolving disagreements

When current reviews disagree, the item becomes **DISPUTED**. Run `hallab benchmark
adjudicate ITEM --adjudicator-id YOU`: read both reviews, then record the final verdict
and rationale. Ideally the adjudicator is not one of the reviewers.

## Approving

```bash
hallab benchmark approve ITEM --approver-id YOU --confirm-own-judgement
```

Approval is refused when:
- the item is not REVIEWED
- the reviews disagree without an adjudication
- the policy needs more or independent reviews
- a SUPPORTED/CONTRADICTED label lacks captured cited evidence
- the approver ID is reserved for automated authors

To change an approved label, reopen it with
`mark ITEM --status DISPUTED --actor YOU --reason "..."`, review again, adjudicate if
needed, and approve again.

## Exporting a version

```bash
hallab benchmark export --version 1.0.0 --created-by YOU
```

This writes `data/benchmark/approved/benchmark-v1.0.0.json` containing approved items
only, with label, review level, approval record, partition and evidence, plus a
`dataset_sha256`. Versions are immutable. Bump MAJOR for label changes, MINOR for new
items, PATCH for metadata fixes.

## Running an evaluation

Predictions file:

```json
{
  "format": "hallab-benchmark-predictions",
  "benchmark_version": "1.0.0",
  "dataset_sha256": "<copy from the export>",
  "system": {"name": "my-verifier", "version": "0.1", "description": "optional"},
  "created_at": "2026-10-12T10:00:00Z",
  "predictions": [
    {"item_id": "bmk_...", "verdict": "CONTRADICTED"},
    {"item_id": "bmk_...", "abstained": true}
  ]
}
```

```bash
hallab benchmark evaluate --predictions preds.json --partition dev    # develop on dev
hallab benchmark evaluate --predictions preds.json --partition test   # final runs only; logged
```

The output gives N, answered/abstained/missing, coverage, conditional and strict
accuracy, per-class P/R/F1 with support, macro-F1, and the confusion matrix. Items
without a prediction count as **missing**, and explicit abstentions as **abstained**.
Both lower strict accuracy and recall, but not conditional accuracy.

## Understanding metric limits

- With 29 test items, one item moves accuracy by about 3.4 points. Always show counts.
- Results describe **this benchmark version only**. They are not an LLM hallucination
  rate.
- Single-review labels are weaker than independently double-reviewed ones; filter by
  `review_level` if needed.
- Retrieval metrics (`benchmark retrieval`) measure finding the evidence, which may
  contradict the claim. They are not verdict accuracy.
