"""CLI commands for reference collections and evidence retrieval.

hallab sources add     --collection NAME --file F (--metadata M.json | --title T ...)
hallab sources list    [--collection NAME]
hallab sources show    --collection NAME SOURCE_ID
hallab sources verify  --collection NAME
hallab index build     --collection NAME [--max-chars N] [--overlap N]
hallab search          --collection NAME --claim TEXT [--top-k N] [--json]
hallab retrieve        --session ID --collection NAME [--claim-id ID ...]
hallab eval retrieval  [--dataset PATH] [--k 1 3 5] [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from datetime import date
from pathlib import Path

from pydantic import ValidationError

from hallucination_lab.config import Settings
from hallucination_lab.errors import HallucinationLabError
from hallucination_lab.models import Evidence, RetrievalOutcome, VerificationSession
from hallucination_lab.retrieval.catalog import CollectionCatalog
from hallucination_lab.retrieval.evaluation import (
    DEFAULT_EVAL_PATH,
    evaluate,
    load_relevance_dataset,
    validate_criteria,
)
from hallucination_lab.retrieval.models import ChunkingConfig, DocumentType, SourceMetadata
from hallucination_lab.retrieval.search import RetrievalConfig, SearchHit
from hallucination_lab.retrieval.session_retrieval import retrieve_for_session
from hallucination_lab.storage import JsonSessionStore, summarize_validation_error

RETRIEVAL_BANNER = (
    "RETRIEVAL CANDIDATES: passages that share words with the claim. They are NOT verified\n"
    "evidence and may support, contradict, or be irrelevant to the claim. BM25 scores measure\n"
    "lexical overlap only; they are not probabilities, confidence, or truth estimates."
)


def _catalog() -> CollectionCatalog:
    return CollectionCatalog(Settings.from_env())


def _wrap(text: str, width: int = 96, indent: str = "      ", limit: int | None = 600) -> str:
    flat = " ".join(text.split())
    if limit and len(flat) > limit:
        flat = flat[:limit].rsplit(" ", 1)[0] + " [...]"
    return textwrap.fill(f'"{flat}"', width=width, initial_indent=indent, subsequent_indent=indent)


def _print_hit(hit: SearchHit, full: bool) -> None:
    entry, meta, p = hit.document.entry, hit.document.entry.metadata, hit.passage
    print(
        f"  [{hit.rank}] BM25 score {hit.score:.3f}   matched terms: {', '.join(hit.matched_terms)}"
    )
    print(f"      Source:   {meta.title}  ({entry.origin.value}; metadata {meta.status.value})")
    if meta.url:
        print(f"      URL:      {meta.url}")
    section = f", section: {p.section}" if p.section else ""
    print(f"      Location: {entry.local_filename} lines {p.line_start}-{p.line_end}{section}")
    print(f"      Passage:  {p.passage_id}   source: {entry.source_id}")
    print(_wrap(p.text, limit=None if full else 600))


def _retrieval_config(args: argparse.Namespace) -> RetrievalConfig:
    return RetrievalConfig(
        top_k=args.top_k,
        max_per_source=args.max_per_source,
        k1=args.k1,
        b=args.b,
        min_score=args.min_score,
    )


# ------------------------------------------------------------------------ sources


def cmd_sources_add(args: argparse.Namespace) -> int:
    if args.metadata:
        try:
            metadata = SourceMetadata.model_validate_json(
                Path(args.metadata).read_text(encoding="utf-8")
            )
        except ValidationError as exc:
            raise HallucinationLabError(
                f"metadata file {args.metadata} is invalid: {summarize_validation_error(exc)}"
            ) from exc
    else:
        if not args.title:
            raise HallucinationLabError("provide --metadata FILE or at least --title")
        metadata = SourceMetadata(
            title=args.title,
            author_or_organization=args.author,
            publisher=args.publisher,
            publication_date=args.published,
            url=args.url,
            accessed_on=args.accessed,
            document_type=DocumentType(args.doc_type),
            license=args.license,
            source_version=args.source_version,
        )
    entry, warnings = _catalog().add_document(
        args.collection, Path(args.file), metadata, encoding=args.encoding, replace=args.replace
    )
    print(f"Registered {entry.local_filename} in collection '{args.collection}'")
    print(f"  source_id:      {entry.source_id}")
    print(f"  content sha256: {entry.content_sha256}")
    print(f"  bytes/encoding: {entry.byte_size} / {entry.encoding}")
    print(f"  metadata:       {entry.metadata.status.value} (not verified by the system)")
    for w in warnings:
        print(f"  warning [{w.code}]: {w.message}")
    print(f"Run `hallab index build --collection {args.collection}` to make it searchable.")
    return 0


def cmd_sources_list(args: argparse.Namespace) -> int:
    catalog = _catalog()
    names = [args.collection] if args.collection else [c.name for c in catalog.list_collections()]
    if not names:
        print("No reference collections found.")
        return 0
    for name in names:
        manifest = catalog.load_manifest(name)
        indexed = "indexed" if catalog.index_path(name).is_file() else "not indexed"
        print(
            f"== {name} ({manifest.origin.value}, {len(manifest.documents)} documents, {indexed})"
        )
        for d in manifest.documents:
            m = d.metadata
            print(f"  {d.source_id}  {d.local_filename}")
            print(f"      {m.title} | {m.license or 'licence unknown'} | metadata {m.status.value}")
            if m.url:
                print(f"      {m.url}")
    return 0


def cmd_sources_show(args: argparse.Namespace) -> int:
    manifest = _catalog().load_manifest(args.collection)
    try:
        entry = manifest.get(args.source_id)
    except KeyError as exc:
        raise HallucinationLabError(
            f"no source {args.source_id} in collection {args.collection}"
        ) from exc
    print(json.dumps(entry.model_dump(mode="json"), indent=2, ensure_ascii=False))
    return 0


def cmd_sources_verify(args: argparse.Namespace) -> int:
    catalog = _catalog()
    manifest = catalog.load_manifest(args.collection)
    problems = catalog.verify(args.collection)
    for d in manifest.documents:
        status = problems.get(d.local_filename, "ok (hash matches registration)")
        print(f"  {d.local_filename:45} {status}")
    if problems:
        print(f"{len(problems)} document(s) failed the integrity check.", file=sys.stderr)
        return 1
    return 0


# -------------------------------------------------------------------------- index


def cmd_index_build(args: argparse.Namespace) -> int:
    chunking = ChunkingConfig(max_chars=args.max_chars, overlap_sentences=args.overlap)
    catalog = _catalog()
    processed = catalog.build_index(args.collection, chunking)
    print(
        f"Indexed '{processed.collection_name}': {len(processed.documents)} documents, "
        f"{len(processed.passages)} passages"
    )
    print(
        f"  chunking: max_chars={chunking.max_chars}, overlap_sentences="
        f"{chunking.overlap_sentences}; tokenizer {processed.tokenizer_version}"
    )
    print(f"  fingerprint: {processed.fingerprint}")
    print(f"  saved to: {catalog.index_path(args.collection)}")
    for w in processed.warnings:
        print(f"  warning [{w.code}] {w.local_filename}: {w.message}")
    return 0


# ------------------------------------------------------------------------- search


def _searcher(name: str):
    searcher, rebuilt = _catalog().searcher(name)
    if rebuilt:
        print(
            f"(index for '{name}' was missing or out of date and has been rebuilt)", file=sys.stderr
        )
    return searcher


def cmd_search(args: argparse.Namespace) -> int:
    result = _searcher(args.collection).search(args.claim, _retrieval_config(args))
    if args.json:
        print(
            json.dumps(
                {
                    "label": "retrieval_candidates_not_verified_evidence",
                    "query": result.query,
                    "query_terms": result.query_terms,
                    "parameters": result.parameters,
                    "notes": result.notes,
                    "hits": [
                        {
                            "rank": h.rank,
                            "bm25_score": h.score,
                            "matched_terms": h.matched_terms,
                            "passage_id": h.passage.passage_id,
                            "source_id": h.passage.source_id,
                            "source_title": h.document.entry.metadata.title,
                            "source_url": h.document.entry.metadata.url,
                            "local_filename": h.document.entry.local_filename,
                            "line_start": h.passage.line_start,
                            "line_end": h.passage.line_end,
                            "section": h.passage.section,
                            "passage_text": h.passage.text,
                        }
                        for h in result.hits
                    ],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    print(RETRIEVAL_BANNER + "\n")
    print(f"Claim:       {result.query}")
    print(f"Query terms: {', '.join(result.query_terms)}")
    for note in result.notes:
        print(f"Note:        {note}")
    print()
    for hit in result.hits:
        _print_hit(hit, args.full)
        print()
    if not result.hits:
        print(
            "  No retrieval candidates. This means only that no passage shares words with "
            "the claim;\n  it is not evidence that the claim is false."
        )
    elif result.suppressed:
        print(f"({result.suppressed} near-duplicate or over-cap passages were skipped)")
    return 0


def _print_session_evidence(session: VerificationSession, run_id: str | None = None) -> None:
    evidence: dict[str, Evidence] = {e.evidence_id: e for e in session.evidence}
    for n, claim in enumerate(session.active_claims, 1):
        latest = session.latest_retrieval(claim.claim_id)
        if latest is None or (
            run_id
            and not any(
                r is latest
                for run in session.retrieval_runs
                if run.run_id == run_id
                for r in run.results
            )
        ):
            continue
        print(f"{n:>3}. {claim.normalized_claim}")
        if latest.outcome is RetrievalOutcome.NO_SEARCHABLE_TERMS:
            print("       no searchable terms in this claim")
        elif latest.outcome is RetrievalOutcome.NO_CANDIDATES:
            print("       no retrieval candidates (not evidence that the claim is false)")
        for eid, score in zip(latest.evidence_ids, latest.scores, strict=True):
            ev = evidence[eid]
            loc = ev.passage_location
            lines = f" lines {loc.line_start}-{loc.line_end}" if loc and loc.line_start else ""
            print(f"       - [{score:.3f}] {ev.source_title}{lines}  ({eid})")
            print(_wrap(ev.quoted_passage, indent="           ", limit=220))


def cmd_retrieve(args: argparse.Namespace) -> int:
    store = JsonSessionStore(Settings.from_env().sessions_dir)
    session = store.load(args.session)
    searcher = _searcher(args.collection)
    updated, run = retrieve_for_session(
        session,
        searcher,
        _retrieval_config(args),
        claim_ids=args.claim_id or None,
        include_non_checkable=args.include_non_checkable,
    )
    path = store.save(updated)
    found = sum(r.outcome is RetrievalOutcome.CANDIDATES_FOUND for r in run.results)
    new = len(updated.evidence) - len(session.evidence)
    print(RETRIEVAL_BANNER + "\n")
    print(f"Retrieval run {run.run_id} on collection '{run.collection_name}'")
    print(
        f"  claims searched: {len(run.results)}; with candidates: {found}; "
        f"new evidence records: {new}"
    )
    print(f"  session saved: {path}\n")
    _print_session_evidence(updated, run.run_id)
    return 0


# ---------------------------------------------------------------------- evaluation


def cmd_eval_retrieval(args: argparse.Namespace) -> int:
    dataset = load_relevance_dataset(Path(args.dataset))
    searcher = _searcher(dataset.collection)
    problems = validate_criteria(dataset, searcher)
    if problems:
        raise HallucinationLabError(
            "relevance labels do not match the collection: " + "; ".join(problems)
        )
    report = evaluate(searcher, dataset, tuple(args.k), _retrieval_config(args))
    summary = report.summary()
    if args.json:
        print(
            json.dumps(
                {
                    "summary": summary,
                    "parameters": report.parameters,
                    "per_query": [o.__dict__ for o in report.outcomes],
                },
                indent=2,
                default=str,
            )
        )
        return 0
    print(f"Retrieval evaluation: {report.dataset} on collection '{report.collection}'")
    print(f"Labels: {dataset.labelled_by}; {dataset.review_status}")
    print("These metrics describe passage retrieval only, not hallucination detection.\n")
    for key, value in summary.items():
        print(f"  {key:32} {value}")
    print("\nPer query (first relevant rank; '-' = not retrieved):")
    for o in report.outcomes:
        if o.negative:
            print(f"  {o.query_id}  negative   top score {o.top_score}  [{o.category}]")
        else:
            rank = o.first_relevant_rank or "-"
            print(
                f"  {o.query_id}  rank {rank!s:<4}  recall@{max(report.k_values)} "
                f"{o.criteria_found[max(report.k_values)]:.2f}  [{o.category}]"
            )
    return 0


# -------------------------------------------------------------------------- parser


def _add_retrieval_options(p: argparse.ArgumentParser) -> None:
    d = RetrievalConfig()
    p.add_argument("--top-k", type=int, default=d.top_k, help=f"results (default {d.top_k})")
    p.add_argument(
        "--max-per-source",
        type=int,
        default=d.max_per_source,
        help=f"max passages per document (default {d.max_per_source})",
    )
    p.add_argument("--k1", type=float, default=d.k1, help=f"BM25 k1 (default {d.k1})")
    p.add_argument("--b", type=float, default=d.b, help=f"BM25 b (default {d.b})")
    p.add_argument(
        "--min-score",
        type=float,
        default=d.min_score,
        help="drop results scoring at or below this (default 0)",
    )


def register(sub: argparse._SubParsersAction) -> None:
    p_sources = sub.add_parser("sources", help="manage reference documents")
    s_sub = p_sources.add_subparsers(dest="action", required=True)

    p_add = s_sub.add_parser("add", help="validate and register a .txt/.md reference document")
    p_add.add_argument("--collection", required=True, help="collection name (created if new)")
    p_add.add_argument("--file", required=True, help="path to the document")
    p_add.add_argument("--metadata", help="JSON file with source metadata (see docs)")
    p_add.add_argument("--title")
    p_add.add_argument("--author", help="author or organization")
    p_add.add_argument("--publisher")
    p_add.add_argument("--url")
    p_add.add_argument("--published", type=date.fromisoformat, help="YYYY-MM-DD")
    p_add.add_argument("--accessed", type=date.fromisoformat, help="YYYY-MM-DD")
    p_add.add_argument("--license")
    p_add.add_argument("--source-version")
    p_add.add_argument("--doc-type", default="other", choices=[t.value for t in DocumentType])
    p_add.add_argument(
        "--encoding", default="utf-8", choices=["utf-8", "utf-16", "latin-1", "cp1252"]
    )
    p_add.add_argument(
        "--replace",
        action="store_true",
        help="replace a registered document with the same file name",
    )
    p_add.set_defaults(func=cmd_sources_add)

    p_list = s_sub.add_parser("list", help="list collections and their documents")
    p_list.add_argument("--collection")
    p_list.set_defaults(func=cmd_sources_list)

    p_show = s_sub.add_parser("show", help="show full metadata for one source")
    p_show.add_argument("--collection", required=True)
    p_show.add_argument("source_id")
    p_show.set_defaults(func=cmd_sources_show)

    p_verify = s_sub.add_parser("verify", help="check documents against registered hashes")
    p_verify.add_argument("--collection", required=True)
    p_verify.set_defaults(func=cmd_sources_verify)

    p_index = sub.add_parser("index", help="build search indexes")
    i_sub = p_index.add_subparsers(dest="action", required=True)
    p_build = i_sub.add_parser("build", help="ingest, segment and index a collection")
    p_build.add_argument("--collection", required=True)
    p_build.add_argument("--max-chars", type=int, default=ChunkingConfig().max_chars)
    p_build.add_argument(
        "--overlap",
        type=int,
        default=ChunkingConfig().overlap_sentences,
        help="sentences shared between consecutive passages",
    )
    p_build.set_defaults(func=cmd_index_build)

    p_search = sub.add_parser("search", help="retrieve candidate passages for a claim")
    p_search.add_argument("--collection", required=True)
    p_search.add_argument("--claim", required=True, help="claim text to search for")
    p_search.add_argument("--json", action="store_true")
    p_search.add_argument("--full", action="store_true", help="print whole passages")
    _add_retrieval_options(p_search)
    p_search.set_defaults(func=cmd_search)

    p_ret = sub.add_parser("retrieve", help="retrieve candidates for a saved session's claims")
    p_ret.add_argument("--session", required=True)
    p_ret.add_argument("--collection", required=True)
    p_ret.add_argument("--claim-id", action="append", help="only this claim (repeatable)")
    p_ret.add_argument(
        "--include-non-checkable",
        action="store_true",
        help="also search claims flagged as opinion/question/etc.",
    )
    _add_retrieval_options(p_ret)
    p_ret.set_defaults(func=cmd_retrieve)

    p_eval = sub.add_parser("eval", help="evaluation utilities")
    e_sub = p_eval.add_subparsers(dest="action", required=True)
    p_er = e_sub.add_parser("retrieval", help="Recall@k / Precision@k / MRR on labelled queries")
    p_er.add_argument("--dataset", default=str(DEFAULT_EVAL_PATH))
    p_er.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    p_er.add_argument("--json", action="store_true")
    _add_retrieval_options(p_er)
    p_er.set_defaults(func=cmd_eval_retrieval)
