"""
CLI entry point for the kombinat ingest tool.

Usage (run with uv from the kombinat/ project root):
    uv run python -m kombinat.tools.ingest --split squad
    uv run python -m kombinat.tools.ingest --split squad --max-docs 1000 --dry-run
    uv run python -m kombinat.tools.ingest --split paq --bm25-top-k 10000 --dense-top-k 10000
    uv run python -m kombinat.tools.ingest --split paq --embedding-model all-mpnet-base-v2

Requires the `ingest` extras: `uv sync --extra ingest`.
"""
from __future__ import annotations

import argparse
import asyncio
import time

import asyncpg
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from kombinat.tools.ingest.bm25 import bm25_retrieve, build_bm25_index
from kombinat.tools.ingest.config import IngestConfig
from kombinat.tools.ingest.dense import (
    build_dense_index,
    compute_nprobe,
    dense_retrieve,
    embed_queries,
)
from kombinat.tools.ingest.fusion import rrf_fuse
from kombinat.tools.ingest.pairs import CandidatePair, build_candidates
from kombinat.tools.ingest.source import load_split
from kombinat.tools.ingest.writer import write_batch

console = Console()


async def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest dataset into kombinat")

    # Required
    parser.add_argument("--split", required=True, help="Dataset split to process (e.g. squad, paq, wikipedia)")

    # Retrieval tuning
    parser.add_argument("--bm25-top-k", type=int, default=10_000, help="BM25 retrieval depth (default: 10000)")
    parser.add_argument("--dense-top-k", type=int, default=10_000, help="Dense retrieval depth (default: 10000)")
    parser.add_argument("--candidates-per-query", type=int, default=5_000, help="Final candidates after RRF (default: 5000)")
    parser.add_argument("--rrf-k", type=int, default=60, help="RRF constant (default: 60)")

    # Embedding model
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", help="HuggingFace model ID for dense retrieval")
    parser.add_argument("--embedding-device", default="mps", choices=["mps", "cuda", "cpu"])
    parser.add_argument("--embedding-batch-size", type=int, default=256)

    # Limits and modes
    parser.add_argument("--max-docs", type=int, default=None, help="Limit corpus size (dev mode)")
    parser.add_argument("--dry-run", action="store_true", help="Build index + retrieve, don't write to DB")

    # Infrastructure
    parser.add_argument("--database-url", default=None, help="Override DATABASE_URL env var")
    parser.add_argument("--faiss-index-dir", default="~/.kombinat/faiss_indexes")
    parser.add_argument("--faiss-min-search-docs", type=int, default=100_000,
                        help="Min docs to search in FAISS (controls nprobe). Small corpora get brute force.")

    args = parser.parse_args()

    overrides: dict[str, object] = dict(
        split=args.split,
        max_docs=args.max_docs,
        bm25_top_k=args.bm25_top_k,
        dense_top_k=args.dense_top_k,
        candidates_per_query=args.candidates_per_query,
        rrf_k=args.rrf_k,
        embedding_model=args.embedding_model,
        embedding_device=args.embedding_device,
        embedding_batch_size=args.embedding_batch_size,
        faiss_index_dir=args.faiss_index_dir,
        faiss_min_search_docs=args.faiss_min_search_docs,
    )
    if args.database_url:
        overrides["database_url"] = args.database_url
    config = IngestConfig(**overrides)

    # ── Header ──
    console.print()
    console.print(Panel(
        f"[bold]kombinat ingest[/bold] · {config.source_dataset_label}",
        subtitle="hard negative candidate mining",
        style="blue",
    ))

    # ── Run summary ──
    summary = Table(show_header=False, box=None, padding=(0, 2))
    summary.add_row("Split", f"[bold]{config.split}[/bold]")
    summary.add_row("Dataset", config.dataset_name)
    summary.add_row("Max docs", str(config.max_docs) if config.max_docs else "all")
    summary.add_row("Embedding model", config.embedding_model)
    summary.add_row("Device", config.embedding_device)
    summary.add_row("BM25 top-k", f"{config.bm25_top_k:,}")
    summary.add_row("Dense top-k", f"{config.dense_top_k:,}")
    summary.add_row("Candidates/query", f"{config.candidates_per_query:,}")
    summary.add_row("Mode", "[yellow]DRY RUN[/yellow]" if args.dry_run else "write to DB")
    console.print(summary)
    console.print()

    # ── 1. Load corpus ──
    with console.status("[bold blue]Loading dataset from HuggingFace..."):
        t0 = time.time()
        corpus = load_split(config)
    console.print(
        f"[green]✓[/green] Loaded [bold]{len(corpus.doc_texts):,}[/bold] unique docs, "
        f"[bold]{len(corpus.queries):,}[/bold] queries from [bold]{config.split}[/bold] "
        f"({time.time() - t0:.1f}s)"
    )

    max_pairs = len(corpus.queries) * config.candidates_per_query
    console.print(
        f"  Will produce up to [bold]{max_pairs:,}[/bold] candidate pairs "
        f"({len(corpus.queries):,} queries × {config.candidates_per_query:,} candidates)"
    )
    console.print()

    # ── 2. Build BM25 index ──
    with console.status("[bold blue]Building BM25 index..."):
        t0 = time.time()
        bm25_index = build_bm25_index(corpus)
    console.print(
        f"[green]✓[/green] BM25 index built over {len(corpus.doc_texts):,} docs ({time.time() - t0:.1f}s)"
    )

    # ── 3. Build dense index ──
    with console.status(f"[bold blue]Building FAISS index ({config.embedding_model})..."):
        t0 = time.time()
        dense_index = build_dense_index(corpus, config)
    nprobe = compute_nprobe(len(corpus.doc_texts), dense_index.nlist, config.faiss_min_search_docs)
    search_pct = (nprobe / dense_index.nlist) * 100
    console.print(
        f"[green]✓[/green] FAISS IVFFlat index: {dense_index.nlist:,} cells, "
        f"nprobe={nprobe} (searching ~{search_pct:.0f}% of corpus) ({time.time() - t0:.1f}s)"
    )

    # ── 4. Embed queries ──
    # Reuse the model loaded during index build. When the index was loaded from cache,
    # dense_index.model is None and embed_queries will load a fresh one.
    with console.status(f"[bold blue]Embedding {len(corpus.queries):,} queries..."):
        t0 = time.time()
        query_embeddings = embed_queries(corpus.queries, config, model=dense_index.model)
    console.print(
        f"[green]✓[/green] Embedded {len(corpus.queries):,} queries ({time.time() - t0:.1f}s)"
    )
    console.print()

    # ── 5. Retrieve + fuse + build pairs (streaming writes to Postgres) ──
    conn: asyncpg.Connection | None = None
    if not args.dry_run:
        conn = await asyncpg.connect(config.database_url)

    buffer: list[CandidatePair] = []
    total_pairs = 0
    total_written = 0

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("· {task.fields[pairs]:,} pairs"),
            console=console,
        ) as progress:
            task = progress.add_task(
                "Retrieving candidates",
                total=len(corpus.queries),
                pairs=0,
            )
            for i, (query, pos_doc_id) in enumerate(
                zip(corpus.queries, corpus.positive_doc_ids)
            ):
                bm25_hits = bm25_retrieve(bm25_index, query, config.bm25_top_k)
                dense_hits = dense_retrieve(dense_index, query_embeddings[i], config.dense_top_k)
                fused = rrf_fuse(bm25_hits, dense_hits, k=config.rrf_k)
                candidates = build_candidates(query, pos_doc_id, fused, corpus, config)
                buffer.extend(candidates)
                total_pairs += len(candidates)
                if conn is not None and len(buffer) >= config.db_batch_size:
                    total_written += await write_batch(conn, buffer)
                    buffer.clear()
                progress.update(task, advance=1, pairs=total_pairs)

        if conn is not None and buffer:
            total_written += await write_batch(conn, buffer)
            buffer.clear()
    finally:
        if conn is not None:
            await conn.close()

    console.print()
    console.print(
        f"[green]✓[/green] Generated [bold]{total_pairs:,}[/bold] candidate pairs "
        f"({total_pairs / max(len(corpus.queries), 1):.1f} avg per query)"
    )
    if args.dry_run:
        console.print("[yellow]⚠[/yellow] Dry run — skipping database write")
    else:
        console.print(f"[green]✓[/green] Wrote [bold]{total_written:,}[/bold] pairs to database")
    console.print()

    console.print()
    console.print(Panel(
        f"[bold green]Done.[/bold green] {total_pairs:,} candidate pairs from {config.split}",
        style="green",
    ))


if __name__ == "__main__":
    asyncio.run(main())
