# tools/ingest — Implementation Instructions

**Scope**: `kombinat/tools/ingest/` module
**Operator**: Maintainers only (you). Not contributor-facing.
**Runtime**: Your M3 MacBook, run manually per dataset.

---

## 1. What the ingest tool does

Takes a source dataset of known positive (query, document) pairs, finds **hard negative candidates** for each query using BM25 + dense retrieval with RRF fusion, and writes the candidates as `unlabeled` pairs into kombinat's PostgreSQL database.

Think of it like this: the source dataset tells us "this document IS relevant to this query." But what we need for contrastive training is to also know which documents AREN'T relevant — and specifically, which ones *look* relevant but aren't (hard negatives). The ingest tool finds those candidates. The annotators then verify them.

---

## 2. Source dataset

**nomic-ai/nomic-embed-unsupervised-data** on HuggingFace.

- 239M rows across 29 splits (reddit_title_body, amazon_reviews, paq, s2orc_*, wikipedia, etc.)
- Each row: `query` (str), `document` (str), `dataset` (str), `shard` (int)
- Each row is a **positive pair** — the document is known-relevant to the query
- The `document` column contains the corpus we retrieve from

**Key insight**: within a single split, the documents form a natural corpus. A reddit query's hard negatives come from other reddit documents, not from academic papers. We process each split independently — build an index per split, retrieve within that split.

---

## 3. Pipeline per split

```
HuggingFace dataset (one split, e.g. "squad")
│
├── 1. Load & deduplicate documents → corpus[]
│
├── 2. Build BM25 index (rank_bm25, in-memory)
│
├── 3. Embed all documents → FAISS IVFFlat index ({embedding_model}, saved to disk)
│       └── Batched encoding on MPS, model configurable via --embedding-model
│
├── 4. For each query:
│   ├── a. BM25 top-1K retrieval → ranked doc IDs
│   ├── b. FAISS top-1K retrieval → ranked doc IDs
│   ├── c. RRF fusion of both rankings
│   ├── d. Filter out the known positive doc
│   └── e. Take top-10 candidates as hard negatives
│         (every candidate costs 2 annotations — this is the
│          project's annotation-budget knob; keep it small)
│
├── 5. Build (query, doc) candidate pairs
│
└── 6. Write to Postgres as unlabeled pairs
        ├── source_dataset = "nomic-ai/nomic-embed-unsupervised-data/{split}"
        └── uuid5(query + doc_id + source_dataset) for deterministic dedup
```

---

## 4. Architecture decisions

### 4.1 Process one split at a time

239M rows won't fit in memory at once. Each split is a manageable unit. The CLI takes `--split` (required) and optionally `--max-docs` for development. Start with the smallest split (`squad`, 25K rows) to validate the pipeline end-to-end, then scale up through larger splits.

### 4.2 Embedding model: configurable, default `all-MiniLM-L6-v2`

The embedding model is configurable via `--embedding-model`. Any model compatible with `sentence-transformers` works. The default is `all-MiniLM-L6-v2`:

- 22M parameters, 384 dimensions
- Runs comfortably on M3 MacBook via MPS (Metal Performance Shaders — Apple Silicon's GPU compute, the Apple equivalent of CUDA). PyTorch detects it automatically when you pass `device="mps"`.
- ~5000 docs/sec on M3 with batch_size=256

For better retrieval quality at the cost of speed, swap to `all-mpnet-base-v2` (110M params, 768d) or `BAAI/bge-small-en-v1.5` (33M params, 384d). The dimension is detected automatically — FAISS index dimensions adapt to whatever model you choose.

This is NOT the embedding model we're training. It's a cheap retrieval tool to find candidate pairs. The quality bar is "find documents that are plausibly relevant" — false positives in retrieval are fine (the annotators will filter them). False negatives in retrieval are acceptable too (we'll miss some hard negatives, but there are plenty).

**Important**: if you change the embedding model for a split that already has a cached FAISS index, delete the old index file first (`~/.kombinat/faiss_indexes/{split}.index`). The dimensions won't match.

### 4.3 FAISS for dense index

- faiss-cpu works on M3 (no GPU required)
- Always use `IndexIVFFlat` with `nlist = max(1, min(4 * int(sqrt(n)), n // 40))`
- **Dynamic nprobe based on corpus size** — the goal is to search enough cells that we're examining a meaningful fraction of the corpus before returning top-K:

```python
def compute_nprobe(n_docs: int, nlist: int, min_search_docs: int = 100_000) -> int:
    """Compute nprobe so we search at least min_search_docs documents.

    For small corpora (n_docs <= min_search_docs), this returns nlist,
    which means searching every cell = brute force.
    For larger corpora, searches ~10-20% of the corpus.
    """
    if n_docs <= min_search_docs:
        return nlist  # brute force — search everything
    docs_per_cell = n_docs / nlist
    return max(1, min(nlist, int(min_search_docs / docs_per_cell)))
```

Examples:
- **squad** (25K docs, nlist=625): `nprobe=625` → searches all 25K docs (brute force)
- **gooaq** (1.28M docs, nlist=4528): `nprobe=354` → searches ~100K docs
- **paq** (53.9M docs, nlist=29,371): `nprobe=54` → searches ~100K docs
- **reddit** (66.2M docs, nlist=32,557): `nprobe=49` → searches ~100K docs

To search deeper (e.g. 200K docs for 1M corpus), increase `--faiss-min-search-docs 200000`.

- Index saved to disk after building — reusable across runs
- In tests with tiny fixtures (5 docs), use `IndexFlatIP` via a test-only helper — don't pollute production code with a conditional

### 4.4 RRF (Reciprocal Rank Fusion)

```python
def rrf_score(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)
```

For each query, get ranked lists from BM25 and FAISS, compute RRF score for each document across both lists, sort by combined score, take top-N. k=60 is the standard constant from the original RRF paper (Cormack et al., 2009).

### 4.5 Deterministic pair IDs

```python
source_dataset = "nomic-ai/nomic-embed-unsupervised-data/squad"
pair_id = uuid5(NAMESPACE_URL, f"{query_text}|{doc_id}|{source_dataset}")
```

`source_dataset` is always the full path: `{dataset_name}/{split}`. This means the same document text appearing in two different splits produces different pair_ids — correct, because the retrieval context is different.

Re-running the ingest tool on the same data produces the same UUIDs. The `ON CONFLICT DO NOTHING` insert makes re-runs safe — idempotent by design.

### 4.6 Chunked processing for large splits

For splits beyond what fits comfortably in memory:
1. Load documents in streaming mode from HuggingFace
2. Build FAISS IVFFlat index incrementally (train on first batch, `add()` subsequent batches)
3. Process queries in chunks of 10K
4. Write to Postgres in batches of 5K pairs

---

## 5. Code structure

```
kombinat/tools/ingest/
├── __init__.py
├── __main__.py          # CLI entry: uv run python -m kombinat.tools.ingest
├── config.py            # IngestConfig (Pydantic)
├── source.py            # load dataset from HuggingFace, deduplicate
├── bm25.py              # BM25 index build + query
├── dense.py             # embedding + FAISS index build + query
├── fusion.py            # RRF merge of ranked lists
├── pairs.py             # candidate pair construction + dedup
└── writer.py            # async batch write to Postgres
```

Plus test files:

```
tests/
├── test_ingest_source.py
├── test_ingest_bm25.py
├── test_ingest_dense.py
├── test_ingest_fusion.py
├── test_ingest_pairs.py
└── test_ingest_writer.py
```

---

## 6. Module specifications

### 6.1 `config.py`

```python
from pydantic import BaseModel

class IngestConfig(BaseModel):
    # Source
    dataset_name: str = "nomic-ai/nomic-embed-unsupervised-data"
    split: str = "squad"  # start with smallest split
    max_docs: int | None = None  # limit for development

    # Retrieval
    bm25_top_k: int = 1_000
    dense_top_k: int = 1_000
    rrf_k: int = 60  # RRF constant
    candidates_per_query: int = 10  # final candidates after RRF + filtering (annotation-budget knob)

    # Embedding
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_batch_size: int = 256
    embedding_device: str = "mps"  # "mps" for M3, "cuda" for GPU, "cpu" fallback

    # FAISS
    faiss_index_dir: str = "~/.kombinat/faiss_indexes"
    faiss_min_search_docs: int = 100_000  # nprobe computed from this

    # Writer
    db_batch_size: int = 5000

    # Database
    database_url: str = ""

    @property
    def source_dataset_label(self) -> str:
        """e.g. 'nomic-ai/nomic-embed-unsupervised-data/squad'"""
        return f"{self.dataset_name}/{self.split}"
```

### 6.2 `source.py` — dataset loading

```python
@dataclass
class Corpus:
    """Deduplicated document corpus from one split."""
    doc_ids: list[str]          # stable identifier per document
    doc_texts: list[str]        # document content
    queries: list[str]          # all queries
    positive_doc_ids: list[str] # positive doc_id for each query (parallel to queries)
    split: str
    doc_id_to_idx: dict[str, int]  # doc_id → index in doc_texts

def load_split(config: IngestConfig) -> Corpus:
    """Load a split, deduplicate documents, return Corpus."""
    ...
```

**Document deduplication**: Multiple queries can share the same positive document. Build a set of unique documents. Use a content hash as the stable `doc_id` (since the source dataset doesn't provide document IDs):

```python
doc_id = hashlib.sha256(doc_text.encode()).hexdigest()[:16]
```

### 6.3 `bm25.py` — BM25 retrieval

```python
@dataclass
class BM25Index:
    index: BM25Okapi
    doc_ids: list[str]

def build_bm25_index(corpus: Corpus) -> BM25Index:
    """Tokenize documents and build BM25Okapi index."""
    tokenized = [doc.lower().split() for doc in corpus.doc_texts]
    return BM25Index(
        index=BM25Okapi(tokenized),
        doc_ids=corpus.doc_ids,
    )

def bm25_retrieve(
    index: BM25Index, query: str, top_k: int
) -> list[tuple[str, float]]:
    """Return [(doc_id, score), ...] sorted by score descending."""
    tokenized_query = query.lower().split()
    scores = index.index.get_scores(tokenized_query)
    top_indices = scores.argsort()[-top_k:][::-1]
    return [(index.doc_ids[i], float(scores[i])) for i in top_indices]
```

### 6.4 `dense.py` — embedding + FAISS

```python
@dataclass
class DenseIndex:
    index: faiss.Index
    doc_ids: list[str]
    dimension: int
    nlist: int  # number of IVF cells (for nprobe computation)

def build_dense_index(
    corpus: Corpus, config: IngestConfig
) -> DenseIndex:
    """Embed all documents and build FAISS index.

    Saves index to disk at {faiss_index_dir}/{split}.index
    If index file already exists, loads from disk (skip re-embedding).
    """
    ...

def dense_retrieve(
    index: DenseIndex,
    query_embedding: np.ndarray,
    top_k: int,
) -> list[tuple[str, float]]:
    """Return [(doc_id, score), ...] sorted by score descending."""
    ...

def embed_queries(
    queries: list[str], config: IngestConfig
) -> np.ndarray:
    """Embed a batch of queries. Returns (n, dim) array."""
    ...
```

**Index persistence**: Building the FAISS index for a large split (encoding millions of documents) takes hours. Save it to disk after building. On re-run, check if `{faiss_index_dir}/{split}.index` exists and load it instead of re-encoding. Store `doc_ids` alongside as `{split}.doc_ids.json`.

**Always IVFFlat**: Use `IndexIVFFlat` for all production runs. Compute `nlist = max(1, min(4 * int(sqrt(n)), n // 40))`. Train on the full corpus (or a sample for very large corpora), then add all vectors. Set `index.nprobe = compute_nprobe(n_docs, nlist, config.faiss_min_search_docs)` before querying — this ensures small corpora get brute-force search and large corpora search a meaningful fraction. For unit tests with 5-doc fixtures, the test helper can use `IndexFlatIP` directly — this is a test concern, not a production code path.

### 6.5 `fusion.py` — RRF

```python
@dataclass
class RankedCandidate:
    doc_id: str
    rrf_score: float
    bm25_rank: int | None
    dense_rank: int | None

def rrf_fuse(
    bm25_results: list[tuple[str, float]],
    dense_results: list[tuple[str, float]],
    k: int = 60,
) -> list[RankedCandidate]:
    """Fuse two ranked lists using Reciprocal Rank Fusion.

    Returns candidates sorted by combined RRF score descending.
    """
    scores: dict[str, RankedCandidate] = {}

    for rank, (doc_id, _) in enumerate(bm25_results):
        scores[doc_id] = RankedCandidate(
            doc_id=doc_id,
            rrf_score=1.0 / (k + rank + 1),
            bm25_rank=rank + 1,
            dense_rank=None,
        )

    for rank, (doc_id, _) in enumerate(dense_results):
        if doc_id in scores:
            scores[doc_id].rrf_score += 1.0 / (k + rank + 1)
            scores[doc_id].dense_rank = rank + 1
        else:
            scores[doc_id] = RankedCandidate(
                doc_id=doc_id,
                rrf_score=1.0 / (k + rank + 1),
                bm25_rank=None,
                dense_rank=rank + 1,
            )

    return sorted(scores.values(), key=lambda c: c.rrf_score, reverse=True)
```

### 6.6 `pairs.py` — candidate pair construction

```python
from uuid import uuid5, NAMESPACE_URL

@dataclass
class CandidatePair:
    pair_id: str           # deterministic UUID
    query_text: str
    doc_id: str
    doc_text: str
    source_dataset: str
    retrieval_method: str  # "bm25+dense"
    source_rank: int       # rank in RRF result

def build_candidates(
    query: str,
    positive_doc_id: str,
    rrf_results: list[RankedCandidate],
    corpus: Corpus,
    config: IngestConfig,
) -> list[CandidatePair]:
    """Filter out positive doc, take top-N, build CandidatePair objects."""
    source_dataset = config.source_dataset_label  # e.g. "nomic-ai/nomic-embed-unsupervised-data/squad"
    candidates = []
    for rank, rc in enumerate(rrf_results):
        if rc.doc_id == positive_doc_id:
            continue  # skip known positive
        if len(candidates) >= config.candidates_per_query:
            break
        doc_idx = corpus.doc_id_to_idx[rc.doc_id]
        pair_id = str(uuid5(
            NAMESPACE_URL,
            f"{query}|{rc.doc_id}|{source_dataset}"
        ))
        candidates.append(CandidatePair(
            pair_id=pair_id,
            query_text=query,
            doc_id=rc.doc_id,
            doc_text=corpus.doc_texts[doc_idx],
            source_dataset=source_dataset,
            retrieval_method="bm25+dense",
            source_rank=rank + 1,
        ))
    return candidates
```

### 6.7 `writer.py` — Postgres batch writer

```python
import asyncpg

async def write_pairs(
    pairs: list[CandidatePair],
    database_url: str,
    batch_size: int = 5000,
) -> int:
    """Write candidate pairs to the pairs table. Returns count inserted.

    Uses ON CONFLICT DO NOTHING for idempotent re-runs.
    """
    conn = await asyncpg.connect(database_url)
    try:
        inserted = 0
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            result = await conn.executemany(
                """
                INSERT INTO pairs (id, query_text, doc_id, doc_text,
                                   source_dataset, retrieval_method,
                                   source_rank, status)
                VALUES ($1, $2, $3, $4, $5, $6, $7, 'unlabeled')
                ON CONFLICT (id) DO NOTHING
                """,
                [
                    (p.pair_id, p.query_text, p.doc_id, p.doc_text,
                     p.source_dataset, p.retrieval_method, p.source_rank)
                    for p in batch
                ],
            )
            inserted += len(batch)  # approximate; ON CONFLICT skips silently
        return inserted
    finally:
        await conn.close()
```

### 6.8 `__main__.py` — CLI orchestration

```python
"""
CLI entry point.

Usage:
    uv run python -m kombinat.tools.ingest --split squad
    uv run python -m kombinat.tools.ingest --split squad --max-docs 1000 --dry-run
    # deep-retrieval experiment (default depth is 1000; each kept candidate costs 2 annotations)
    uv run python -m kombinat.tools.ingest --split paq --bm25-top-k 2000 --dense-top-k 2000
    uv run python -m kombinat.tools.ingest --split paq --embedding-model all-mpnet-base-v2
"""
import argparse
import asyncio
import os
import time

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn
from rich.table import Table

console = Console()

def main():
    parser = argparse.ArgumentParser(description="Ingest dataset into kombinat")

    # Required
    parser.add_argument("--split", required=True, help="Dataset split to process (e.g. squad, paq, wikipedia)")

    # Retrieval tuning
    parser.add_argument("--bm25-top-k", type=int, default=1_000, help="BM25 retrieval depth (default: 1000)")
    parser.add_argument("--dense-top-k", type=int, default=1_000, help="Dense retrieval depth (default: 1000)")
    parser.add_argument("--candidates-per-query", type=int, default=10, help="Hard-negative candidates kept per query after RRF (default: 10)")
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

    config = IngestConfig(
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
        database_url=args.database_url or os.environ.get("DATABASE_URL", ""),
    )

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
    with console.status(f"[bold blue]Embedding {len(corpus.queries):,} queries..."):
        t0 = time.time()
        query_embeddings = embed_queries(corpus.queries, config)
    console.print(
        f"[green]✓[/green] Embedded {len(corpus.queries):,} queries ({time.time() - t0:.1f}s)"
    )
    console.print()

    # ── 5. Retrieve + fuse + build pairs ──
    all_pairs: list[CandidatePair] = []
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
            all_pairs.extend(candidates)
            progress.update(task, advance=1, pairs=len(all_pairs))

    console.print()
    console.print(
        f"[green]✓[/green] Generated [bold]{len(all_pairs):,}[/bold] candidate pairs "
        f"({len(all_pairs) / len(corpus.queries):.1f} avg per query)"
    )
    console.print()

    # ── 6. Write to Postgres ──
    if not args.dry_run:
        with console.status(f"[bold blue]Writing {len(all_pairs):,} pairs to Postgres..."):
            t0 = time.time()
            inserted = asyncio.run(
                write_pairs(all_pairs, config.database_url, config.db_batch_size)
            )
        console.print(
            f"[green]✓[/green] Wrote {inserted:,} pairs to database ({time.time() - t0:.1f}s)"
        )
    else:
        console.print("[yellow]⚠[/yellow] Dry run — skipping database write")

    console.print()
    console.print(Panel(
        f"[bold green]Done.[/bold green] {len(all_pairs):,} candidate pairs from {config.split}",
        style="green",
    ))

if __name__ == "__main__":
    main()
```

---

## 7. Dependencies

Add to `pyproject.toml` under `[project.optional-dependencies]`:

```toml
ingest = [
    "rank-bm25>=0.2",
    "sentence-transformers>=3.3",
    "datasets>=3.1",
    "faiss-cpu>=1.9",
    "numpy>=1.26",
    "rich>=13.9",
]
```

Install with: `uv sync --extra ingest`

Note: `faiss-cpu` works on M3 via pip. No GPU build needed.

---

## 8. TDD build order

Each phase follows RED→GREEN→REFACTOR strictly. Use small synthetic fixtures (5-10 documents) for unit tests — never download the real dataset in CI.

### Phase 0: Test fixtures

Create `tests/conftest_ingest.py` with shared fixtures:

```python
@pytest.fixture
def tiny_corpus() -> Corpus:
    """5 documents, 3 queries, known positives."""
    docs = [
        "Paris is the capital of France and its largest city.",
        "Berlin is the capital of Germany, located on the River Spree.",
        "The Eiffel Tower is a wrought-iron lattice tower in Paris.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language for data science.",
    ]
    queries = [
        "What is the capital of France?",
        "Tell me about the Eiffel Tower",
        "What programming language is used for ML?",
    ]
    positive_doc_ids = [
        hashlib.sha256(docs[0].encode()).hexdigest()[:16],  # Paris doc
        hashlib.sha256(docs[2].encode()).hexdigest()[:16],  # Eiffel doc
        hashlib.sha256(docs[4].encode()).hexdigest()[:16],  # Python doc
    ]
    ...
```

### Phase 1: Source loading + deduplication

**RED**: `test_ingest_source.py`
- `load_split()` with a mocked HuggingFace dataset returns a `Corpus`
- duplicate documents in the source are deduplicated (same text → same doc_id)
- `doc_id_to_idx` maps back correctly
- `max_docs` limits the number of documents loaded
- queries and positive_doc_ids are parallel (same length, correct alignment)

**GREEN**: Implement `source.py`

### Phase 2: BM25 index + retrieval

**RED**: `test_ingest_bm25.py`
- `build_bm25_index()` accepts a Corpus, returns BM25Index
- `bm25_retrieve()` for "capital of France" returns the Paris doc in top-3
- `bm25_retrieve()` returns `top_k` results sorted by score descending
- `bm25_retrieve()` with a nonsense query returns results (even if low-scored — BM25 always returns something)
- returned doc_ids are valid members of the corpus

**GREEN**: Implement `bm25.py`

### Phase 3: Dense embedding + FAISS index

**RED**: `test_ingest_dense.py`
- `build_dense_index()` creates a FAISS index with correct dimension (384)
- `build_dense_index()` saves index to disk at configured path
- `build_dense_index()` loads from disk if index file already exists (skip re-encoding)
- `dense_retrieve()` for an embedded query returns `top_k` results
- `dense_retrieve()` for "capital of France" returns the Paris doc in top-3
- `embed_queries()` returns array of shape `(n_queries, 384)`

**GREEN**: Implement `dense.py`

**Note**: These tests will actually load `all-MiniLM-L6-v2` (22MB download, cached after first run). Mark them with `@pytest.mark.slow` so CI can skip them if needed. For fast unit tests, mock the embedding model to return random vectors — the retrieval quality doesn't matter for correctness tests, only the shapes and plumbing.

### Phase 4: RRF fusion

**RED**: `test_ingest_fusion.py`
- `rrf_fuse()` with two identical rankings returns doc order preserved
- `rrf_fuse()` with disjoint rankings merges all documents
- `rrf_fuse()` with overlapping rankings: docs appearing in both lists rank higher than docs in one list
- `rrf_fuse()` returns results sorted by rrf_score descending
- `rrf_fuse()` correctly records `bm25_rank` and `dense_rank` (None when absent from a list)
- RRF score for a doc at rank 1 in both lists (k=60) equals `1/61 + 1/61 ≈ 0.0328`

**GREEN**: Implement `fusion.py`

### Phase 5: Candidate pair construction

**RED**: `test_ingest_pairs.py`
- `build_candidates()` excludes the known positive doc_id from results
- `build_candidates()` returns at most `candidates_per_query` pairs
- `build_candidates()` generates deterministic UUIDs (same input → same pair_id)
- `build_candidates()` sets `retrieval_method = "bm25+dense"` and correct `source_rank`
- `build_candidates()` sets `source_dataset = "nomic-ai/nomic-embed-unsupervised-data/{split}"` (full path, not just split name)
- `build_candidates()` with positive doc at rank 1 still returns `candidates_per_query` candidates (skips positive, continues)

**GREEN**: Implement `pairs.py`

### Phase 6: Postgres writer

**RED**: `test_ingest_writer.py` (requires test database — reuse `db_pool` fixture from main conftest)
- `write_pairs()` inserts pairs into the `pairs` table with `status = 'unlabeled'`
- `write_pairs()` with duplicate pair_ids (re-run) doesn't fail (ON CONFLICT DO NOTHING)
- `write_pairs()` with duplicate pair_ids doesn't overwrite existing data
- inserted pairs have correct `source_dataset`, `retrieval_method`, `source_rank`

**GREEN**: Implement `writer.py`

### Phase 7: CLI integration

**RED**: `test_ingest_cli.py`
- `--dry-run` flag processes but doesn't write to database
- `--split` is required (missing → error)
- `--max-docs 10` limits the corpus size
- `--embedding-model` overrides the default model in config
- `--bm25-top-k` and `--dense-top-k` flow through to retrieval
- `--candidates-per-query` controls output size
- full pipeline with tiny fixture: load → index → retrieve → fuse → build → write

**GREEN**: Implement `__main__.py`

---

## 9. Postgres workflow: local dev → remote production

### Local development (recommended starting point)

```bash
# Start local Postgres (from kombinat repo root)
docker compose up -d postgres

# Apply schema migrations
dbmate up

# Verify tables exist
psql postgresql://kombinat:kombinat@localhost:5432/kombinat -c "\dt"
# → contributors, pairs, batches, batch_pairs, annotations, honeypots

# Run ingest against local DB
uv run python -m kombinat.tools.ingest --split squad --dry-run
uv run python -m kombinat.tools.ingest --split squad  # real write
```

### When you're ready for production

You don't migrate data from local to remote. The pipeline is deterministic — same input produces identical output. The workflow is:

1. Develop and validate locally on the squad split
2. When satisfied, spin up managed Postgres (Supabase, Railway, Neon)
3. Run `dbmate up` against the remote URL to create tables
4. Re-run the ingest tool pointing at the remote database

```bash
# Apply migrations to remote
DATABASE_URL="postgresql://postgres:xxx@db.abc123.supabase.co:5432/postgres" dbmate up

# Re-run ingest against remote (identical output, deterministic UUIDs)
uv run python -m kombinat.tools.ingest \
    --split squad \
    --database-url "postgresql://postgres:xxx@db.abc123.supabase.co:5432/postgres"
```

The local Postgres is throwaway. You can `docker compose down -v` at any time. The source of truth is the ingest tool + the source dataset on HuggingFace — the data is always reproducible.

---

## 10. Running it for real

### Development run (verify pipeline works)

```bash
# Squad is the smallest split (25K rows) — perfect for verifying the pipeline
uv run python -m kombinat.tools.ingest \
    --split squad \
    --max-docs 1000 \
    --candidates-per-query 100 \
    --dry-run
```

### First real run (smallest split end-to-end)

```bash
# Full squad split — small enough to finish in minutes
uv run python -m kombinat.tools.ingest --split squad
```

This produces ~125M candidate pairs at default settings (25K queries × 5K candidates). If that's too many for a first run, dial it back:

```bash
uv run python -m kombinat.tools.ingest --split squad --candidates-per-query 500
```

### Swapping the embedding model

```bash
# Use a larger model if you want better retrieval quality
uv run python -m kombinat.tools.ingest \
    --split squad \
    --embedding-model all-mpnet-base-v2

# Or a multilingual model
uv run python -m kombinat.tools.ingest \
    --split squad \
    --embedding-model paraphrase-multilingual-MiniLM-L12-v2
```

Any model on HuggingFace that works with `sentence-transformers` can be passed here. The dimension is detected automatically from the model.

### Recommended split order

Start small, validate the pipeline, then scale up.

| Priority | Split | Rows | Why |
|----------|-------|------|-----|
| 1 | squad | 25.1K | Smallest, fast iteration, good for pipeline validation |
| 2 | quora | 44.9K | Small, duplicate-detection pairs, interesting edge cases |
| 3 | gooaq | 1.28M | Google autocomplete Q&A, diverse, still manageable |
| 4 | wikipedia | 6.2M | High-quality factual content |
| 5 | paq | 53.9M | Clean Q&A pairs, well-structured, large scale |
| 6 | s2orc_title_abstract | 36.1M | Academic domain, different distribution |
| 7 | reddit_title_body | 66.2M | Largest, noisy, do last |

### Estimated timing on M3 MacBook Pro

For squad split (25K docs, 25K queries, **brute force** — nprobe=nlist):
- Document embedding: ~5 seconds
- BM25 index build: ~2 seconds
- FAISS IVF training + add: ~3 seconds
- Query retrieval (25K queries × BM25 top-1K + FAISS top-1K): ~1 minute
- RRF + pair construction: ~30 seconds
- Postgres write (up to 250K pairs at 10/query): ~10 seconds
- Total: ~2 minutes

For a 1M document split (e.g. gooaq, FAISS searches ~100K per query):
- Document embedding: ~3.5 minutes
- FAISS IVF training: ~2 minutes
- Query retrieval: ~3 hours
- Total: ~4 hours

Retrieval depth is the bottleneck. The defaults (1K per method, top-10 kept) are sized to the annotation budget: every kept candidate needs `required_annotations` (2) labels from contributors, so candidates-per-query multiplies the entire labeling workload by 2N. Depth beyond the fused top-N comes from re-mining with the improved embedding model in later training cycles, not from deeper static candidate lists. For large corpora, `--faiss-min-search-docs 200000` searches deeper at the cost of ~2× slower FAISS queries.

### Resume and re-run safety

The deterministic `pair_id = uuid5(query + doc_id + source_dataset_label)` combined with `ON CONFLICT DO NOTHING` means you can:
- Kill the process mid-run and restart — already-written pairs are skipped
- Re-run the same split with different parameters — existing pairs preserved, new ones added
- Run overlapping splits — the `source_dataset` includes the split name (e.g. `.../squad` vs `.../paq`), so the same document text in two splits gets different pair_ids

---

## 11. What this does NOT do

- **No positive pair insertion**: The ingest tool only writes *candidate* pairs (potential negatives). The known positives from the source dataset are not written to the pairs table — they don't need annotation. They'll be used directly in training as positive examples.
- **No labeling**: That's the annotator's job.
- **No quality assessment of candidates**: We don't filter by retrieval score threshold. Even low-scored candidates might be interesting edge cases. The annotators decide.
- **No cross-split retrieval**: Each split's index is independent. A reddit query doesn't retrieve wikipedia documents. This is intentional — hard negatives are most valuable within domain.
