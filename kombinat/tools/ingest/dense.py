from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from math import sqrt

import faiss  # type: ignore[import-untyped]
import numpy as np
from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

from kombinat.tools.ingest.config import IngestConfig
from kombinat.tools.ingest.source import Corpus


@dataclass
class DenseIndex:
    index: faiss.Index
    doc_ids: list[str]
    dimension: int
    nlist: int


def compute_nprobe(n_docs: int, nlist: int, min_search_docs: int = 100_000) -> int:
    """Compute nprobe so we search at least min_search_docs documents.

    For small corpora (n_docs <= min_search_docs), returns nlist (brute force).
    For larger corpora, searches ~min_search_docs documents worth of cells.
    """
    if n_docs <= min_search_docs:
        return nlist
    docs_per_cell = n_docs / nlist
    return max(1, min(nlist, int(min_search_docs / docs_per_cell)))


def _cache_key(config: IngestConfig) -> str:
    suffix = f"_{config.max_docs}" if config.max_docs is not None else ""
    return f"{config.split}{suffix}"


def _index_path(config: IngestConfig) -> pathlib.Path:
    return pathlib.Path(config.faiss_index_dir).expanduser() / f"{_cache_key(config)}.index"


def _doc_ids_path(config: IngestConfig) -> pathlib.Path:
    return pathlib.Path(config.faiss_index_dir).expanduser() / f"{_cache_key(config)}.doc_ids.json"


def build_dense_index(corpus: Corpus, config: IngestConfig) -> DenseIndex:
    """Embed all documents and build FAISS IVFFlat index.

    Saves index to disk. On re-run, loads from disk if file already exists.
    """
    idx_path = _index_path(config)
    ids_path = _doc_ids_path(config)

    if idx_path.exists() and ids_path.exists():
        index = faiss.read_index(str(idx_path))
        doc_ids: list[str] = json.loads(ids_path.read_text())
        nlist = index.nlist if hasattr(index, "nlist") else 1
        dim = index.d
        nprobe = compute_nprobe(len(doc_ids), nlist, config.faiss_min_search_docs)
        index.nprobe = nprobe
        return DenseIndex(index=index, doc_ids=doc_ids, dimension=dim, nlist=nlist)

    model = SentenceTransformer(config.embedding_model, device=config.embedding_device)
    dim: int = model.get_sentence_embedding_dimension()

    # Encode all documents in batches
    vectors: np.ndarray = model.encode(
        corpus.doc_texts,
        batch_size=config.embedding_batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    ).astype(np.float32)

    n = len(corpus.doc_texts)
    nlist = max(1, min(4 * int(sqrt(n)), n // 40))

    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(vectors)
    index.add(vectors)

    nprobe = compute_nprobe(n, nlist, config.faiss_min_search_docs)
    index.nprobe = nprobe

    idx_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(idx_path))
    ids_path.write_text(json.dumps(corpus.doc_ids))

    return DenseIndex(index=index, doc_ids=corpus.doc_ids, dimension=dim, nlist=nlist)


def dense_retrieve(
    index: DenseIndex,
    query_embedding: np.ndarray,
    top_k: int,
) -> list[tuple[str, float]]:
    """Return [(doc_id, score), ...] sorted by score descending."""
    vec = query_embedding.reshape(1, -1).astype(np.float32)
    n_results = min(top_k, len(index.doc_ids))
    scores, indices = index.index.search(vec, n_results)
    results: list[tuple[str, float]] = []
    for idx, score in zip(indices[0], scores[0]):
        if idx >= 0:
            results.append((index.doc_ids[idx], float(score)))
    return results


def embed_queries(queries: list[str], config: IngestConfig) -> np.ndarray:
    """Embed a batch of queries. Returns (n, dim) float32 array."""
    model = SentenceTransformer(config.embedding_model, device=config.embedding_device)
    return model.encode(
        queries,
        batch_size=config.embedding_batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    ).astype(np.float32)
