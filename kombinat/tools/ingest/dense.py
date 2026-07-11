from __future__ import annotations

import json
import pathlib
from math import sqrt
from typing import TYPE_CHECKING, Any

import faiss  # type: ignore[import-untyped]
import numpy as np
from pydantic import BaseModel, ConfigDict
from sentence_transformers import SentenceTransformer

if TYPE_CHECKING:
    from kombinat.tools.ingest.config import IngestConfig
    from kombinat.tools.ingest.source import Corpus


class DenseIndex(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    index: faiss.Index
    doc_ids: list[str]
    dimension: int
    nlist: int
    # Model used to build the index. None when the index was loaded from disk cache,
    # in which case embed_queries() will load its own. Exposed so callers can reuse
    # the freshly-loaded model for query embedding and avoid a second model load.
    # Typed as Any because tests substitute SentenceTransformer with a MagicMock, which
    # would fail pydantic's isinstance check; the real type contract lives on embed_queries().
    model: Any = None


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


def _meta_path(config: IngestConfig) -> pathlib.Path:
    return pathlib.Path(config.faiss_index_dir).expanduser() / f"{_cache_key(config)}.meta.json"


def _cache_fingerprint(config: IngestConfig) -> dict[str, object]:
    """Build parameters that make a cached index compatible with the current run.

    A cache built with a different embedding model or normalization setting
    would silently degrade retrieval (e.g. normalized queries against a
    non-normalized corpus index), so loading checks this fingerprint and
    rebuilds on mismatch instead of trusting whatever is on disk.
    """
    return {"embedding_model": config.embedding_model, "normalized": True}


def build_dense_index(corpus: Corpus, config: IngestConfig) -> DenseIndex:
    """Embed all documents and build FAISS IVFFlat index.

    Saves index to disk. On re-run, loads from disk if the cached index
    exists AND its fingerprint (embedding model + normalization) matches;
    otherwise the index is rebuilt. Caches from before fingerprinting have
    no meta file and are rebuilt too.
    """
    idx_path = _index_path(config)
    ids_path = _doc_ids_path(config)
    meta_path = _meta_path(config)

    if idx_path.exists() and ids_path.exists():
        cached_meta: dict[str, object] | None = None
        if meta_path.exists():
            try:
                cached_meta = json.loads(meta_path.read_text())
            except json.JSONDecodeError:
                cached_meta = None
        if cached_meta == _cache_fingerprint(config):
            index = faiss.read_index(str(idx_path))
            doc_ids: list[str] = json.loads(ids_path.read_text())
            nlist = index.nlist if hasattr(index, "nlist") else 1
            dim = index.d
            nprobe = compute_nprobe(len(doc_ids), nlist, config.faiss_min_search_docs)
            index.nprobe = nprobe
            return DenseIndex(index=index, doc_ids=doc_ids, dimension=dim, nlist=nlist)

    model = SentenceTransformer(config.embedding_model, device=config.embedding_device)
    dim = model.get_sentence_embedding_dimension()

    vectors: np.ndarray = model.encode(
        corpus.doc_texts,
        batch_size=config.embedding_batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        # The index uses METRIC_INNER_PRODUCT; without unit-norm vectors that
        # ranks by raw dot product, which is biased toward longer documents
        # rather than cosine similarity.
        normalize_embeddings=True,
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
    meta_path.write_text(json.dumps(_cache_fingerprint(config)))

    return DenseIndex(
        index=index,
        doc_ids=corpus.doc_ids,
        dimension=dim,
        nlist=nlist,
        model=model,
    )


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
    for idx, score in zip(indices[0], scores[0], strict=False):
        if idx >= 0:
            results.append((index.doc_ids[idx], float(score)))
    return results


def embed_queries(
    queries: list[str],
    config: IngestConfig,
    model: SentenceTransformer | None = None,
) -> np.ndarray:
    """Embed a batch of queries. Returns (n, dim) float32 array.

    If `model` is provided, reuses it instead of loading a fresh SentenceTransformer —
    callers should pass `DenseIndex.model` when available to avoid double loading.
    """
    if model is None:
        model = SentenceTransformer(config.embedding_model, device=config.embedding_device)
    return model.encode(
        queries,
        batch_size=config.embedding_batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        # Must match the document encoding: unit-norm so inner product = cosine.
        normalize_embeddings=True,
    ).astype(np.float32)
