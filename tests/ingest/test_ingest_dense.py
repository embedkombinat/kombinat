from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from kombinat.tools.ingest.config import IngestConfig
from kombinat.tools.ingest.dense import (
    DenseIndex,
    build_dense_index,
    compute_nprobe,
    dense_retrieve,
    embed_queries,
)

if TYPE_CHECKING:
    import pathlib

    from kombinat.tools.ingest.source import Corpus

DIM = 8  # tiny dimension for fast unit tests


def _mock_encoder(dim: int = DIM) -> MagicMock:
    """Mock SentenceTransformer that returns random unit vectors."""
    rng = np.random.default_rng(42)

    def encode(texts: list[str], **kwargs: object) -> np.ndarray:
        vecs = rng.standard_normal((len(texts), dim)).astype(np.float32)
        # L2-normalize for inner-product search
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

    mock = MagicMock()
    mock.encode.side_effect = encode
    mock.get_sentence_embedding_dimension.return_value = dim
    return mock


@pytest.fixture
def ingest_config(tmp_faiss_dir: pathlib.Path) -> IngestConfig:
    return IngestConfig(
        split="squad",
        embedding_model="all-MiniLM-L6-v2",
        embedding_device="cpu",
        embedding_batch_size=8,
        faiss_index_dir=str(tmp_faiss_dir),
        faiss_min_search_docs=100_000,
    )


# ── compute_nprobe ──────────────────────────────────────────────────────────


def test_compute_nprobe_small_corpus_brute_force() -> None:
    # 5 docs, nlist=5, min_search=100K → brute force (nprobe == nlist)
    nprobe = compute_nprobe(n_docs=5, nlist=5, min_search_docs=100_000)
    assert nprobe == 5


def test_compute_nprobe_large_corpus_limits_nprobe() -> None:
    # 1M docs, nlist=4000, min_search=100K → nprobe = ~100 (not 4000)
    nprobe = compute_nprobe(n_docs=1_000_000, nlist=4000, min_search_docs=100_000)
    assert nprobe < 4000
    assert nprobe >= 1


def test_compute_nprobe_never_exceeds_nlist() -> None:
    nprobe = compute_nprobe(n_docs=500, nlist=50, min_search_docs=100_000)
    assert nprobe <= 50


# ── build_dense_index ───────────────────────────────────────────────────────


def test_build_dense_index_returns_dense_index(
    tiny_corpus: Corpus, ingest_config: IngestConfig
) -> None:
    mock_enc = _mock_encoder()
    with patch("kombinat.tools.ingest.dense.SentenceTransformer", return_value=mock_enc):
        index = build_dense_index(tiny_corpus, ingest_config)
    assert isinstance(index, DenseIndex)


def test_build_dense_index_correct_dimension(
    tiny_corpus: Corpus, ingest_config: IngestConfig
) -> None:
    mock_enc = _mock_encoder(DIM)
    with patch("kombinat.tools.ingest.dense.SentenceTransformer", return_value=mock_enc):
        index = build_dense_index(tiny_corpus, ingest_config)
    assert index.dimension == DIM


def test_build_dense_index_saves_to_disk(
    tiny_corpus: Corpus, ingest_config: IngestConfig, tmp_faiss_dir: pathlib.Path
) -> None:
    mock_enc = _mock_encoder()
    with patch("kombinat.tools.ingest.dense.SentenceTransformer", return_value=mock_enc):
        build_dense_index(tiny_corpus, ingest_config)
    assert (tmp_faiss_dir / "squad.index").exists()
    assert (tmp_faiss_dir / "squad.doc_ids.json").exists()


def test_build_dense_index_loads_from_disk_on_rerun(
    tiny_corpus: Corpus, ingest_config: IngestConfig
) -> None:
    mock_enc = _mock_encoder()
    with patch("kombinat.tools.ingest.dense.SentenceTransformer", return_value=mock_enc):
        # First build
        build_dense_index(tiny_corpus, ingest_config)
        # Second build — encoder should NOT be called again
        mock_enc.encode.reset_mock()
        index2 = build_dense_index(tiny_corpus, ingest_config)
    mock_enc.encode.assert_not_called()
    assert isinstance(index2, DenseIndex)


# ── dense_retrieve ──────────────────────────────────────────────────────────


def test_dense_retrieve_returns_top_k(tiny_corpus: Corpus, ingest_config: IngestConfig) -> None:
    mock_enc = _mock_encoder()
    with patch("kombinat.tools.ingest.dense.SentenceTransformer", return_value=mock_enc):
        index = build_dense_index(tiny_corpus, ingest_config)
    rng = np.random.default_rng(0)
    query_vec = rng.standard_normal(DIM).astype(np.float32)
    query_vec /= np.linalg.norm(query_vec)
    results = dense_retrieve(index, query_vec, top_k=3)
    assert len(results) == 3


def test_dense_retrieve_doc_ids_in_corpus(tiny_corpus: Corpus, ingest_config: IngestConfig) -> None:
    mock_enc = _mock_encoder()
    with patch("kombinat.tools.ingest.dense.SentenceTransformer", return_value=mock_enc):
        index = build_dense_index(tiny_corpus, ingest_config)
    rng = np.random.default_rng(1)
    query_vec = rng.standard_normal(DIM).astype(np.float32)
    query_vec /= np.linalg.norm(query_vec)
    results = dense_retrieve(index, query_vec, top_k=5)
    corpus_ids = set(tiny_corpus.doc_ids)
    for doc_id, _ in results:
        assert doc_id in corpus_ids


# ── embed_queries ────────────────────────────────────────────────────────────


def test_embed_queries_shape(ingest_config: IngestConfig) -> None:
    mock_enc = _mock_encoder(DIM)
    with patch("kombinat.tools.ingest.dense.SentenceTransformer", return_value=mock_enc):
        embeddings = embed_queries(["Q1", "Q2", "Q3"], ingest_config)
    assert embeddings.shape == (3, DIM)


def test_embed_queries_float32(ingest_config: IngestConfig) -> None:
    mock_enc = _mock_encoder(DIM)
    with patch("kombinat.tools.ingest.dense.SentenceTransformer", return_value=mock_enc):
        embeddings = embed_queries(["Q1"], ingest_config)
    assert embeddings.dtype == np.float32


# ── slow tests (real model) ─────────────────────────────────────────────────


@pytest.mark.slow
def test_build_dense_index_real_model_paris_in_top3(
    tiny_corpus: Corpus, ingest_config: IngestConfig
) -> None:
    import hashlib

    real_config = ingest_config.model_copy(update={"embedding_model": "all-MiniLM-L6-v2"})
    index = build_dense_index(tiny_corpus, real_config)
    query_vecs = embed_queries(["What is the capital of France?"], real_config)
    results = dense_retrieve(index, query_vecs[0], top_k=3)
    paris_id = hashlib.sha256(b"Paris is the capital of France and its largest city.").hexdigest()[
        :16
    ]
    doc_ids = [did for did, _ in results]
    assert paris_id in doc_ids
