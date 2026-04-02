from __future__ import annotations

import pytest

from kombinat.tools.ingest.bm25 import BM25Index, bm25_retrieve, build_bm25_index
from kombinat.tools.ingest.source import Corpus


def test_build_bm25_index_returns_bm25_index(tiny_corpus: Corpus) -> None:
    index = build_bm25_index(tiny_corpus)
    assert isinstance(index, BM25Index)


def test_bm25_retrieve_returns_paris_in_top3(tiny_corpus: Corpus) -> None:
    index = build_bm25_index(tiny_corpus)
    results = bm25_retrieve(index, "capital of France", top_k=3)
    doc_ids = [did for did, _ in results]
    # Paris doc should be in top-3
    import hashlib

    paris_id = hashlib.sha256("Paris is the capital of France and its largest city.".encode()).hexdigest()[:16]
    assert paris_id in doc_ids


def test_bm25_retrieve_sorted_descending(tiny_corpus: Corpus) -> None:
    index = build_bm25_index(tiny_corpus)
    results = bm25_retrieve(index, "capital France Berlin", top_k=5)
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


def test_bm25_retrieve_top_k_limits_results(tiny_corpus: Corpus) -> None:
    index = build_bm25_index(tiny_corpus)
    results = bm25_retrieve(index, "capital", top_k=2)
    assert len(results) == 2


def test_bm25_retrieve_top_k_larger_than_corpus(tiny_corpus: Corpus) -> None:
    index = build_bm25_index(tiny_corpus)
    results = bm25_retrieve(index, "capital", top_k=100)
    # Should return at most len(corpus) results
    assert len(results) <= len(tiny_corpus.doc_ids)


def test_bm25_retrieve_nonsense_query_returns_results(tiny_corpus: Corpus) -> None:
    index = build_bm25_index(tiny_corpus)
    results = bm25_retrieve(index, "xyzzy frobnicator flibbertigibbet", top_k=3)
    # BM25 always returns results (even if all scores are 0)
    assert len(results) > 0


def test_bm25_retrieve_doc_ids_in_corpus(tiny_corpus: Corpus) -> None:
    index = build_bm25_index(tiny_corpus)
    corpus_ids = set(tiny_corpus.doc_ids)
    results = bm25_retrieve(index, "machine learning AI", top_k=5)
    for doc_id, _ in results:
        assert doc_id in corpus_ids
