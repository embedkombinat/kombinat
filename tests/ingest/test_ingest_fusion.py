from __future__ import annotations

import pytest

from kombinat.tools.ingest.fusion import RankedCandidate, rrf_fuse


def test_rrf_fuse_identical_rankings_preserves_order() -> None:
    ranking = [("doc_a", 1.0), ("doc_b", 0.8), ("doc_c", 0.5)]
    results = rrf_fuse(ranking, ranking, k=60)
    doc_ids = [r.doc_id for r in results]
    assert doc_ids == ["doc_a", "doc_b", "doc_c"]


def test_rrf_fuse_identical_rankings_doubles_scores() -> None:
    ranking = [("doc_a", 1.0), ("doc_b", 0.8)]
    single = rrf_fuse(ranking, [], k=60)
    double = rrf_fuse(ranking, ranking, k=60)
    for s, d in zip(single, double):
        assert abs(d.rrf_score - 2 * s.rrf_score) < 1e-9


def test_rrf_fuse_disjoint_rankings_merges_all() -> None:
    bm25 = [("doc_a", 1.0), ("doc_b", 0.8)]
    dense = [("doc_c", 0.9), ("doc_d", 0.7)]
    results = rrf_fuse(bm25, dense, k=60)
    doc_ids = {r.doc_id for r in results}
    assert doc_ids == {"doc_a", "doc_b", "doc_c", "doc_d"}


def test_rrf_fuse_overlapping_ranks_higher() -> None:
    # doc_a appears in both lists at rank 1; doc_b only in bm25; doc_c only in dense
    bm25 = [("doc_a", 1.0), ("doc_b", 0.8)]
    dense = [("doc_a", 0.9), ("doc_c", 0.7)]
    results = rrf_fuse(bm25, dense, k=60)
    top = results[0]
    assert top.doc_id == "doc_a"


def test_rrf_fuse_sorted_descending() -> None:
    bm25 = [("doc_a", 1.0), ("doc_b", 0.8), ("doc_c", 0.5)]
    dense = [("doc_c", 0.9), ("doc_b", 0.6), ("doc_a", 0.3)]
    results = rrf_fuse(bm25, dense, k=60)
    scores = [r.rrf_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_rrf_fuse_bm25_rank_none_when_absent() -> None:
    bm25: list[tuple[str, float]] = []
    dense = [("doc_a", 0.9)]
    results = rrf_fuse(bm25, dense, k=60)
    assert results[0].bm25_rank is None
    assert results[0].dense_rank == 1


def test_rrf_fuse_dense_rank_none_when_absent() -> None:
    bm25 = [("doc_a", 1.0)]
    dense: list[tuple[str, float]] = []
    results = rrf_fuse(bm25, dense, k=60)
    assert results[0].dense_rank is None
    assert results[0].bm25_rank == 1


def test_rrf_fuse_score_at_rank1_in_both() -> None:
    # doc_a rank 1 in both lists with k=60: 1/61 + 1/61
    bm25 = [("doc_a", 1.0)]
    dense = [("doc_a", 0.9)]
    results = rrf_fuse(bm25, dense, k=60)
    expected = 1.0 / 61 + 1.0 / 61
    assert abs(results[0].rrf_score - expected) < 1e-9


def test_rrf_fuse_returns_ranked_candidates() -> None:
    bm25 = [("doc_a", 1.0)]
    dense = [("doc_b", 0.9)]
    results = rrf_fuse(bm25, dense, k=60)
    for r in results:
        assert isinstance(r, RankedCandidate)
