from __future__ import annotations

from dataclasses import dataclass


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
    Ranks are 1-based; the RRF formula is 1/(k + rank).
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
