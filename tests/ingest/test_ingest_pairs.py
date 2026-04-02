from __future__ import annotations

import hashlib
from uuid import NAMESPACE_URL, uuid5

import pytest

from kombinat.tools.ingest.config import IngestConfig
from kombinat.tools.ingest.fusion import RankedCandidate
from kombinat.tools.ingest.pairs import CandidatePair, build_candidates
from kombinat.tools.ingest.source import Corpus


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


@pytest.fixture
def config() -> IngestConfig:
    return IngestConfig(split="squad", candidates_per_query=3)


@pytest.fixture
def small_corpus() -> Corpus:
    docs = [f"Document number {i}." for i in range(5)]
    doc_ids = [_sha(d) for d in docs]
    return Corpus(
        doc_ids=doc_ids,
        doc_texts=docs,
        queries=["query one"],
        positive_doc_ids=[doc_ids[0]],
        split="squad",
        doc_id_to_idx={did: i for i, did in enumerate(doc_ids)},
    )


def _rrf_results(corpus: Corpus) -> list[RankedCandidate]:
    """All docs ranked in order, no scores needed."""
    return [
        RankedCandidate(doc_id=did, rrf_score=1.0 / (i + 1), bm25_rank=i + 1, dense_rank=i + 1)
        for i, did in enumerate(corpus.doc_ids)
    ]


def test_build_candidates_excludes_positive(
    small_corpus: Corpus, config: IngestConfig
) -> None:
    rrf = _rrf_results(small_corpus)
    pos_id = small_corpus.positive_doc_ids[0]
    candidates = build_candidates("query one", pos_id, rrf, small_corpus, config)
    assert all(c.doc_id != pos_id for c in candidates)


def test_build_candidates_max_candidates_per_query(
    small_corpus: Corpus, config: IngestConfig
) -> None:
    rrf = _rrf_results(small_corpus)
    candidates = build_candidates("query one", small_corpus.positive_doc_ids[0], rrf, small_corpus, config)
    assert len(candidates) <= config.candidates_per_query


def test_build_candidates_deterministic_uuids(
    small_corpus: Corpus, config: IngestConfig
) -> None:
    rrf = _rrf_results(small_corpus)
    pos_id = small_corpus.positive_doc_ids[0]
    run1 = build_candidates("query one", pos_id, rrf, small_corpus, config)
    run2 = build_candidates("query one", pos_id, rrf, small_corpus, config)
    assert [c.pair_id for c in run1] == [c.pair_id for c in run2]


def test_build_candidates_uuid_formula(small_corpus: Corpus, config: IngestConfig) -> None:
    rrf = _rrf_results(small_corpus)
    pos_id = small_corpus.positive_doc_ids[0]
    candidates = build_candidates("query one", pos_id, rrf, small_corpus, config)
    source_dataset = config.source_dataset_label
    for c in candidates:
        expected_id = str(uuid5(NAMESPACE_URL, f"query one|{c.doc_id}|{source_dataset}"))
        assert c.pair_id == expected_id


def test_build_candidates_retrieval_method(small_corpus: Corpus, config: IngestConfig) -> None:
    rrf = _rrf_results(small_corpus)
    candidates = build_candidates("query one", small_corpus.positive_doc_ids[0], rrf, small_corpus, config)
    assert all(c.retrieval_method == "bm25+dense" for c in candidates)


def test_build_candidates_source_rank(small_corpus: Corpus, config: IngestConfig) -> None:
    rrf = _rrf_results(small_corpus)
    pos_id = small_corpus.positive_doc_ids[0]
    candidates = build_candidates("query one", pos_id, rrf, small_corpus, config)
    # source_rank should reflect position in rrf_results (skipping positive)
    for i, c in enumerate(candidates):
        assert c.source_rank >= 1


def test_build_candidates_source_dataset_full_path(
    small_corpus: Corpus, config: IngestConfig
) -> None:
    rrf = _rrf_results(small_corpus)
    candidates = build_candidates("query one", small_corpus.positive_doc_ids[0], rrf, small_corpus, config)
    expected = "nomic-ai/nomic-embed-unsupervised-data/squad"
    assert all(c.source_dataset == expected for c in candidates)


def test_build_candidates_positive_at_rank1_still_returns_max(
    small_corpus: Corpus, config: IngestConfig
) -> None:
    # positive doc is first in rrf results — should be skipped, still get max candidates
    rrf = _rrf_results(small_corpus)
    pos_id = small_corpus.doc_ids[0]  # positive is rank 1
    candidates = build_candidates("query one", pos_id, rrf, small_corpus, config)
    # corpus has 5 docs, positive skipped, so 4 available; config.candidates_per_query=3
    assert len(candidates) == 3
