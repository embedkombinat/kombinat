from __future__ import annotations

from unittest.mock import patch

import pytest

from kombinat.tools.ingest.config import IngestConfig
from kombinat.tools.ingest.source import Corpus, load_split


def _make_hf_rows(docs: list[str], queries: list[str]) -> list[dict[str, str]]:
    """Simulate HuggingFace dataset rows."""
    assert len(docs) == len(queries)
    return [
        {"query": q, "document": d, "dataset": "squad", "shard": 0}
        for q, d in zip(queries, docs, strict=True)
    ]


@pytest.fixture
def base_config() -> IngestConfig:
    return IngestConfig(split="squad", dataset_name="nomic-ai/nomic-embed-unsupervised-data")


def test_load_split_returns_corpus(base_config: IngestConfig) -> None:
    rows = _make_hf_rows(
        ["Doc about Paris.", "Doc about Berlin."],
        ["What is Paris?", "What is Berlin?"],
    )
    with patch("kombinat.tools.ingest.source.load_dataset", return_value=rows):
        corpus = load_split(base_config)
    assert isinstance(corpus, Corpus)


def test_load_split_deduplicates_documents(base_config: IngestConfig) -> None:
    duplicate_doc = "Paris is the capital of France."
    rows = _make_hf_rows(
        [duplicate_doc, duplicate_doc, "Berlin is in Germany."],
        ["Capital of France?", "Largest city in France?", "Capital of Germany?"],
    )
    with patch("kombinat.tools.ingest.source.load_dataset", return_value=rows):
        corpus = load_split(base_config)
    assert len(corpus.doc_texts) == 2
    assert len(corpus.doc_ids) == 2
    assert len(set(corpus.doc_ids)) == len(corpus.doc_ids)


def test_load_split_doc_id_to_idx_correct(base_config: IngestConfig) -> None:
    rows = _make_hf_rows(["Doc A.", "Doc B.", "Doc C."], ["Q1", "Q2", "Q3"])
    with patch("kombinat.tools.ingest.source.load_dataset", return_value=rows):
        corpus = load_split(base_config)
    for doc_id, idx in corpus.doc_id_to_idx.items():
        assert corpus.doc_ids[idx] == doc_id
        assert doc_id in corpus.doc_ids


def test_load_split_queries_and_positives_are_parallel(base_config: IngestConfig) -> None:
    rows = _make_hf_rows(["Doc A.", "Doc B.", "Doc C."], ["Q1", "Q2", "Q3"])
    with patch("kombinat.tools.ingest.source.load_dataset", return_value=rows):
        corpus = load_split(base_config)
    assert len(corpus.queries) == len(corpus.positive_doc_ids)
    assert len(corpus.queries) == 3


def test_load_split_positive_doc_ids_are_in_corpus(base_config: IngestConfig) -> None:
    rows = _make_hf_rows(["Doc A.", "Doc B."], ["Q1", "Q2"])
    with patch("kombinat.tools.ingest.source.load_dataset", return_value=rows):
        corpus = load_split(base_config)
    for pos_id in corpus.positive_doc_ids:
        assert pos_id in corpus.doc_ids


def test_load_split_max_docs_limits_rows(base_config: IngestConfig) -> None:
    config = base_config.model_copy(update={"max_docs": 2})
    rows = _make_hf_rows(
        ["Doc A.", "Doc B.", "Doc C.", "Doc D."],
        ["Q1", "Q2", "Q3", "Q4"],
    )
    with patch("kombinat.tools.ingest.source.load_dataset", return_value=rows):
        corpus = load_split(config)
    assert len(corpus.queries) == 2


def test_load_split_max_docs_deduplicates_within_limit(base_config: IngestConfig) -> None:
    # 2 rows with the same doc → 1 unique doc, 2 queries
    config = base_config.model_copy(update={"max_docs": 2})
    rows = _make_hf_rows(
        ["Same doc.", "Same doc.", "Other doc."],
        ["Q1", "Q2", "Q3"],
    )
    with patch("kombinat.tools.ingest.source.load_dataset", return_value=rows):
        corpus = load_split(config)
    assert len(corpus.queries) == 2
    assert len(corpus.doc_texts) == 1  # deduped


def test_load_split_split_field_set(base_config: IngestConfig) -> None:
    rows = _make_hf_rows(["Doc A."], ["Q1"])
    with patch("kombinat.tools.ingest.source.load_dataset", return_value=rows):
        corpus = load_split(base_config)
    assert corpus.split == "squad"
