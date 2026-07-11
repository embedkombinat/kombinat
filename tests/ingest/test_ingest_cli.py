from __future__ import annotations

import hashlib
import sys
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kombinat.tools.ingest.pairs import CandidatePair
from kombinat.tools.ingest.source import Corpus

if TYPE_CHECKING:
    from kombinat.tools.ingest.config import IngestConfig


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


DOCS = [f"Doc {i}." for i in range(3)]
DOC_IDS = [_sha(d) for d in DOCS]

FAKE_CORPUS = Corpus(
    doc_ids=DOC_IDS,
    doc_texts=DOCS,
    queries=["Query 1", "Query 2"],
    positive_doc_ids=[DOC_IDS[0], DOC_IDS[1]],
    split="squad",
    doc_id_to_idx={did: i for i, did in enumerate(DOC_IDS)},
)

FAKE_PAIR = CandidatePair(
    pair_id="00000000-0000-0000-0000-000000000000",
    query_text="Query 1",
    doc_id=DOC_IDS[2],
    doc_text=DOCS[2],
    source_dataset="nomic-ai/nomic-embed-unsupervised-data/squad",
    retrieval_method="bm25+dense",
    source_rank=1,
)


def _run_cli(argv: list[str]) -> None:
    """Run the CLI entry point with given argv."""
    import asyncio

    with patch.object(sys, "argv", ["kombinat.tools.ingest"] + argv):
        from kombinat.tools.ingest.__main__ import main

        asyncio.run(main())


def test_cli_missing_split_exits() -> None:
    with pytest.raises(SystemExit):
        _run_cli([])


def test_cli_dry_run_does_not_write(capsys: pytest.CaptureFixture[str]) -> None:
    import numpy as np

    mock_load = MagicMock(return_value=FAKE_CORPUS)
    mock_bm25_build = MagicMock()
    mock_bm25_retrieve = MagicMock(return_value=[])
    mock_dense_build = MagicMock(return_value=MagicMock(nlist=1))
    mock_dense_retrieve = MagicMock(return_value=[])
    mock_embed_queries = MagicMock(return_value=np.zeros((2, 8), dtype="float32"))
    mock_rrf = MagicMock(return_value=[])
    mock_build_candidates = MagicMock(return_value=[FAKE_PAIR])
    mock_write = AsyncMock(return_value=0)

    with (
        patch("kombinat.tools.ingest.__main__.load_split", mock_load),
        patch("kombinat.tools.ingest.__main__.build_bm25_index", mock_bm25_build),
        patch("kombinat.tools.ingest.__main__.bm25_retrieve", mock_bm25_retrieve),
        patch("kombinat.tools.ingest.__main__.build_dense_index", mock_dense_build),
        patch("kombinat.tools.ingest.__main__.dense_retrieve", mock_dense_retrieve),
        patch("kombinat.tools.ingest.__main__.embed_queries", mock_embed_queries),
        patch("kombinat.tools.ingest.__main__.rrf_fuse", mock_rrf),
        patch("kombinat.tools.ingest.__main__.build_candidates", mock_build_candidates),
        patch("kombinat.tools.ingest.__main__.write_batch", mock_write),
    ):
        _run_cli(["--split", "squad", "--dry-run"])

    mock_write.assert_not_called()


def test_cli_max_docs_flows_to_config() -> None:
    import numpy as np

    captured_config: list[IngestConfig] = []

    def fake_load(config: IngestConfig) -> Corpus:
        captured_config.append(config)
        return FAKE_CORPUS

    mock_bm25_build = MagicMock()
    mock_bm25_retrieve = MagicMock(return_value=[])
    mock_dense_build = MagicMock(return_value=MagicMock(nlist=1))
    mock_dense_retrieve = MagicMock(return_value=[])
    mock_embed = MagicMock(return_value=np.zeros((2, 8), dtype="float32"))
    mock_rrf = MagicMock(return_value=[])
    mock_candidates = MagicMock(return_value=[])

    with (
        patch("kombinat.tools.ingest.__main__.load_split", fake_load),
        patch("kombinat.tools.ingest.__main__.build_bm25_index", mock_bm25_build),
        patch("kombinat.tools.ingest.__main__.bm25_retrieve", mock_bm25_retrieve),
        patch("kombinat.tools.ingest.__main__.build_dense_index", mock_dense_build),
        patch("kombinat.tools.ingest.__main__.dense_retrieve", mock_dense_retrieve),
        patch("kombinat.tools.ingest.__main__.embed_queries", mock_embed),
        patch("kombinat.tools.ingest.__main__.rrf_fuse", mock_rrf),
        patch("kombinat.tools.ingest.__main__.build_candidates", mock_candidates),
    ):
        _run_cli(["--split", "squad", "--max-docs", "10", "--dry-run"])

    assert captured_config[0].max_docs == 10


def test_cli_embedding_model_overrides_default() -> None:
    import numpy as np

    captured_config: list[IngestConfig] = []

    def fake_load(config: IngestConfig) -> Corpus:
        captured_config.append(config)
        return FAKE_CORPUS

    mock_bm25_build = MagicMock()
    mock_bm25_retrieve = MagicMock(return_value=[])
    mock_dense_build = MagicMock(return_value=MagicMock(nlist=1))
    mock_dense_retrieve = MagicMock(return_value=[])
    mock_embed = MagicMock(return_value=np.zeros((2, 8), dtype="float32"))
    mock_rrf = MagicMock(return_value=[])
    mock_candidates = MagicMock(return_value=[])

    with (
        patch("kombinat.tools.ingest.__main__.load_split", fake_load),
        patch("kombinat.tools.ingest.__main__.build_bm25_index", mock_bm25_build),
        patch("kombinat.tools.ingest.__main__.bm25_retrieve", mock_bm25_retrieve),
        patch("kombinat.tools.ingest.__main__.build_dense_index", mock_dense_build),
        patch("kombinat.tools.ingest.__main__.dense_retrieve", mock_dense_retrieve),
        patch("kombinat.tools.ingest.__main__.embed_queries", mock_embed),
        patch("kombinat.tools.ingest.__main__.rrf_fuse", mock_rrf),
        patch("kombinat.tools.ingest.__main__.build_candidates", mock_candidates),
    ):
        _run_cli(["--split", "squad", "--embedding-model", "all-mpnet-base-v2", "--dry-run"])

    assert captured_config[0].embedding_model == "all-mpnet-base-v2"


def test_cli_retrieval_params_flow_through() -> None:
    import numpy as np

    captured_config: list[IngestConfig] = []

    def fake_load(config: IngestConfig) -> Corpus:
        captured_config.append(config)
        return FAKE_CORPUS

    mock_bm25_build = MagicMock()
    mock_bm25_retrieve = MagicMock(return_value=[])
    mock_dense_build = MagicMock(return_value=MagicMock(nlist=1))
    mock_dense_retrieve = MagicMock(return_value=[])
    mock_embed = MagicMock(return_value=np.zeros((2, 8), dtype="float32"))
    mock_rrf = MagicMock(return_value=[])
    mock_candidates = MagicMock(return_value=[])

    with (
        patch("kombinat.tools.ingest.__main__.load_split", fake_load),
        patch("kombinat.tools.ingest.__main__.build_bm25_index", mock_bm25_build),
        patch("kombinat.tools.ingest.__main__.bm25_retrieve", mock_bm25_retrieve),
        patch("kombinat.tools.ingest.__main__.build_dense_index", mock_dense_build),
        patch("kombinat.tools.ingest.__main__.dense_retrieve", mock_dense_retrieve),
        patch("kombinat.tools.ingest.__main__.embed_queries", mock_embed),
        patch("kombinat.tools.ingest.__main__.rrf_fuse", mock_rrf),
        patch("kombinat.tools.ingest.__main__.build_candidates", mock_candidates),
    ):
        _run_cli(
            [
                "--split",
                "squad",
                "--bm25-top-k",
                "500",
                "--dense-top-k",
                "300",
                "--candidates-per-query",
                "50",
                "--dry-run",
            ]
        )

    cfg = captured_config[0]
    assert cfg.bm25_top_k == 500
    assert cfg.dense_top_k == 300
    assert cfg.candidates_per_query == 50


def test_default_candidate_budget_is_small() -> None:
    """The per-query candidate budget multiplies the whole annotation workload
    by 2N (required_annotations=2 per pair). Guard against it silently creeping
    back to depths no contributor fleet can label (5000/query = ~400M pairs
    from the squad split alone)."""
    from kombinat.tools.ingest.config import IngestConfig

    cfg = IngestConfig(split="squad")
    assert cfg.candidates_per_query <= 50
    assert cfg.bm25_top_k <= 2_000
    assert cfg.dense_top_k <= 2_000
