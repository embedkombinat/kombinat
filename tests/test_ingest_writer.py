from __future__ import annotations

import hashlib
from uuid import NAMESPACE_URL, uuid5

import asyncpg
import pytest

from kombinat.tools.ingest.pairs import CandidatePair
from kombinat.tools.ingest.writer import write_batch, write_pairs

# Tests here need the real DB (db_pool from conftest.py).
pytestmark = pytest.mark.usefixtures("clean_tables")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _make_pair(n: int) -> CandidatePair:
    query = f"test query {n}"
    doc_id = _sha(f"document {n}")
    source_dataset = "nomic-ai/nomic-embed-unsupervised-data/squad"
    pair_id = str(uuid5(NAMESPACE_URL, f"{query}|{doc_id}|{source_dataset}"))
    return CandidatePair(
        pair_id=pair_id,
        query_text=query,
        doc_id=doc_id,
        doc_text=f"document {n}",
        source_dataset=source_dataset,
        retrieval_method="bm25+dense",
        source_rank=n + 1,
    )


async def test_write_pairs_inserts_rows(db_pool: asyncpg.Pool) -> None:
    pairs = [_make_pair(i) for i in range(5)]
    from tests.conftest import TEST_DSN

    count = await write_pairs(pairs, TEST_DSN, batch_size=10)
    assert count == 5

    rows = await db_pool.fetch("SELECT * FROM pairs WHERE source_dataset = $1", pairs[0].source_dataset)
    assert len(rows) == 5


async def test_write_pairs_status_is_unlabeled(db_pool: asyncpg.Pool) -> None:
    pairs = [_make_pair(0)]
    from tests.conftest import TEST_DSN

    await write_pairs(pairs, TEST_DSN, batch_size=10)
    row = await db_pool.fetchrow("SELECT status FROM pairs WHERE id = $1::uuid", pairs[0].pair_id)
    assert row is not None
    assert row["status"] == "unlabeled"


async def test_write_pairs_idempotent_on_conflict(db_pool: asyncpg.Pool) -> None:
    pairs = [_make_pair(0)]
    from tests.conftest import TEST_DSN

    await write_pairs(pairs, TEST_DSN, batch_size=10)
    # Second write — should not raise
    await write_pairs(pairs, TEST_DSN, batch_size=10)

    rows = await db_pool.fetch("SELECT * FROM pairs WHERE id = $1::uuid", pairs[0].pair_id)
    assert len(rows) == 1


async def test_write_pairs_does_not_overwrite_existing(db_pool: asyncpg.Pool) -> None:
    original = _make_pair(0)
    from tests.conftest import TEST_DSN

    await write_pairs([original], TEST_DSN, batch_size=10)

    # Overwrite attempt with different doc_text but same pair_id
    tampered = CandidatePair(
        pair_id=original.pair_id,
        query_text=original.query_text,
        doc_id=original.doc_id,
        doc_text="TAMPERED TEXT",
        source_dataset=original.source_dataset,
        retrieval_method=original.retrieval_method,
        source_rank=original.source_rank,
    )
    await write_pairs([tampered], TEST_DSN, batch_size=10)

    row = await db_pool.fetchrow("SELECT doc_text FROM pairs WHERE id = $1::uuid", original.pair_id)
    assert row is not None
    assert row["doc_text"] == original.doc_text


async def test_write_pairs_correct_fields(db_pool: asyncpg.Pool) -> None:
    pair = _make_pair(7)
    from tests.conftest import TEST_DSN

    await write_pairs([pair], TEST_DSN, batch_size=10)
    row = await db_pool.fetchrow("SELECT * FROM pairs WHERE id = $1::uuid", pair.pair_id)
    assert row is not None
    assert row["source_dataset"] == pair.source_dataset
    assert row["retrieval_method"] == pair.retrieval_method
    assert row["source_rank"] == pair.source_rank
    assert row["doc_id"] == pair.doc_id


async def test_write_batch_inserts_using_existing_conn(db_pool: asyncpg.Pool) -> None:
    pairs = [_make_pair(i) for i in range(3)]
    from tests.conftest import TEST_DSN

    conn = await asyncpg.connect(TEST_DSN)
    try:
        count = await write_batch(conn, pairs)
    finally:
        await conn.close()

    assert count == 3
    rows = await db_pool.fetch("SELECT id FROM pairs WHERE source_dataset = $1", pairs[0].source_dataset)
    assert len(rows) == 3
