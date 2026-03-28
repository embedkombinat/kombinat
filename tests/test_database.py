import uuid
from typing import Any

import asyncpg


async def test_tables_exist(db_pool: asyncpg.Pool) -> None:
    """Verify all 6 tables exist after migration."""
    rows = await db_pool.fetch(
        """SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'public'
        ORDER BY table_name"""
    )
    table_names = {row["table_name"] for row in rows}
    expected = {"contributors", "pairs", "batches", "batch_pairs", "annotations", "honeypots"}
    assert expected.issubset(table_names)


async def test_pairs_insert(db_pool: asyncpg.Pool) -> None:
    """Insert a pair row and read it back."""
    pair_id = uuid.uuid5(uuid.NAMESPACE_DNS, "test_query:test_doc:msmarco")
    await db_pool.execute(
        """INSERT INTO pairs (id, query_text, doc_id, doc_text, source_dataset, retrieval_method)
        VALUES ($1, $2, $3, $4, $5, $6)""",
        pair_id,
        "what is python",
        "doc_1",
        "Python is a programming language",
        "msmarco",
        "bm25",
    )
    row = await db_pool.fetchrow("SELECT * FROM pairs WHERE id = $1", pair_id)
    assert row is not None
    assert row["query_text"] == "what is python"
    assert row["status"] == "unlabeled"
    assert row["required_annotations"] == 2


async def test_contributors_insert(db_pool: asyncpg.Pool) -> None:
    """Insert a contributor and verify defaults."""
    row: dict[str, Any] | None = await db_pool.fetchrow(  # type: ignore[assignment]
        """INSERT INTO contributors (github_id, github_username)
        VALUES ($1, $2)
        RETURNING *""",
        12345,
        "testuser",
    )
    assert row is not None
    assert row["reputation_score"] == 0.5
    assert row["total_annotations"] == 0
    assert row["total_input_tokens"] == 0
    assert row["total_output_tokens"] == 0
