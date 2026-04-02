from __future__ import annotations

import asyncpg

from kombinat.tools.ingest.pairs import CandidatePair

_INSERT_SQL = """
    INSERT INTO pairs (id, query_text, doc_id, doc_text,
                       source_dataset, retrieval_method,
                       source_rank, status)
    VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, 'unlabeled')
    ON CONFLICT (id) DO NOTHING
"""


def _row(p: CandidatePair) -> tuple:
    return (p.pair_id, p.query_text, p.doc_id, p.doc_text, p.source_dataset, p.retrieval_method, p.source_rank)


async def write_batch(conn: asyncpg.Connection, pairs: list[CandidatePair]) -> int:
    """Write one batch of pairs using an existing connection. Returns len(pairs)."""
    await conn.executemany(_INSERT_SQL, [_row(p) for p in pairs])
    return len(pairs)


async def write_pairs(
    pairs: list[CandidatePair],
    database_url: str,
    batch_size: int = 5000,
) -> int:
    """Write candidate pairs to the pairs table. Returns count inserted (approximate).

    Uses ON CONFLICT DO NOTHING for idempotent re-runs.
    """
    conn = await asyncpg.connect(database_url)
    try:
        inserted = 0
        for i in range(0, len(pairs), batch_size):
            inserted += await write_batch(conn, pairs[i : i + batch_size])
        return inserted
    finally:
        await conn.close()
