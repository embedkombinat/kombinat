from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import asyncpg
    from httpx import AsyncClient

from kombinat.expiry import expire_batches


async def _create_expired_batch(
    db_pool: asyncpg.Pool,
    contributor_id: object,
    pair_ids: list[object],
) -> uuid.UUID:
    """Insert a batch that is already expired."""
    batch_id = uuid.uuid4()
    await db_pool.execute(
        """INSERT INTO batches (id, contributor_id, size, status, expires_at)
        VALUES ($1, $2, $3, 'assigned', NOW() - interval '1 hour')""",
        batch_id,
        contributor_id,
        len(pair_ids),
    )
    for pid in pair_ids:
        await db_pool.execute(
            "INSERT INTO batch_pairs (batch_id, pair_id) VALUES ($1, $2)",
            batch_id,
            pid,
        )
    return batch_id


async def test_expired_batches_marked(
    db_pool: asyncpg.Pool,
    contributor: dict[str, Any],
    seeded_pairs: list[dict[str, Any]],
) -> None:
    """Expired batches are marked as 'expired' by cleanup task."""
    pair_ids = [p["id"] for p in seeded_pairs[:5]]
    batch_id = await _create_expired_batch(db_pool, contributor["id"], pair_ids)

    count = await expire_batches(db_pool)
    assert count >= 1

    row = await db_pool.fetchrow("SELECT status FROM batches WHERE id = $1", batch_id)
    assert row is not None
    assert row["status"] == "expired"


async def test_expired_batch_pairs_reclaimable(
    authed_client: AsyncClient,
    db_pool: asyncpg.Pool,
    contributor: dict[str, Any],
    seeded_pairs: list[dict[str, Any]],
) -> None:
    """Pairs from expired batches are claimable again."""
    pair_ids = [p["id"] for p in seeded_pairs[:5]]
    await _create_expired_batch(db_pool, contributor["id"], pair_ids)

    # Run expiry
    await expire_batches(db_pool)

    # All 50 pairs should be claimable (the 5 expired ones + 45 never claimed)
    resp = await authed_client.post("/v1/batches/claim", json={"size": 50})
    assert resp.status_code == 201
    assert len(resp.json()["pairs"]) == 50
