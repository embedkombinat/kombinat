from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import asyncpg
    from httpx import AsyncClient


async def test_claim_batch_returns_pairs(
    authed_client: AsyncClient,
    seeded_pairs: list[dict[str, Any]],
) -> None:
    """POST /v1/batches/claim returns 201 with batch_id, expires_at, and pairs."""
    resp = await authed_client.post("/v1/batches/claim", json={"size": 10})
    assert resp.status_code == 201
    data = resp.json()
    assert "batch_id" in data
    assert "expires_at" in data
    assert len(data["pairs"]) == 10

    # expires_at should be ~24h from now
    expires = datetime.fromisoformat(data["expires_at"])
    now = datetime.now(tz=UTC)
    assert timedelta(hours=23) < (expires - now) < timedelta(hours=25)


async def test_claimed_pairs_not_reclaimable(
    authed_client: AsyncClient,
    authed_client_b: AsyncClient,
    seeded_pairs: list[dict[str, Any]],
) -> None:
    """Claimed pairs are not returned in a subsequent claim by another contributor."""
    resp1 = await authed_client.post("/v1/batches/claim", json={"size": 10})
    assert resp1.status_code == 201
    ids1 = {p["pair_id"] for p in resp1.json()["pairs"]}

    resp2 = await authed_client_b.post("/v1/batches/claim", json={"size": 10})
    assert resp2.status_code == 201
    ids2 = {p["pair_id"] for p in resp2.json()["pairs"]}

    assert ids1.isdisjoint(ids2)


async def test_claim_no_pairs_returns_204(
    authed_client: AsyncClient,
) -> None:
    """When no unlabeled pairs exist, returns 204 No Content."""
    resp = await authed_client.post("/v1/batches/claim", json={"size": 10})
    assert resp.status_code == 204


async def test_expired_batch_pairs_reclaimable(
    authed_client: AsyncClient,
    seeded_pairs: list[dict[str, Any]],
    db_pool: asyncpg.Pool,
) -> None:
    """After a batch expires, the same contributor can reclaim those pairs."""
    resp1 = await authed_client.post("/v1/batches/claim", json={"size": 10})
    assert resp1.status_code == 201
    batch_id = resp1.json()["batch_id"]
    # Force-expire the batch
    await db_pool.execute(
        """UPDATE batches SET status = 'expired',
        expires_at = NOW() - interval '1 hour' WHERE id = $1""",
        uuid.UUID(batch_id),
    )

    # Should be able to reclaim
    resp2 = await authed_client.post("/v1/batches/claim", json={"size": 10})
    assert resp2.status_code == 201
    ids2 = {p["pair_id"] for p in resp2.json()["pairs"]}
    # At least some of the original pairs should be available again
    assert len(ids2) == 10


async def test_annotated_pair_not_reclaimable(
    authed_client: AsyncClient,
    contributor: dict[str, Any],
    seeded_pairs: list[dict[str, Any]],
    db_pool: asyncpg.Pool,
) -> None:
    """A contributor never receives a pair they have already annotated."""
    resp1 = await authed_client.post("/v1/batches/claim", json={"size": 5})
    assert resp1.status_code == 201
    batch_id = uuid.UUID(resp1.json()["batch_id"])
    first_pair_id = uuid.UUID(resp1.json()["pairs"][0]["pair_id"])

    # Manually insert an annotation for the first pair
    await db_pool.execute(
        """INSERT INTO annotations
        (pair_id, contributor_id, batch_id, label, model_id, quantization,
         input_tokens, output_tokens, raw_response_hash)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)""",
        first_pair_id,
        contributor["id"],
        batch_id,
        2,
        "test-model",
        "Q8_0",
        100,
        10,
        "sha256:abc",
    )

    # Expire the batch so pairs are available again
    await db_pool.execute("UPDATE batches SET status = 'expired' WHERE id = $1", batch_id)

    # Claim again — should NOT include the annotated pair
    resp2 = await authed_client.post("/v1/batches/claim", json={"size": 50})
    assert resp2.status_code == 201
    ids2 = {p["pair_id"] for p in resp2.json()["pairs"]}
    assert str(first_pair_id) not in ids2


async def test_delete_batch_releases_pairs(
    authed_client: AsyncClient,
    authed_client_b: AsyncClient,
    seeded_pairs: list[dict[str, Any]],
) -> None:
    """DELETE /v1/batches/{id} releases pairs back to pool."""
    resp1 = await authed_client.post("/v1/batches/claim", json={"size": 10})
    assert resp1.status_code == 201
    batch_id = resp1.json()["batch_id"]

    # Release the batch
    del_resp = await authed_client.delete(f"/v1/batches/{batch_id}")
    assert del_resp.status_code == 204

    # Another contributor should now be able to claim those pairs
    resp2 = await authed_client_b.post("/v1/batches/claim", json={"size": 50})
    assert resp2.status_code == 201
    assert len(resp2.json()["pairs"]) == 50  # all 50 seeded pairs available


async def test_cannot_delete_others_batch(
    authed_client: AsyncClient,
    authed_client_b: AsyncClient,
    seeded_pairs: list[dict[str, Any]],
) -> None:
    """Cannot delete another contributor's batch."""
    resp = await authed_client.post("/v1/batches/claim", json={"size": 5})
    assert resp.status_code == 201
    batch_id = resp.json()["batch_id"]

    # Try to delete from a different contributor
    del_resp = await authed_client_b.delete(f"/v1/batches/{batch_id}")
    assert del_resp.status_code == 403
