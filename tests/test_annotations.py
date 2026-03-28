from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import asyncpg
    from httpx import AsyncClient


async def _claim_and_get_pairs(
    client: AsyncClient, size: int = 10
) -> tuple[str, list[dict[str, Any]]]:
    """Helper: claim a batch and return (batch_id, pairs)."""
    resp = await client.post("/v1/batches/claim", json={"size": size})
    assert resp.status_code == 201
    data = resp.json()
    return data["batch_id"], data["pairs"]


def _make_annotations(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build annotation payloads for all pairs in a batch."""
    return [
        {
            "pair_id": p["pair_id"],
            "label": 2,
            "input_tokens": 100,
            "output_tokens": 10,
            "raw_response_hash": f"sha256:{p['pair_id'][:8]}",
        }
        for p in pairs
    ]


async def test_submit_annotations_accepted(
    authed_client: AsyncClient,
    seeded_pairs: list[dict[str, Any]],
) -> None:
    """POST /v1/annotations with valid batch returns accepted count."""
    batch_id, pairs = await _claim_and_get_pairs(authed_client, size=5)
    annotations = _make_annotations(pairs)
    resp = await authed_client.post(
        "/v1/annotations",
        json={
            "batch_id": batch_id,
            "model_id": "Qwen2.5-7B-Instruct",
            "quantization": "Q8_0",
            "annotations": annotations,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["accepted"] == 5
    assert data["rejected"] == 0


async def test_submit_updates_contributor_tokens(
    authed_client: AsyncClient,
    seeded_pairs: list[dict[str, Any]],
    db_pool: asyncpg.Pool,
    contributor: dict[str, Any],
) -> None:
    """Submission updates contributor token totals."""
    batch_id, pairs = await _claim_and_get_pairs(authed_client, size=3)
    annotations = _make_annotations(pairs)
    await authed_client.post(
        "/v1/annotations",
        json={
            "batch_id": batch_id,
            "model_id": "test-model",
            "quantization": "FP16",
            "annotations": annotations,
        },
    )
    row = await db_pool.fetchrow("SELECT * FROM contributors WHERE id = $1", contributor["id"])
    assert row is not None
    assert row["total_input_tokens"] == 300  # 3 * 100
    assert row["total_output_tokens"] == 30  # 3 * 10
    assert row["total_annotations"] == 3


async def test_submit_wrong_batch_owner(
    authed_client: AsyncClient,
    authed_client_b: AsyncClient,
    seeded_pairs: list[dict[str, Any]],
) -> None:
    """Cannot submit annotations for another contributor's batch."""
    batch_id, pairs = await _claim_and_get_pairs(authed_client, size=3)
    annotations = _make_annotations(pairs)
    resp = await authed_client_b.post(
        "/v1/annotations",
        json={
            "batch_id": batch_id,
            "model_id": "test-model",
            "quantization": "Q8_0",
            "annotations": annotations,
        },
    )
    assert resp.status_code == 403


async def test_submit_expired_batch(
    authed_client: AsyncClient,
    seeded_pairs: list[dict[str, Any]],
    db_pool: asyncpg.Pool,
) -> None:
    """Cannot submit annotations for an expired batch."""
    batch_id, pairs = await _claim_and_get_pairs(authed_client, size=3)
    # Force-expire
    await db_pool.execute(
        "UPDATE batches SET status = 'expired' WHERE id = $1",
        uuid.UUID(batch_id),
    )
    annotations = _make_annotations(pairs)
    resp = await authed_client.post(
        "/v1/annotations",
        json={
            "batch_id": batch_id,
            "model_id": "test-model",
            "quantization": "Q8_0",
            "annotations": annotations,
        },
    )
    assert resp.status_code == 400


async def test_submit_duplicate_pair(
    authed_client: AsyncClient,
    seeded_pairs: list[dict[str, Any]],
) -> None:
    """Duplicate pair_id in submission is rejected."""
    batch_id, pairs = await _claim_and_get_pairs(authed_client, size=3)
    annotations = _make_annotations(pairs)
    # Duplicate the first annotation
    annotations.append(annotations[0].copy())
    resp = await authed_client.post(
        "/v1/annotations",
        json={
            "batch_id": batch_id,
            "model_id": "test-model",
            "quantization": "Q8_0",
            "annotations": annotations,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["accepted"] == 3
    assert data["rejected"] == 1


async def test_submit_persists_model_metadata(
    authed_client: AsyncClient,
    seeded_pairs: list[dict[str, Any]],
    db_pool: asyncpg.Pool,
) -> None:
    """model_id and quantization are stored in annotations table."""
    batch_id, pairs = await _claim_and_get_pairs(authed_client, size=2)
    annotations = _make_annotations(pairs)
    await authed_client.post(
        "/v1/annotations",
        json={
            "batch_id": batch_id,
            "model_id": "Qwen2.5-7B-Instruct",
            "quantization": "Q8_0",
            "annotations": annotations,
        },
    )
    rows = await db_pool.fetch(
        "SELECT model_id, quantization FROM annotations WHERE batch_id = $1",
        uuid.UUID(batch_id),
    )
    assert len(rows) == 2
    for row in rows:
        assert row["model_id"] == "Qwen2.5-7B-Instruct"
        assert row["quantization"] == "Q8_0"
