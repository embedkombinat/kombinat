from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import asyncpg
    from httpx import AsyncClient


async def test_stats_returns_correct_counts(
    client: AsyncClient,
    db_pool: asyncpg.Pool,
) -> None:
    """GET /v1/stats returns correct counts after seeding data."""
    # Seed 10 pairs: 7 unlabeled, 2 verified, 1 rejected
    for i in range(10):
        pair_id = uuid.uuid5(uuid.NAMESPACE_DNS, f"stats_q{i}:stats_d{i}:test")
        status = "unlabeled"
        if i < 2:
            status = "verified"
        elif i == 2:
            status = "rejected"
        await db_pool.execute(
            """INSERT INTO pairs (id, query_text, doc_id, doc_text,
            source_dataset, retrieval_method, status)
            VALUES ($1, $2, $3, $4, $5, $6, $7)""",
            pair_id,
            f"q{i}",
            f"d{i}",
            f"doc{i}",
            "test",
            "bm25",
            status,
        )

    # Seed 2 contributors
    for i in range(2):
        await db_pool.execute(
            """INSERT INTO contributors
            (github_id, github_username, total_input_tokens, total_output_tokens)
            VALUES ($1, $2, $3, $4)""",
            i + 1000,
            f"stats_user_{i}",
            5000 * (i + 1),
            500 * (i + 1),
        )

    resp = await client.get("/v1/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_pairs"] == 10
    assert data["unlabeled_pairs"] == 7
    assert data["verified_pairs"] == 2
    assert data["rejected_pairs"] == 1
    assert data["total_contributors"] == 2
    assert data["total_input_tokens"] == 15000  # 5000 + 10000
    assert data["total_output_tokens"] == 1500  # 500 + 1000


async def test_stats_no_auth_required(client: AsyncClient) -> None:
    """GET /v1/stats requires no authentication."""
    resp = await client.get("/v1/stats")
    assert resp.status_code == 200


async def test_stats_active_contributors_24h(
    client: AsyncClient,
    db_pool: asyncpg.Pool,
) -> None:
    """active_contributors_24h only counts recent activity."""
    # Active contributor (seen recently)
    await db_pool.execute(
        """INSERT INTO contributors (github_id, github_username, last_seen_at)
        VALUES ($1, $2, $3)""",
        9001,
        "active_user",
        datetime.now(tz=UTC),
    )
    # Inactive contributor (seen 3 days ago)
    await db_pool.execute(
        """INSERT INTO contributors (github_id, github_username, last_seen_at)
        VALUES ($1, $2, $3)""",
        9002,
        "inactive_user",
        datetime.now(tz=UTC) - timedelta(days=3),
    )

    resp = await client.get("/v1/stats")
    data = resp.json()
    assert data["active_contributors_24h"] == 1
    assert data["total_contributors"] == 2
