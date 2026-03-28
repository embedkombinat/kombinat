from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import asyncpg

from kombinat.validator.checks import anomaly_check, honeypot_check
from kombinat.validator.promote import maybe_promote_pair
from kombinat.validator.reputation import update_reputation


async def test_honeypot_check_correct_label(
    db_pool: asyncpg.Pool,
    honeypot_pairs: list[dict[str, Any]],
) -> None:
    """Honeypot check passes when label matches known answer."""
    hp = honeypot_pairs[0]
    result = await honeypot_check(db_pool, hp["id"], hp["known_label"])
    assert result is True


async def test_honeypot_check_wrong_label(
    db_pool: asyncpg.Pool,
    honeypot_pairs: list[dict[str, Any]],
) -> None:
    """Honeypot check fails when label does not match."""
    hp = honeypot_pairs[0]
    wrong_label = (hp["known_label"] + 1) % 4
    result = await honeypot_check(db_pool, hp["id"], wrong_label)
    assert result is False


async def test_pair_promotion_on_sufficient_annotations(
    db_pool: asyncpg.Pool,
    seeded_pairs: list[dict[str, Any]],
) -> None:
    """Pair is promoted to 'verified' when required annotations are met with agreement."""
    pair = seeded_pairs[0]
    pair_id = pair["id"]

    # Insert a contributor for annotations
    c1 = await db_pool.fetchrow(
        """INSERT INTO contributors (github_id, github_username)
        VALUES ($1, $2) RETURNING *""",
        111,
        "validator_c1",
    )
    c2 = await db_pool.fetchrow(
        """INSERT INTO contributors (github_id, github_username)
        VALUES ($1, $2) RETURNING *""",
        222,
        "validator_c2",
    )
    assert c1 is not None and c2 is not None

    # Create two batches (one per contributor)
    b1_id = uuid.uuid4()
    b2_id = uuid.uuid4()
    for bid, cid in [(b1_id, c1["id"]), (b2_id, c2["id"])]:
        await db_pool.execute(
            """INSERT INTO batches (id, contributor_id, size, expires_at)
            VALUES ($1, $2, 1, NOW() + interval '24 hours')""",
            bid,
            cid,
        )
        await db_pool.execute(
            "INSERT INTO batch_pairs (batch_id, pair_id) VALUES ($1, $2)",
            bid,
            pair_id,
        )

    # Both annotators agree: label=2
    for cid, bid in [(c1["id"], b1_id), (c2["id"], b2_id)]:
        await db_pool.execute(
            """INSERT INTO annotations
            (pair_id, contributor_id, batch_id, label, model_id, quantization,
             input_tokens, output_tokens, raw_response_hash)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)""",
            pair_id,
            cid,
            bid,
            2,
            "test-model",
            "Q8_0",
            100,
            10,
            "sha256:test",
        )

    promoted = await maybe_promote_pair(db_pool, pair_id)
    assert promoted is True

    row = await db_pool.fetchrow("SELECT status FROM pairs WHERE id = $1", pair_id)
    assert row is not None
    assert row["status"] == "verified"


async def test_anomaly_check_always_passes(db_pool: asyncpg.Pool) -> None:
    """Stub anomaly_check always returns True."""
    result = await anomaly_check(
        db_pool,
        uuid.uuid4(),
        uuid.uuid4(),
        2,
    )
    assert result is True


async def test_reputation_update_noop(
    db_pool: asyncpg.Pool,
    contributor: dict[str, Any],
) -> None:
    """Stub update_reputation does not change the score."""
    original = contributor["reputation_score"]
    await update_reputation(db_pool, contributor["id"], [True, False, True])
    row = await db_pool.fetchrow(
        "SELECT reputation_score FROM contributors WHERE id = $1",
        contributor["id"],
    )
    assert row is not None
    assert row["reputation_score"] == original
