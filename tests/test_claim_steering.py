"""Judge-diversity claim steering: claims prefer pairs not yet labeled by
the requesting model's family, with graceful fallback when only same-family
pairs remain."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from kombinat.families import model_family

if TYPE_CHECKING:
    import asyncpg
    from httpx import AsyncClient


# The db_pool parameter is unused but required: the autouse clean_tables
# fixture resolves db_pool via getfixturevalue, which only works when the
# session-scoped pool is already in the test's fixture closure.
async def test_model_family_known_families_from_basename(db_pool: asyncpg.Pool) -> None:
    assert model_family("Qwen/Qwen2.5-7B-Instruct-AWQ") == "qwen"
    assert model_family("mlx-community/Qwen2.5-7B-Instruct-4bit") == "qwen"
    assert model_family("mistralai/Mistral-7B-Instruct-v0.3") == "mistral"
    assert model_family("microsoft/Phi-3.5-mini-instruct") == "phi"


async def test_model_family_unknown_and_empty(db_pool: asyncpg.Pool) -> None:
    assert model_family("some-org/UnknownNet-7B") is None
    assert model_family(None) is None
    assert model_family("") is None


async def test_model_family_word_boundaries(db_pool: asyncpg.Pool) -> None:
    """Families match only at word starts within the basename: 'phi' must not
    hit mid-word (dol-PHI-n), version suffixes still count, and an org name
    containing a keyword must not classify an unknown basename."""
    assert model_family("dphn/dolphin-2.9-llama3-8b") == "llama"
    assert model_family("Qwen/Qwen2.5-7B-Instruct") == "qwen"
    assert model_family("meta-llama/Llama-3.1-8B-Instruct") == "llama"
    assert model_family("microsoft/Phi-3.5-mini-instruct") == "phi"
    assert model_family("mistralai/SomethingElse-7B") is None


async def _annotate_pair(
    pool: asyncpg.Pool,
    pair_id: object,
    github_id: int,
    model_id: str,
) -> None:
    """Insert one annotation on a pair from a fresh contributor."""
    c = await pool.fetchrow(
        """INSERT INTO contributors (github_id, github_username)
        VALUES ($1, $2) RETURNING id""",
        github_id,
        f"steer_c{github_id}",
    )
    assert c is not None
    batch_id = uuid.uuid4()
    await pool.execute(
        """INSERT INTO batches (id, contributor_id, size, status, expires_at)
        VALUES ($1, $2, 1, 'completed', NOW() + interval '24 hours')""",
        batch_id,
        c["id"],
    )
    await pool.execute(
        "INSERT INTO batch_pairs (batch_id, pair_id) VALUES ($1, $2)",
        batch_id,
        pair_id,
    )
    await pool.execute(
        """INSERT INTO annotations
        (pair_id, contributor_id, batch_id, label, model_id, quantization,
         input_tokens, output_tokens, raw_response_hash)
        VALUES ($1, $2, $3, 2, $4, 'awq', 100, 10, 'sha256:steer')""",
        pair_id,
        c["id"],
        batch_id,
        model_id,
    )


async def test_claim_steers_away_from_same_family(
    authed_client: AsyncClient,
    seeded_pairs: list[dict[str, Any]],
    db_pool: asyncpg.Pool,
) -> None:
    """A qwen judge is not handed the pair a qwen model already labeled while
    unlabeled pairs remain (earliest-created pair would otherwise be first)."""
    await _annotate_pair(db_pool, seeded_pairs[0]["id"], 9001, "Qwen/Qwen2.5-7B-Instruct")

    resp = await authed_client.post(
        "/v1/batches/claim",
        json={"size": 1, "model_id": "mlx-community/Qwen2.5-3B-Instruct-4bit"},
    )
    assert resp.status_code == 201
    claimed = {p["pair_id"] for p in resp.json()["pairs"]}
    assert str(seeded_pairs[0]["id"]) not in claimed


async def test_claim_cross_family_gets_the_pair(
    authed_client: AsyncClient,
    seeded_pairs: list[dict[str, Any]],
    db_pool: asyncpg.Pool,
) -> None:
    """A mistral judge IS handed the qwen-labeled pair — that's the point:
    its opinion is worth the most there."""
    await _annotate_pair(db_pool, seeded_pairs[0]["id"], 9002, "Qwen/Qwen2.5-7B-Instruct")

    resp = await authed_client.post(
        "/v1/batches/claim",
        json={"size": 1, "model_id": "mistralai/Mistral-7B-Instruct-v0.3"},
    )
    assert resp.status_code == 201
    claimed = {p["pair_id"] for p in resp.json()["pairs"]}
    assert str(seeded_pairs[0]["id"]) in claimed


async def test_claim_falls_back_when_only_same_family_pairs_remain(
    authed_client: AsyncClient,
    seeded_pairs: list[dict[str, Any]],
    db_pool: asyncpg.Pool,
) -> None:
    """Steering must never starve a contributor: when every claimable pair
    already has a same-family annotation, the shortfall pass hands them out
    anyway."""
    for i, pair in enumerate(seeded_pairs):
        await _annotate_pair(db_pool, pair["id"], 9100 + i, "Qwen/Qwen2.5-7B-Instruct")

    resp = await authed_client.post(
        "/v1/batches/claim",
        json={"size": 10, "model_id": "Qwen/Qwen2.5-7B-Instruct-AWQ"},
    )
    assert resp.status_code == 201
    assert len(resp.json()["pairs"]) == 10


async def test_claim_without_model_id_is_unsteered(
    authed_client: AsyncClient,
    seeded_pairs: list[dict[str, Any]],
    db_pool: asyncpg.Pool,
) -> None:
    """Old clients that send no model_id keep the created_at ordering."""
    await _annotate_pair(db_pool, seeded_pairs[0]["id"], 9200, "Qwen/Qwen2.5-7B-Instruct")

    resp = await authed_client.post("/v1/batches/claim", json={"size": 1})
    assert resp.status_code == 201
    claimed = {p["pair_id"] for p in resp.json()["pairs"]}
    assert str(seeded_pairs[0]["id"]) in claimed


async def test_claim_with_unknown_family_is_unsteered(
    authed_client: AsyncClient,
    seeded_pairs: list[dict[str, Any]],
    db_pool: asyncpg.Pool,
) -> None:
    """Unrecognized model ids skip steering rather than guessing a family."""
    await _annotate_pair(db_pool, seeded_pairs[0]["id"], 9300, "Qwen/Qwen2.5-7B-Instruct")

    resp = await authed_client.post(
        "/v1/batches/claim",
        json={"size": 1, "model_id": "some-org/UnknownNet-7B"},
    )
    assert resp.status_code == 201
    claimed = {p["pair_id"] for p in resp.json()["pairs"]}
    assert str(seeded_pairs[0]["id"]) in claimed


async def test_sql_family_match_uses_word_boundaries(
    authed_client: AsyncClient,
    seeded_pairs: list[dict[str, Any]],
    db_pool: asyncpg.Pool,
) -> None:
    """The SQL side must agree with model_family(): a dolphin (llama-family)
    annotation is not a 'phi' match, and a phi judge is therefore still
    handed that pair."""
    await _annotate_pair(db_pool, seeded_pairs[0]["id"], 9400, "dphn/dolphin-2.9-llama3-8b")

    # phi judge: dolphin is not family phi -> earliest pair still claimable
    resp = await authed_client.post(
        "/v1/batches/claim",
        json={"size": 1, "model_id": "microsoft/Phi-3.5-mini-instruct"},
    )
    assert resp.status_code == 201
    claimed = {p["pair_id"] for p in resp.json()["pairs"]}
    assert str(seeded_pairs[0]["id"]) in claimed


async def test_sql_family_match_steers_llama_from_dolphin(
    authed_client: AsyncClient,
    seeded_pairs: list[dict[str, Any]],
    db_pool: asyncpg.Pool,
) -> None:
    """...and a llama judge IS steered away from the dolphin-labeled pair."""
    await _annotate_pair(db_pool, seeded_pairs[0]["id"], 9500, "dphn/dolphin-2.9-llama3-8b")

    resp = await authed_client.post(
        "/v1/batches/claim",
        json={"size": 1, "model_id": "meta-llama/Llama-3.1-8B-Instruct"},
    )
    assert resp.status_code == 201
    claimed = {p["pair_id"] for p in resp.json()["pairs"]}
    assert str(seeded_pairs[0]["id"]) not in claimed


async def test_sql_family_match_ignores_org_segment(
    authed_client: AsyncClient,
    seeded_pairs: list[dict[str, Any]],
    db_pool: asyncpg.Pool,
) -> None:
    """An org name containing a family keyword must not create a match when
    the basename has none."""
    await _annotate_pair(db_pool, seeded_pairs[0]["id"], 9600, "mistralai/UnknownNet-7B")

    resp = await authed_client.post(
        "/v1/batches/claim",
        json={"size": 1, "model_id": "mistralai/Mistral-7B-Instruct-v0.3"},
    )
    assert resp.status_code == 201
    claimed = {p["pair_id"] for p in resp.json()["pairs"]}
    assert str(seeded_pairs[0]["id"]) in claimed
