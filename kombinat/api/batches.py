from __future__ import annotations

import math
import uuid
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, HTTPException, Response

from kombinat.config import get_settings
from kombinat.dependencies import get_current_contributor, get_db
from kombinat.schemas.batches import BatchClaimRequest, BatchOut
from kombinat.schemas.pairs import PairBrief

if TYPE_CHECKING:
    import asyncpg

router = APIRouter(tags=["batches"])


@router.post(
    "/batches/claim",
    response_model=BatchOut,
    status_code=201,
    summary="Claim a batch of unlabeled pairs",
    responses={
        204: {"description": "No pairs available"},
        401: {"description": "Not authenticated"},
    },
)
async def claim_batch(
    body: BatchClaimRequest,
    response: Response,
    contributor: dict[str, Any] = Depends(get_current_contributor),  # noqa: B008
    db: asyncpg.Pool = Depends(get_db),  # noqa: B008
) -> BatchOut | Response:
    """Claim a batch of unlabeled pairs for annotation."""
    settings = get_settings()
    requested_size = min(body.size, settings.batch_max_size)
    contributor_id = contributor["id"]

    # Calculate honeypot count
    honeypot_count = math.ceil(requested_size * settings.honeypot_ratio)
    regular_count = requested_size - honeypot_count

    async with db.acquire() as conn, conn.transaction():
        # Claim regular (non-honeypot) pairs
        regular_rows = await conn.fetch(
            """WITH claimable AS (
                    SELECT p.id FROM pairs p
                    WHERE p.status = 'unlabeled'
                      AND (SELECT COUNT(*) FROM annotations a WHERE a.pair_id = p.id)
                          < p.required_annotations
                      AND NOT EXISTS (
                          SELECT 1 FROM annotations a
                          WHERE a.pair_id = p.id AND a.contributor_id = $2
                      )
                      AND NOT EXISTS (
                          SELECT 1 FROM batch_pairs bp
                          JOIN batches b ON b.id = bp.batch_id
                          WHERE bp.pair_id = p.id AND b.status = 'assigned'
                      )
                      AND NOT EXISTS (
                          SELECT 1 FROM honeypots h WHERE h.pair_id = p.id
                      )
                    ORDER BY p.created_at
                    LIMIT $1
                    FOR UPDATE OF p SKIP LOCKED
                )
                SELECT p.* FROM pairs p JOIN claimable c ON p.id = c.id""",
            regular_count,
            contributor_id,
        )

        # Claim honeypot pairs
        honeypot_rows = await conn.fetch(
            """WITH claimable_hp AS (
                    SELECT p.id FROM pairs p
                    JOIN honeypots h ON h.pair_id = p.id
                    WHERE p.status = 'unlabeled'
                      AND NOT EXISTS (
                          SELECT 1 FROM annotations a
                          WHERE a.pair_id = p.id AND a.contributor_id = $2
                      )
                      AND NOT EXISTS (
                          SELECT 1 FROM batch_pairs bp
                          JOIN batches b ON b.id = bp.batch_id
                          WHERE bp.pair_id = p.id AND b.status = 'assigned'
                      )
                    ORDER BY p.created_at
                    LIMIT $1
                    FOR UPDATE OF p SKIP LOCKED
                )
                SELECT p.* FROM pairs p JOIN claimable_hp c ON p.id = c.id""",
            honeypot_count,
            contributor_id,
        )

        import random

        all_rows = list(regular_rows) + list(honeypot_rows)
        random.shuffle(all_rows)

        # If fewer honeypots than desired, fill remainder with regular pairs
        shortfall = requested_size - len(all_rows)
        if shortfall > 0:
            claimed_ids = {row["id"] for row in all_rows}
            extra_rows = await conn.fetch(
                """WITH claimable_extra AS (
                        SELECT p.id FROM pairs p
                        WHERE p.status = 'unlabeled'
                          AND (SELECT COUNT(*) FROM annotations a WHERE a.pair_id = p.id)
                              < p.required_annotations
                          AND NOT EXISTS (
                              SELECT 1 FROM annotations a
                              WHERE a.pair_id = p.id AND a.contributor_id = $2
                          )
                          AND NOT EXISTS (
                              SELECT 1 FROM batch_pairs bp
                              JOIN batches b ON b.id = bp.batch_id
                              WHERE bp.pair_id = p.id AND b.status = 'assigned'
                          )
                          AND NOT EXISTS (
                              SELECT 1 FROM honeypots h WHERE h.pair_id = p.id
                          )
                          AND p.id != ALL($3::uuid[])
                        ORDER BY p.created_at
                        LIMIT $1
                        FOR UPDATE OF p SKIP LOCKED
                    )
                    SELECT p.* FROM pairs p JOIN claimable_extra c ON p.id = c.id""",
                shortfall,
                contributor_id,
                list(claimed_ids),
            )
            all_rows.extend(extra_rows)

        if not all_rows:
            return Response(status_code=204)

        # Create batch
        expires_at = datetime.now(tz=UTC) + timedelta(hours=settings.batch_expiry_hours)
        batch_id = uuid.uuid4()
        await conn.execute(
            """INSERT INTO batches (id, contributor_id, size, status, expires_at)
                VALUES ($1, $2, $3, 'assigned', $4)""",
            batch_id,
            contributor_id,
            len(all_rows),
            expires_at,
        )

        # Link pairs to batch
        for row in all_rows:
            await conn.execute(
                "INSERT INTO batch_pairs (batch_id, pair_id) VALUES ($1, $2)",
                batch_id,
                row["id"],
            )

    pairs = [
        PairBrief(
            pair_id=row["id"],
            query_text=row["query_text"],
            doc_text=row["doc_text"],
            source_dataset=row["source_dataset"],
            is_honeypot=False,  # always false in response
        )
        for row in all_rows
    ]

    return BatchOut(batch_id=batch_id, expires_at=expires_at, pairs=pairs)


@router.delete(
    "/batches/{batch_id}",
    status_code=204,
    summary="Release a batch early",
    responses={
        403: {"description": "Not the batch owner"},
        404: {"description": "Batch not found"},
    },
)
async def delete_batch(
    batch_id: uuid.UUID,
    contributor: dict[str, Any] = Depends(get_current_contributor),  # noqa: B008
    db: asyncpg.Pool = Depends(get_db),  # noqa: B008
) -> Response:
    """Release a batch early, returning pairs to the available pool."""
    row = await db.fetchrow("SELECT * FROM batches WHERE id = $1", batch_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found")

    if row["contributor_id"] != contributor["id"]:
        raise HTTPException(
            status_code=403,
            detail=f"Batch {batch_id} is owned by another contributor",
        )

    if row["status"] != "assigned":
        # Already released or completed — treat as success (idempotent)
        if row["status"] in ("expired", "completed"):
            return Response(status_code=204)
        raise HTTPException(
            status_code=400,
            detail=f"Batch {batch_id} is in '{row['status']}' status, expected 'assigned'",
        )

    await db.execute("UPDATE batches SET status = 'expired' WHERE id = $1", batch_id)
    return Response(status_code=204)
