from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, HTTPException

from kombinat.dependencies import get_current_contributor, get_db
from kombinat.schemas.annotations import AnnotationResult, AnnotationSubmission
from kombinat.validator.promote import maybe_promote_pair
from kombinat.validator.reputation import update_reputation

if TYPE_CHECKING:
    import uuid

    import asyncpg
    from asyncpg import Pool

logger = logging.getLogger(__name__)

router = APIRouter(tags=["annotations"])


@router.post(
    "/annotations",
    response_model=AnnotationResult,
    status_code=200,
    summary="Submit labels for a batch",
    responses={
        400: {"description": "Batch expired or invalid"},
        403: {"description": "Not the batch owner"},
    },
)
async def submit_annotations(
    body: AnnotationSubmission,
    contributor: dict[str, Any] = Depends(get_current_contributor),  # noqa: B008
    db: asyncpg.Pool = Depends(get_db),  # noqa: B008
) -> AnnotationResult:
    """Submit annotation labels for a claimed batch.

    The whole submission runs in a single transaction: annotation inserts,
    contributor totals, pair promotion, and batch completion either all
    persist or none do. Duplicates are absorbed with ON CONFLICT DO NOTHING
    (an exception inside the transaction would abort it).
    """
    contributor_id = contributor["id"]

    # Validate batch (reads only — safe outside the write transaction)
    batch = await db.fetchrow("SELECT * FROM batches WHERE id = $1", body.batch_id)
    if batch is None:
        logger.warning(
            "Batch not found: batch_id=%s contributor_id=%s", body.batch_id, contributor_id
        )
        raise HTTPException(status_code=404, detail=f"Batch {body.batch_id} not found")
    if batch["contributor_id"] != contributor_id:
        logger.warning(
            "Batch owner mismatch: batch_id=%s batch_owner=%s request_contributor=%s",
            body.batch_id,
            batch["contributor_id"],
            contributor_id,
        )
        raise HTTPException(
            status_code=403,
            detail=f"Batch {body.batch_id} is owned by another contributor",
        )
    if batch["status"] != "assigned":
        # Idempotent retry: if the batch is already completed by this contributor,
        # return the result from the existing annotations instead of erroring.
        if batch["status"] == "completed":
            logger.info(
                "Idempotent retry: batch %s already completed by contributor %s",
                body.batch_id,
                contributor_id,
            )
            return await _build_result_from_existing(db, body.batch_id, contributor_id)

        logger.warning(
            "Batch status invalid for submission: batch_id=%s status=%s contributor_id=%s",
            body.batch_id,
            batch["status"],
            contributor_id,
        )
        raise HTTPException(
            status_code=400,
            detail=(
                f"Batch {body.batch_id} is in '{batch['status']}' status, expected 'assigned'. "
                f"This batch may have expired (24h TTL) before annotations were submitted."
            ),
        )

    accepted = 0
    rejected = 0
    total_input = 0
    total_output = 0
    seen_pair_ids: set[uuid.UUID] = set()
    annotated_pair_ids: list[uuid.UUID] = []
    honeypot_results: list[bool] = []
    pairs_verified = 0

    async with db.acquire() as conn, conn.transaction():
        # Get valid pair_ids for this batch
        batch_pair_rows = await conn.fetch(
            "SELECT pair_id FROM batch_pairs WHERE batch_id = $1", body.batch_id
        )
        valid_pair_ids = {row["pair_id"] for row in batch_pair_rows}

        # Prefetch honeypot labels for the whole batch in one query
        hp_rows = await conn.fetch(
            "SELECT pair_id, known_label FROM honeypots WHERE pair_id = ANY($1::uuid[])",
            list(valid_pair_ids),
        )
        honeypot_labels = {row["pair_id"]: int(row["known_label"]) for row in hp_rows}

        for ann in body.annotations:
            # Skip duplicates within this submission
            if ann.pair_id in seen_pair_ids:
                rejected += 1
                continue
            seen_pair_ids.add(ann.pair_id)

            # Skip pairs not in this batch
            if ann.pair_id not in valid_pair_ids:
                rejected += 1
                continue

            is_honeypot = ann.pair_id in honeypot_labels

            status = await conn.execute(
                """INSERT INTO annotations
                (pair_id, contributor_id, batch_id, label, model_id, quantization,
                 input_tokens, output_tokens, raw_response_hash, is_honeypot)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ON CONFLICT (pair_id, contributor_id) DO NOTHING""",
                ann.pair_id,
                contributor_id,
                body.batch_id,
                ann.label,
                body.model_id,
                body.quantization,
                ann.input_tokens,
                ann.output_tokens,
                ann.raw_response_hash,
                is_honeypot,
            )
            if status.split()[-1] == "0":
                # Conflict: this contributor already annotated the pair
                rejected += 1
                continue

            accepted += 1
            total_input += ann.input_tokens
            total_output += ann.output_tokens
            annotated_pair_ids.append(ann.pair_id)

            if is_honeypot:
                honeypot_results.append(honeypot_labels[ann.pair_id] == ann.label)

        # Update contributor totals
        await conn.execute(
            """UPDATE contributors SET
                total_annotations = total_annotations + $2,
                total_input_tokens = total_input_tokens + $3,
                total_output_tokens = total_output_tokens + $4
            WHERE id = $1""",
            contributor_id,
            accepted,
            total_input,
            total_output,
        )

        # Promote pairs that have reached required annotations (honeypots never promote)
        for pair_id in annotated_pair_ids:
            if pair_id in honeypot_labels:
                continue
            promoted = await maybe_promote_pair(conn, pair_id)
            if promoted:
                pairs_verified += 1

        # Update reputation (stub: no-op in v0)
        await update_reputation(conn, contributor_id, honeypot_results)

        # Mark batch as completed only if at least one annotation was accepted
        if accepted > 0:
            await conn.execute(
                "UPDATE batches SET status = 'completed', completed_at = NOW() WHERE id = $1",
                body.batch_id,
            )

        # Reload contributor for response
        updated = await conn.fetchrow(
            "SELECT total_input_tokens, total_output_tokens FROM contributors WHERE id = $1",
            contributor_id,
        )

    if updated is None:
        raise HTTPException(status_code=500, detail="Contributor record missing after update")

    # Compute honeypot accuracy
    honeypot_accuracy: float | None = None
    if honeypot_results:
        honeypot_accuracy = sum(honeypot_results) / len(honeypot_results)

    return AnnotationResult(
        accepted=accepted,
        rejected=rejected,
        honeypot_accuracy=honeypot_accuracy,
        pairs_verified=pairs_verified,
        contributor_tokens={
            "input_tokens": updated["total_input_tokens"],
            "output_tokens": updated["total_output_tokens"],
        },
    )


async def _build_result_from_existing(
    db: Pool, batch_id: uuid.UUID, contributor_id: uuid.UUID
) -> AnnotationResult:
    """Reconstruct an AnnotationResult from already-persisted annotations (idempotent retry)."""
    row = await db.fetchrow(
        """SELECT COUNT(*) AS accepted,
                  COALESCE(SUM(input_tokens), 0) AS total_input,
                  COALESCE(SUM(output_tokens), 0) AS total_output
           FROM annotations
           WHERE batch_id = $1 AND contributor_id = $2""",
        batch_id,
        contributor_id,
    )
    accepted = row["accepted"] if row else 0

    # Honeypot accuracy from existing annotations, compared in one query
    hp_rows = await db.fetch(
        """SELECT a.label, h.known_label
           FROM annotations a
           JOIN honeypots h ON h.pair_id = a.pair_id
           WHERE a.batch_id = $1 AND a.contributor_id = $2""",
        batch_id,
        contributor_id,
    )
    honeypot_accuracy: float | None = None
    if hp_rows:
        results = [int(r["known_label"]) == int(r["label"]) for r in hp_rows]
        honeypot_accuracy = sum(results) / len(results)

    updated = await db.fetchrow(
        "SELECT total_input_tokens, total_output_tokens FROM contributors WHERE id = $1",
        contributor_id,
    )

    return AnnotationResult(
        accepted=accepted,
        rejected=0,
        honeypot_accuracy=honeypot_accuracy,
        pairs_verified=0,
        contributor_tokens={
            "input_tokens": updated["total_input_tokens"] if updated else 0,
            "output_tokens": updated["total_output_tokens"] if updated else 0,
        },
    )
