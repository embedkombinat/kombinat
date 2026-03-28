from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, HTTPException

from kombinat.dependencies import get_current_contributor, get_db
from kombinat.schemas.annotations import AnnotationResult, AnnotationSubmission
from kombinat.validator.checks import honeypot_check
from kombinat.validator.promote import maybe_promote_pair
from kombinat.validator.reputation import update_reputation

if TYPE_CHECKING:
    import uuid

    import asyncpg

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
    """Submit annotation labels for a claimed batch."""
    contributor_id = contributor["id"]

    # Validate batch
    batch = await db.fetchrow("SELECT * FROM batches WHERE id = $1", body.batch_id)
    if batch is None:
        raise HTTPException(status_code=404, detail="Batch not found")
    if batch["contributor_id"] != contributor_id:
        raise HTTPException(status_code=403, detail="Not the batch owner")
    if batch["status"] != "assigned":
        raise HTTPException(status_code=400, detail="Batch not in assigned status")

    # Get valid pair_ids for this batch
    batch_pair_rows = await db.fetch(
        "SELECT pair_id FROM batch_pairs WHERE batch_id = $1", body.batch_id
    )
    valid_pair_ids = {row["pair_id"] for row in batch_pair_rows}

    accepted = 0
    rejected = 0
    total_input = 0
    total_output = 0
    seen_pair_ids: set[uuid.UUID] = set()
    annotated_pair_ids: list[uuid.UUID] = []
    honeypot_results: list[bool] = []

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

        # Check if this is a honeypot pair
        hp_row = await db.fetchrow("SELECT 1 FROM honeypots WHERE pair_id = $1", ann.pair_id)
        is_honeypot = hp_row is not None

        try:
            await db.execute(
                """INSERT INTO annotations
                (pair_id, contributor_id, batch_id, label, model_id, quantization,
                 input_tokens, output_tokens, raw_response_hash, is_honeypot)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)""",
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
            accepted += 1
            total_input += ann.input_tokens
            total_output += ann.output_tokens
            annotated_pair_ids.append(ann.pair_id)

            # Run honeypot check
            if is_honeypot:
                hp_pass = await honeypot_check(db, ann.pair_id, ann.label)
                honeypot_results.append(hp_pass)

        except Exception:  # noqa: BLE001
            # Unique constraint violation or other DB error
            rejected += 1

    # Update contributor totals
    await db.execute(
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

    # Promote pairs that have reached required annotations
    pairs_verified = 0
    for pair_id in annotated_pair_ids:
        promoted = await maybe_promote_pair(db, pair_id)
        if promoted:
            pairs_verified += 1

    # Update reputation (stub: no-op in v0)
    await update_reputation(db, contributor_id, honeypot_results)

    # Mark batch as completed
    await db.execute(
        "UPDATE batches SET status = 'completed', completed_at = NOW() WHERE id = $1",
        body.batch_id,
    )

    # Compute honeypot accuracy
    honeypot_accuracy: float | None = None
    if honeypot_results:
        honeypot_accuracy = sum(honeypot_results) / len(honeypot_results)

    # Reload contributor for response
    updated = await db.fetchrow(
        "SELECT total_input_tokens, total_output_tokens FROM contributors WHERE id = $1",
        contributor_id,
    )
    assert updated is not None

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
