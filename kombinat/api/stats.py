from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends

from kombinat.dependencies import get_db
from kombinat.schemas.stats import StatsOut

if TYPE_CHECKING:
    import asyncpg

router = APIRouter(tags=["stats"])


@router.get(
    "/stats",
    response_model=StatsOut,
    status_code=200,
    summary="Public progress statistics",
)
async def get_stats(
    db: asyncpg.Pool = Depends(get_db),  # noqa: B008
) -> StatsOut:
    """Return public statistics about annotation progress. No auth required."""
    # Pair counts by status
    pair_counts = await db.fetchrow(
        """SELECT
            COUNT(*) AS total,
            COUNT(*) FILTER (WHERE status = 'unlabeled') AS unlabeled,
            COUNT(*) FILTER (WHERE status = 'verified') AS verified,
            COUNT(*) FILTER (WHERE status = 'rejected') AS rejected
        FROM pairs"""
    )
    assert pair_counts is not None

    # Contributor counts
    contributor_counts = await db.fetchrow(
        """SELECT
            COUNT(*) AS total,
            COUNT(*) FILTER (WHERE last_seen_at > NOW() - interval '24 hours') AS active_24h
        FROM contributors"""
    )
    assert contributor_counts is not None

    # Token totals
    token_totals = await db.fetchrow(
        """SELECT
            COALESCE(SUM(total_input_tokens), 0) AS input_tokens,
            COALESCE(SUM(total_output_tokens), 0) AS output_tokens
        FROM contributors"""
    )
    assert token_totals is not None

    # Pairs verified in the last 24h (approximate via annotations created)
    pairs_per_day_row = await db.fetchrow(
        """SELECT COUNT(DISTINCT pair_id) AS cnt
        FROM annotations
        WHERE created_at > NOW() - interval '24 hours'"""
    )
    pairs_per_day = pairs_per_day_row["cnt"] if pairs_per_day_row else 0

    return StatsOut(
        total_pairs=pair_counts["total"],
        unlabeled_pairs=pair_counts["unlabeled"],
        verified_pairs=pair_counts["verified"],
        rejected_pairs=pair_counts["rejected"],
        active_contributors_24h=contributor_counts["active_24h"],
        total_contributors=contributor_counts["total"],
        pairs_per_day=pairs_per_day,
        total_input_tokens=token_totals["input_tokens"],
        total_output_tokens=token_totals["output_tokens"],
    )
