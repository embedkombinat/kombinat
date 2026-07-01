from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, HTTPException

from kombinat.dependencies import get_db
from kombinat.schemas.stats import LeaderboardEntry, LeaderboardOut, StatsOut

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
    if pair_counts is None:
        raise HTTPException(status_code=500, detail="Failed to query pair counts")

    # Contributor counts
    contributor_counts = await db.fetchrow(
        """SELECT
            COUNT(*) AS total,
            COUNT(*) FILTER (WHERE last_seen_at > NOW() - interval '24 hours') AS active_24h
        FROM contributors"""
    )
    if contributor_counts is None:
        raise HTTPException(status_code=500, detail="Failed to query contributor counts")

    # Token totals
    token_totals = await db.fetchrow(
        """SELECT
            COALESCE(SUM(total_input_tokens), 0) AS input_tokens,
            COALESCE(SUM(total_output_tokens), 0) AS output_tokens
        FROM contributors"""
    )
    if token_totals is None:
        raise HTTPException(status_code=500, detail="Failed to query token totals")

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


@router.get(
    "/stats/leaderboard",
    response_model=LeaderboardOut,
    status_code=200,
    summary="Top contributors by annotation count",
)
async def get_leaderboard(
    db: asyncpg.Pool = Depends(get_db),  # noqa: B008
    limit: int = 20,
) -> LeaderboardOut:
    """Return top contributors ranked by total annotations. No auth required."""
    rows = await db.fetch(
        """SELECT github_username, github_avatar_url, total_annotations
        FROM contributors
        WHERE total_annotations > 0
        ORDER BY total_annotations DESC
        LIMIT $1""",
        # Clamp instead of erroring so out-of-range values from third parties
        # can't 500 this public endpoint (Postgres rejects negative LIMIT).
        max(1, min(limit, 50)),
    )
    return LeaderboardOut(entries=[LeaderboardEntry(**dict(r)) for r in rows])
