import asyncpg


async def honeypot_check(pool: asyncpg.Pool, pair_id: object, label: int) -> bool:
    """Check if annotation matches honeypot known label. Returns True if pass."""
    row = await pool.fetchrow("SELECT known_label FROM honeypots WHERE pair_id = $1", pair_id)
    if row is None:
        return True  # not a honeypot, passes
    return int(row["known_label"]) == label


async def anomaly_check(
    pool: asyncpg.Pool, contributor_id: object, pair_id: object, label: int
) -> bool:
    """Stub: always passes in v0."""
    return True
