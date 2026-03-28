import asyncpg


async def update_reputation(
    pool: asyncpg.Pool,
    contributor_id: object,
    honeypot_results: list[bool],
) -> None:
    """Stub: no-op in v0. Does not change reputation score."""
