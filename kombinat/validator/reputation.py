import asyncpg


async def update_reputation(
    pool: asyncpg.Pool,
    contributor_id: object,
    honeypot_results: list[bool],
) -> None:
    """Update a contributor's reputation from a batch's honeypot outcomes.

    v0 is a no-op: honeypot misses are recorded elsewhere but do not yet
    affect claim quota or vote weight. v1 will decay reputation on misses
    and gate claim eligibility on a threshold. See:
    https://github.com/embedkombinat/kombinat/issues — label `v1-reputation`.
    """
    # TODO(v1-reputation): exponential decay on honeypot miss, threshold-gated claims.
