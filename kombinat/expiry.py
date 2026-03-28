from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import asyncpg


async def expire_batches(pool: asyncpg.Pool) -> int:
    """Mark expired batches and return count of affected rows."""
    result: str = await pool.execute(
        "UPDATE batches SET status = 'expired' WHERE status = 'assigned' AND expires_at < NOW()"
    )
    # asyncpg returns 'UPDATE N' where N is the count
    return int(result.split()[-1])
