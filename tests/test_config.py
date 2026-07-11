"""Settings immutability: get_settings() caches one shared instance per
process, so mutation would silently change process-wide config."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from kombinat.config import Settings

if TYPE_CHECKING:
    import asyncpg


# The db_pool parameter is unused but required: the autouse clean_tables
# fixture resolves db_pool via getfixturevalue, which only works when the
# session-scoped pool is already in the test's fixture closure.
async def test_settings_are_frozen(db_pool: asyncpg.Pool) -> None:
    settings = Settings()
    with pytest.raises(ValidationError):
        settings.jwt_secret = "attacker-controlled"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        settings.database_url = "postgresql://evil/db"  # type: ignore[misc]
