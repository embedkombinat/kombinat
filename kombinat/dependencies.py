from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jwt
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from kombinat.auth import decode_jwt

if TYPE_CHECKING:
    import asyncpg

security = HTTPBearer(auto_error=False)


async def get_db(request: Request) -> asyncpg.Pool:
    pool: asyncpg.Pool = request.app.state.db
    return pool


async def get_current_contributor(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),  # noqa: B008
    db: asyncpg.Pool = Depends(get_db),  # noqa: B008
) -> dict[str, Any]:
    if credentials is None:
        raise HTTPException(status_code=401, detail="Not authenticated")

    try:
        payload = decode_jwt(credentials.credentials)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired") from None
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token") from None

    contributor_id = payload["sub"]
    row = await db.fetchrow("SELECT * FROM contributors WHERE id = $1", contributor_id)
    if row is None:
        raise HTTPException(status_code=401, detail="Contributor not found")

    return dict(row)
