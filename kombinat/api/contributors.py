from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, HTTPException

from kombinat.auth import create_jwt, exchange_github_code
from kombinat.config import get_settings
from kombinat.dependencies import get_current_contributor, get_db
from kombinat.schemas.contributors import AuthRequest, AuthResponse, ContributorOut

if TYPE_CHECKING:
    import asyncpg

router = APIRouter(tags=["auth"])


@router.post(
    "/auth/github",
    response_model=AuthResponse,
    status_code=200,
    summary="Exchange GitHub OAuth code for kombinat JWT",
    responses={401: {"description": "Invalid GitHub code"}},
)
async def auth_github(
    body: AuthRequest,
    db: asyncpg.Pool = Depends(get_db),  # noqa: B008
) -> AuthResponse:
    """Exchange a GitHub OAuth code for a kombinat access token."""
    try:
        user_data = await exchange_github_code(body.code)
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid GitHub code") from None

    github_id = user_data["id"]
    github_username = user_data["login"]
    github_avatar_url = user_data.get("avatar_url")

    # Upsert contributor
    row = await db.fetchrow(
        """INSERT INTO contributors (github_id, github_username, github_avatar_url)
        VALUES ($1, $2, $3)
        ON CONFLICT (github_id) DO UPDATE SET
            github_username = EXCLUDED.github_username,
            github_avatar_url = EXCLUDED.github_avatar_url,
            last_seen_at = NOW()
        RETURNING *""",
        github_id,
        github_username,
        github_avatar_url,
    )
    if row is None:
        raise HTTPException(status_code=500, detail="Failed to upsert contributor")

    settings = get_settings()
    token = create_jwt(str(row["id"]), github_id)

    contributor = ContributorOut(
        id=row["id"],
        github_username=row["github_username"],
        github_avatar_url=row["github_avatar_url"],
        reputation_score=row["reputation_score"],
        total_annotations=row["total_annotations"],
        total_input_tokens=row["total_input_tokens"],
        total_output_tokens=row["total_output_tokens"],
        created_at=row["created_at"],
        last_seen_at=row["last_seen_at"],
    )

    return AuthResponse(
        access_token=token,
        expires_in=settings.jwt_expiry_seconds,
        contributor=contributor,
    )


@router.get(
    "/contributors/me",
    response_model=ContributorOut,
    status_code=200,
    tags=["contributors"],
    summary="Get current contributor profile",
    responses={401: {"description": "Not authenticated"}},
)
async def get_me(
    contributor: dict[str, Any] = Depends(get_current_contributor),  # noqa: B008
) -> ContributorOut:
    """Return the authenticated contributor's profile."""
    return ContributorOut(
        id=contributor["id"],
        github_username=contributor["github_username"],
        github_avatar_url=contributor["github_avatar_url"],
        reputation_score=contributor["reputation_score"],
        total_annotations=contributor["total_annotations"],
        total_input_tokens=contributor["total_input_tokens"],
        total_output_tokens=contributor["total_output_tokens"],
        created_at=contributor["created_at"],
        last_seen_at=contributor["last_seen_at"],
    )
