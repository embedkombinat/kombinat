from __future__ import annotations

import time
from typing import Any

import httpx
import jwt

from kombinat.config import get_settings


async def fetch_github_user(access_token: str) -> dict[str, Any]:
    """Look up a GitHub user via an OAuth access token. Raises ValueError if invalid."""
    async with httpx.AsyncClient() as client:
        user_resp = await client.get(
            "https://api.github.com/user",
            headers={"Authorization": f"Bearer {access_token}"},
        )
    if user_resp.status_code != 200:
        raise ValueError("Invalid GitHub access token")
    user_data: dict[str, Any] = user_resp.json()
    return user_data


async def exchange_github_code(code: str) -> dict[str, Any]:
    """Exchange a GitHub OAuth code (web flow) for user profile data."""
    settings = get_settings()
    async with httpx.AsyncClient() as client:
        token_resp = await client.post(
            "https://github.com/login/oauth/access_token",
            json={
                "client_id": settings.github_client_id,
                "client_secret": settings.github_client_secret,
                "code": code,
            },
            headers={"Accept": "application/json"},
        )
        token_data = token_resp.json()

        if "access_token" not in token_data:
            raise ValueError("Invalid GitHub OAuth code")

        user_resp = await client.get(
            "https://api.github.com/user",
            headers={"Authorization": f"Bearer {token_data['access_token']}"},
        )
        user_data: dict[str, Any] = user_resp.json()
        return user_data


def create_jwt(contributor_id: str, github_id: int) -> str:
    """Create a kombinat JWT for a contributor."""
    settings = get_settings()
    now = int(time.time())
    payload = {
        "sub": contributor_id,
        "github_id": github_id,
        "iat": now,
        "exp": now + settings.jwt_expiry_seconds,
    }
    token: str = jwt.encode(payload, settings.jwt_secret, algorithm="HS256")
    return token


def decode_jwt(token: str) -> dict[str, Any]:
    """Decode and verify a kombinat JWT."""
    settings = get_settings()
    payload: dict[str, Any] = jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
    return payload
