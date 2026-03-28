from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import jwt as pyjwt

from kombinat.config import get_settings

if TYPE_CHECKING:
    import asyncpg
    from httpx import AsyncClient


async def _mock_github_exchange(
    github_id: int = 12345,
    login: str = "testuser",
    avatar_url: str = "https://example.com/avatar.png",
) -> AsyncMock:
    """Create a mock for httpx.AsyncClient that returns valid GitHub responses."""
    mock_client = AsyncMock()

    # Use MagicMock for responses since .json() is synchronous in httpx
    token_response = MagicMock()
    token_response.json.return_value = {"access_token": "gho_fake_token"}
    mock_client.post.return_value = token_response

    user_response = MagicMock()
    user_response.json.return_value = {
        "id": github_id,
        "login": login,
        "avatar_url": avatar_url,
    }
    mock_client.get.return_value = user_response

    # Make it work as an async context manager
    mock_cm = AsyncMock()
    mock_cm.__aenter__.return_value = mock_client
    mock_cm.__aexit__.return_value = None
    return mock_cm


async def test_auth_github_valid_code(client: AsyncClient, db_pool: asyncpg.Pool) -> None:
    """POST /v1/auth/github with valid code returns contributor + JWT."""
    mock_cm = await _mock_github_exchange()
    with patch("kombinat.auth.httpx.AsyncClient", return_value=mock_cm):
        resp = await client.post(
            "/v1/auth/github", json={"code": "valid_code", "state": "csrf_token"}
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert "expires_in" in data
    assert data["contributor"]["github_username"] == "testuser"


async def test_auth_github_invalid_code(client: AsyncClient) -> None:
    """POST /v1/auth/github with invalid code returns 401."""
    mock_client = AsyncMock()
    token_response = MagicMock()
    token_response.json.return_value = {"error": "bad_verification_code"}
    mock_client.post.return_value = token_response

    mock_cm = AsyncMock()
    mock_cm.__aenter__.return_value = mock_client
    mock_cm.__aexit__.return_value = None

    with patch("kombinat.auth.httpx.AsyncClient", return_value=mock_cm):
        resp = await client.post("/v1/auth/github", json={"code": "invalid", "state": "csrf"})
    assert resp.status_code == 401


async def test_auth_jwt_expiry(client: AsyncClient, db_pool: asyncpg.Pool) -> None:
    """JWT exp claim equals iat + jwt_expiry_seconds."""
    settings = get_settings()
    mock_cm = await _mock_github_exchange()
    with patch("kombinat.auth.httpx.AsyncClient", return_value=mock_cm):
        resp = await client.post("/v1/auth/github", json={"code": "valid", "state": "csrf"})
    data = resp.json()
    decoded = pyjwt.decode(data["access_token"], settings.jwt_secret, algorithms=["HS256"])
    assert decoded["exp"] == decoded["iat"] + settings.jwt_expiry_seconds


async def test_auth_jwt_claims(client: AsyncClient, db_pool: asyncpg.Pool) -> None:
    """JWT contains correct sub, github_id, iat, exp."""
    settings = get_settings()
    mock_cm = await _mock_github_exchange(github_id=67890)
    with patch("kombinat.auth.httpx.AsyncClient", return_value=mock_cm):
        resp = await client.post("/v1/auth/github", json={"code": "valid", "state": "csrf"})
    data = resp.json()
    decoded = pyjwt.decode(data["access_token"], settings.jwt_secret, algorithms=["HS256"])
    assert "sub" in decoded
    assert decoded["github_id"] == 67890
    assert "iat" in decoded
    assert "exp" in decoded


async def test_contributors_me_valid(
    authed_client: AsyncClient,
    contributor: dict[str, Any],
) -> None:
    """GET /v1/contributors/me with valid JWT returns contributor profile."""
    resp = await authed_client.get("/v1/contributors/me")
    assert resp.status_code == 200
    data = resp.json()
    assert data["github_username"] == "testuser"
    assert data["id"] == str(contributor["id"])


async def test_contributors_me_no_token(client: AsyncClient) -> None:
    """GET /v1/contributors/me without token returns 401."""
    resp = await client.get("/v1/contributors/me")
    assert resp.status_code in (401, 403)


async def test_contributors_me_expired(client: AsyncClient, contributor: dict[str, Any]) -> None:
    """GET /v1/contributors/me with expired JWT returns 401 with 'Token expired'."""
    settings = get_settings()
    payload = {
        "sub": str(contributor["id"]),
        "github_id": contributor["github_id"],
        "iat": int(time.time()) - 1000,
        "exp": int(time.time()) - 500,  # expired 500 seconds ago
    }
    expired_token = pyjwt.encode(payload, settings.jwt_secret, algorithm="HS256")
    resp = await client.get(
        "/v1/contributors/me",
        headers={"Authorization": f"Bearer {expired_token}"},
    )
    assert resp.status_code == 401
    assert resp.json()["detail"] == "Token expired"


async def test_contributors_me_deleted_user(client: AsyncClient) -> None:
    """GET /v1/contributors/me with JWT for non-existent contributor returns 401."""
    settings = get_settings()
    fake_id = "00000000-0000-0000-0000-000000000000"
    payload = {
        "sub": fake_id,
        "github_id": 99999,
        "iat": int(time.time()),
        "exp": int(time.time()) + 3600,
    }
    token = pyjwt.encode(payload, settings.jwt_secret, algorithm="HS256")
    resp = await client.get(
        "/v1/contributors/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 401


async def test_auth_repeat_login_updates_profile(
    client: AsyncClient, db_pool: asyncpg.Pool
) -> None:
    """Two logins with same github_id but different username updates the contributor."""
    mock_cm1 = await _mock_github_exchange(
        github_id=11111, login="user_v1", avatar_url="https://example.com/v1.png"
    )
    with patch("kombinat.auth.httpx.AsyncClient", return_value=mock_cm1):
        resp1 = await client.post("/v1/auth/github", json={"code": "code1", "state": "csrf"})
    assert resp1.status_code == 200
    assert resp1.json()["contributor"]["github_username"] == "user_v1"

    # Second login with updated profile
    mock_cm2 = await _mock_github_exchange(
        github_id=11111, login="user_v2", avatar_url="https://example.com/v2.png"
    )
    with patch("kombinat.auth.httpx.AsyncClient", return_value=mock_cm2):
        resp2 = await client.post("/v1/auth/github", json={"code": "code2", "state": "csrf"})
    assert resp2.status_code == 200
    assert resp2.json()["contributor"]["github_username"] == "user_v2"
    assert resp2.json()["contributor"]["github_avatar_url"] == "https://example.com/v2.png"
