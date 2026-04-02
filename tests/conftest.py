from __future__ import annotations

import os
import pathlib
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

import asyncpg
import pytest
from httpx import ASGITransport, AsyncClient

from kombinat.auth import create_jwt
from kombinat.main import app

TEST_DB = "kombinat_test"
BASE_DSN = os.environ.get("DATABASE_URL", "postgresql://kombinat:kombinat@localhost:5432/kombinat")
# Derive the admin DSN (connect to 'postgres' db) and test DSN
_parts = BASE_DSN.rsplit("/", 1)
ADMIN_DSN = _parts[0] + "/postgres"
TEST_DSN = _parts[0] + f"/{TEST_DB}"

MIGRATION_FILE = (
    pathlib.Path(__file__).parent.parent / "db" / "migrations" / "20260328000000_initial_schema.sql"
)


def _extract_up_migration(path: pathlib.Path) -> str:
    """Extract the -- migrate:up section from a dbmate migration file."""
    text = path.read_text()
    up_marker = "-- migrate:up"
    down_marker = "-- migrate:down"
    up_idx = text.index(up_marker) + len(up_marker)
    down_idx = text.index(down_marker)
    return text[up_idx:down_idx].strip()


@pytest.fixture(scope="session")
async def db_pool() -> AsyncIterator[asyncpg.Pool]:
    """Create a test database, run migrations, yield pool, drop DB after."""
    admin_conn = await asyncpg.connect(ADMIN_DSN)
    try:
        # Drop if exists from a previous failed run
        await admin_conn.execute(f"DROP DATABASE IF EXISTS {TEST_DB}")
        await admin_conn.execute(f"CREATE DATABASE {TEST_DB}")
    finally:
        await admin_conn.close()

    # Run migration SQL directly
    migration_sql = _extract_up_migration(MIGRATION_FILE)
    test_conn = await asyncpg.connect(TEST_DSN)
    try:
        await test_conn.execute(migration_sql)
    finally:
        await test_conn.close()

    pool: asyncpg.Pool = await asyncpg.create_pool(  # type: ignore[assignment]
        TEST_DSN, min_size=2, max_size=10
    )
    yield pool
    await pool.close()

    # Drop test database
    admin_conn = await asyncpg.connect(ADMIN_DSN)
    try:
        await admin_conn.execute(f"DROP DATABASE IF EXISTS {TEST_DB}")
    finally:
        await admin_conn.close()


@pytest.fixture(autouse=True)
async def clean_tables(request: pytest.FixtureRequest) -> None:
    """Truncate all tables between tests (skipped for tests that don't use the DB)."""
    # Avoid setting up db_pool for tests that don't need it (e.g. ingest unit tests)
    if "tests/ingest" in str(request.fspath):
        return
    pool: asyncpg.Pool = request.getfixturevalue("db_pool")
    await pool.execute(
        "TRUNCATE pairs, annotations, batches, batch_pairs, contributors, honeypots CASCADE"
    )


@pytest.fixture
async def client(db_pool: asyncpg.Pool) -> AsyncIterator[AsyncClient]:
    """httpx AsyncClient pointed at test app."""
    app.state.db = db_pool
    transport = ASGITransport(app=app)  # type: ignore[arg-type]
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def _insert_contributor(
    pool: asyncpg.Pool,
    github_id: int = 12345,
    github_username: str = "testuser",
    github_avatar_url: str = "https://example.com/avatar.png",
) -> dict[str, Any]:
    """Insert a contributor and return the row as a dict."""
    row = await pool.fetchrow(
        """INSERT INTO contributors (github_id, github_username, github_avatar_url)
        VALUES ($1, $2, $3)
        RETURNING *""",
        github_id,
        github_username,
        github_avatar_url,
    )
    assert row is not None
    return dict(row)


@pytest.fixture
async def contributor(db_pool: asyncpg.Pool) -> dict[str, Any]:
    """Insert a test contributor."""
    return await _insert_contributor(db_pool)


@pytest.fixture
async def contributor_token(contributor: dict[str, Any]) -> str:
    """Generate a JWT for the test contributor."""
    return create_jwt(str(contributor["id"]), contributor["github_id"])


@pytest.fixture
async def authed_client(
    db_pool: asyncpg.Pool, contributor: dict[str, Any], contributor_token: str
) -> AsyncIterator[AsyncClient]:
    """Client with a valid contributor auth token."""
    app.state.db = db_pool
    transport = ASGITransport(app=app)  # type: ignore[arg-type]
    async with AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={"Authorization": f"Bearer {contributor_token}"},
    ) as c:
        yield c


async def _insert_contributor_b(pool: asyncpg.Pool) -> dict[str, Any]:
    """Insert a second contributor for multi-user tests."""
    return await _insert_contributor(pool, github_id=99999, github_username="testuser2")


@pytest.fixture
async def contributor_b(db_pool: asyncpg.Pool) -> dict[str, Any]:
    return await _insert_contributor_b(db_pool)


@pytest.fixture
async def contributor_b_token(contributor_b: dict[str, Any]) -> str:
    return create_jwt(str(contributor_b["id"]), contributor_b["github_id"])


@pytest.fixture
async def authed_client_b(
    db_pool: asyncpg.Pool,
    contributor_b: dict[str, Any],
    contributor_b_token: str,
) -> AsyncIterator[AsyncClient]:
    """Second authenticated client for multi-user tests."""
    app.state.db = db_pool
    transport = ASGITransport(app=app)  # type: ignore[arg-type]
    async with AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={"Authorization": f"Bearer {contributor_b_token}"},
    ) as c:
        yield c


@pytest.fixture
async def seeded_pairs(db_pool: asyncpg.Pool) -> list[dict[str, Any]]:
    """Insert 50 unlabeled pairs for testing."""
    pairs = []
    for i in range(50):
        pair_id = uuid.uuid5(uuid.NAMESPACE_DNS, f"query_{i}:doc_{i}:msmarco")
        row = await db_pool.fetchrow(
            """INSERT INTO pairs
            (id, query_text, doc_id, doc_text, source_dataset, retrieval_method, source_rank)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING *""",
            pair_id,
            f"query text {i}",
            f"doc_{i}",
            f"document text {i}",
            "msmarco",
            "bm25",
            i + 1,
        )
        assert row is not None
        pairs.append(dict(row))
    return pairs


@pytest.fixture
async def honeypot_pairs(db_pool: asyncpg.Pool) -> list[dict[str, Any]]:
    """Insert 5 honeypot pairs with known labels."""
    pairs = []
    for i in range(5):
        pair_id = uuid.uuid5(uuid.NAMESPACE_DNS, f"honeypot_query_{i}:honeypot_doc_{i}:msmarco")
        row = await db_pool.fetchrow(
            """INSERT INTO pairs
            (id, query_text, doc_id, doc_text, source_dataset, retrieval_method, source_rank)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING *""",
            pair_id,
            f"honeypot query {i}",
            f"honeypot_doc_{i}",
            f"honeypot document {i}",
            "msmarco",
            "bm25",
            i + 1,
        )
        assert row is not None
        known_label = i % 4  # labels 0, 1, 2, 3, 0
        await db_pool.execute(
            "INSERT INTO honeypots (pair_id, known_label) VALUES ($1, $2)",
            pair_id,
            known_label,
        )
        pair_dict = dict(row)
        pair_dict["known_label"] = known_label
        pairs.append(pair_dict)
    return pairs
