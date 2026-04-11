# CLAUDE.md — kombinat

This file provides guidance to Claude Code (claude.ai/code) when working in the **kombinat** sub-repo (FastAPI coordination server, Python 3.12+, asyncpg, PostgreSQL 16).

## Build and dev commands

```bash
pip install -e ".[dev]"          # or: uv sync --all-extras

# Dependencies (Postgres 16 on :5432)
docker compose up -d

# Migrations (requires dbmate)
DATABASE_URL=postgresql://kombinat:kombinat@localhost:5432/kombinat dbmate up

# Server
uvicorn kombinat.main:app --reload

# Lint, format, type check
uv run ruff check .
uv run ruff format --check .
uv run mypy kombinat/

# Tests (need Postgres running)
DATABASE_URL=postgresql://kombinat:kombinat@localhost:5432/kombinat \
GITHUB_CLIENT_ID=test \
GITHUB_CLIENT_SECRET=test \
JWT_SECRET=deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef \
uv run pytest -v --tb=short

# Single test
uv run pytest tests/test_batches.py::test_claim_batch -v --tb=long

# Ingest pipeline (separate extras)
uv sync --extra ingest
uv run python -m kombinat.tools.ingest --split squad
```

## Linting policy

CI runs `ruff check .` and `ruff format --check .` as hard gates. Both must pass on every push.

**Always fix lint errors — never ignore them.** When you finish editing Python files, run:

```bash
uv run ruff check --fix .
uv run ruff format .
uv run ruff check .              # confirm zero errors
uv run ruff format --check .     # confirm formatting clean
```

Fix every error surfaced, including in files you did not touch directly. Do not add `# noqa` suppressions or extend `tool.ruff.lint.ignore` to silence pre-existing errors — delete dead code, rename unused variables to `_`, shorten long lines, add `strict=` to `zip()`, etc. The only acceptable reason to touch ignore lists is when a rule is genuinely wrong for this codebase, and that change must be called out explicitly.

Same rule for `uv run mypy kombinat/` — fix the types, don't add `# type: ignore`.

## Architecture

FastAPI monolith, no ORM. Raw SQL via asyncpg with connection pooling. Database pool lives in `app.state.db`, injected via `get_db()` dependency.

- `kombinat/api/` — Route handlers: batches, annotations, contributors (auth), stats
- `kombinat/schemas/` — Pydantic request/response models
- `kombinat/validator/` — honeypot_check, promote (majority vote), reputation (stub)
- `kombinat/tools/ingest/` — CLI pipeline: source → bm25 → dense → fusion → pairs → writer
- `kombinat/auth.py` — GitHub OAuth code exchange + JWT (HS256) issuance
- `kombinat/dependencies.py` — `get_db()` and `get_current_contributor()` (Bearer JWT → DB lookup)
- `db/migrations/` — dbmate SQL migrations (raw SQL, no ORM)

### Data flow

1. **Ingest** (`kombinat/tools/ingest/`): Loads HuggingFace datasets, builds BM25 + FAISS indexes, fuses rankings via RRF, writes candidate (query, doc) pairs to PostgreSQL.
2. **Claim** (`POST /v1/batches/claim`): Annotators claim a batch of unlabeled pairs. Uses `SELECT FOR UPDATE SKIP LOCKED` for concurrent safety. ~5% of pairs are honeypots (shuffled in, never revealed).
3. **Validate** (`POST /v1/annotations`): Honeypot check inline, update contributor tokens, promote pairs via majority vote when `required_annotations` (default 2) are met.
4. **Expire** (background loop): Hourly task marks stale `assigned` batches as `expired` (24h TTL).

### Key conventions

- **Pydantic everywhere** (BaseSettings, API schemas, internal DTOs). No dataclasses.
- **Deterministic IDs**: Ingest pipeline uses `uuid5(NAMESPACE_URL, f"{query}|{doc_id}|{source}")` for idempotent re-ingestion.
- **Pair promotion is inline**: Consensus voting happens synchronously during annotation submission, not as a separate job.
- **Config via env vars**: `DATABASE_URL`, `JWT_SECRET`, `GITHUB_CLIENT_ID`, `GITHUB_CLIENT_SECRET`.
- **Tests create their own DB**: `conftest.py` creates/migrates/truncates a test database per session using migration SQL extracted from dbmate files.
- **pytest uses `asyncio_mode = "auto"`** with session-scoped event loop.
