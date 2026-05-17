# Contributing to kombinat

Thanks for your interest. This document covers the dev loop for the embedkombinat coordination server.

## Development setup

Prerequisites: Python 3.12+, Docker (for PostgreSQL), [uv](https://docs.astral.sh/uv/), and [dbmate](https://github.com/amacneil/dbmate) for migrations.

```bash
git clone https://github.com/embedkombinat/kombinat.git
cd kombinat
uv sync --all-extras

# Start Postgres 16 on :5432
docker compose up -d

# Apply migrations
DATABASE_URL=postgresql://kombinat:kombinat@localhost:5432/kombinat dbmate up

# Run the server with auto-reload
uvicorn kombinat.main:app --reload
```

Local `.env` lives at the repo root and is gitignored; copy `.env.example` and fill in the placeholders for `GITHUB_CLIENT_ID`, `GITHUB_CLIENT_SECRET`, and `JWT_SECRET`. For local dev you can use throwaway values — only `JWT_SECRET` needs to be at least 32 hex bytes.

## Workflow

- Branch off `main`. Open a pull request when ready.
- CI runs `ruff check`, `ruff format --check`, `mypy`, and `pytest`. Run them locally before pushing:

```bash
uv run ruff check --fix .
uv run ruff format .
uv run ruff check .              # confirm zero errors
uv run ruff format --check .     # confirm formatting clean
uv run mypy kombinat/

DATABASE_URL=postgresql://kombinat:kombinat@localhost:5432/kombinat \
  GITHUB_CLIENT_ID=test \
  GITHUB_CLIENT_SECRET=test \
  JWT_SECRET=deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef \
  uv run pytest -v --tb=short
```

We do not accept `# noqa` or `# type: ignore` to silence existing errors. Fix the root cause, including in files you didn't touch.

## Migrations

Migrations are raw SQL under `db/migrations/`. We do not use an ORM. Create new migrations with `dbmate new <name>`. Keep them additive and idempotent where possible.

The `honeypots` table is populated **operationally**, not via migrations. Do not commit honeypot pair fixtures to this repo — see [db/migrations/README.md](db/migrations/README.md).

## Reporting bugs and proposing features

File an issue at https://github.com/embedkombinat/kombinat/issues. For substantive proposals (new API endpoints, schema changes), an issue first to align on direction is appreciated.

## Security disclosures

For anything that looks like a real vulnerability, please don't file it as a public issue. Use [GitHub Private Vulnerability Reporting](https://github.com/embedkombinat/kombinat/security/advisories/new) or email security@embedkombinat.org.
