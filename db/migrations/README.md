# Migrations

Raw SQL migrations managed by [dbmate](https://github.com/amacneil/dbmate). No ORM, no migration framework other than dbmate's filename-ordered apply.

## Running locally

```bash
DATABASE_URL=postgresql://kombinat:kombinat@localhost:5432/kombinat dbmate up
```

`dbmate up` is idempotent — it applies any migrations whose timestamp prefix hasn't been recorded in the `schema_migrations` table.

## Creating a new migration

```bash
dbmate new add_some_column
```

Edit the generated file under `db/migrations/`. Keep the migration additive and idempotent where possible. Run it locally end-to-end before committing.

## Important: do not commit honeypot data

The `honeypots` table is populated **operationally**, not via migrations. It holds curated query-document pairs with their ground-truth labels — that's the validation oracle the server uses to detect bad annotations.

If we shipped a honeypot fixture in this repo, an attacker could read it from the public source tree and pre-compute the "correct" answers for every honeypot they get served. That defeats the entire validation mechanism.

When you write a migration that touches `honeypots`, only change the schema. Never insert honeypot rows in a migration, a test fixture, or any file in this repository.
