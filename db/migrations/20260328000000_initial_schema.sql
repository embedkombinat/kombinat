-- migrate:up

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE contributors (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    github_id       BIGINT UNIQUE NOT NULL,
    github_username TEXT NOT NULL,
    github_avatar_url TEXT,
    reputation_score DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    total_annotations INTEGER NOT NULL DEFAULT 0,
    total_input_tokens BIGINT NOT NULL DEFAULT 0,
    total_output_tokens BIGINT NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE pairs (
    id                  UUID PRIMARY KEY,
    query_text          TEXT NOT NULL,
    doc_id              TEXT NOT NULL,
    doc_text            TEXT NOT NULL,
    source_dataset      TEXT NOT NULL,
    retrieval_method    TEXT NOT NULL,
    source_rank         INTEGER,
    status              TEXT NOT NULL DEFAULT 'unlabeled'
                        CHECK (status IN ('unlabeled', 'verified', 'rejected')),
    required_annotations INTEGER NOT NULL DEFAULT 2,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_pairs_status ON pairs (status);
CREATE INDEX idx_pairs_source ON pairs (source_dataset);
CREATE INDEX idx_pairs_created ON pairs (created_at);

CREATE TABLE batches (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    contributor_id  UUID NOT NULL REFERENCES contributors(id),
    size            INTEGER NOT NULL,
    status          TEXT NOT NULL DEFAULT 'assigned'
                    CHECK (status IN ('assigned', 'completed', 'expired')),
    assigned_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at      TIMESTAMPTZ NOT NULL,
    completed_at    TIMESTAMPTZ
);

CREATE INDEX idx_batches_status ON batches (status);
CREATE INDEX idx_batches_expires ON batches (expires_at) WHERE status = 'assigned';

CREATE TABLE batch_pairs (
    batch_id    UUID NOT NULL REFERENCES batches(id) ON DELETE CASCADE,
    pair_id     UUID NOT NULL REFERENCES pairs(id),
    PRIMARY KEY (batch_id, pair_id)
);

CREATE INDEX idx_batch_pairs_pair ON batch_pairs (pair_id);

CREATE TABLE annotations (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pair_id             UUID NOT NULL REFERENCES pairs(id),
    contributor_id      UUID NOT NULL REFERENCES contributors(id),
    batch_id            UUID NOT NULL REFERENCES batches(id),
    label               SMALLINT NOT NULL CHECK (label BETWEEN 0 AND 3),
    model_id            TEXT NOT NULL,
    quantization        TEXT NOT NULL,
    input_tokens        INTEGER NOT NULL,
    output_tokens       INTEGER NOT NULL,
    raw_response_hash   TEXT NOT NULL,
    is_honeypot         BOOLEAN NOT NULL DEFAULT FALSE,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (pair_id, contributor_id)
);

CREATE INDEX idx_annotations_pair ON annotations (pair_id);
CREATE INDEX idx_annotations_contributor ON annotations (contributor_id);
CREATE INDEX idx_annotations_created ON annotations (created_at);

-- Honeypot pairs: pairs with known ground-truth labels for quality control
CREATE TABLE honeypots (
    pair_id         UUID PRIMARY KEY REFERENCES pairs(id),
    known_label     SMALLINT NOT NULL CHECK (known_label BETWEEN 0 AND 3),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- migrate:down

DROP TABLE IF EXISTS honeypots;
DROP TABLE IF EXISTS annotations;
DROP TABLE IF EXISTS batch_pairs;
DROP TABLE IF EXISTS batches;
DROP TABLE IF EXISTS pairs;
DROP TABLE IF EXISTS contributors;
