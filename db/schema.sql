-- RECsysX PostgreSQL schema.
-- Users, items and ratings are loaded from data/processed/ by scripts/db_load.py.
-- `recommendations` is written by the final pipeline (scripts/recommend.py --to-db).

CREATE TABLE IF NOT EXISTS users (
    user_idx      INTEGER PRIMARY KEY,
    movielens_id  INTEGER UNIQUE NOT NULL,
    gender        CHAR(1) NOT NULL,
    age           SMALLINT NOT NULL,          -- ML-1M age group code (1, 18, 25, ...)
    occupation    SMALLINT NOT NULL,
    zip           TEXT,
    eval_role     TEXT NOT NULL DEFAULT 'none' CHECK (eval_role IN ('val', 'test', 'none')),
    user_group    TEXT NOT NULL CHECK (user_group IN ('cold', 'sparse', 'warm'))
);

CREATE TABLE IF NOT EXISTS items (
    item_idx      INTEGER PRIMARY KEY,
    movielens_id  INTEGER UNIQUE NOT NULL,
    title         TEXT NOT NULL,
    release_year  SMALLINT,
    genres        TEXT[] NOT NULL,
    is_head       BOOLEAN NOT NULL            -- top 20% by training popularity
);

CREATE TABLE IF NOT EXISTS ratings (
    user_idx      INTEGER NOT NULL REFERENCES users (user_idx),
    item_idx      INTEGER NOT NULL REFERENCES items (item_idx),
    rating        SMALLINT NOT NULL CHECK (rating BETWEEN 1 AND 5),
    rated_at      TIMESTAMP NOT NULL,
    period        TEXT NOT NULL CHECK (period IN ('train', 'eval')),
    PRIMARY KEY (user_idx, item_idx)
);
CREATE INDEX IF NOT EXISTS ratings_user_time ON ratings (user_idx, rated_at);
CREATE INDEX IF NOT EXISTS ratings_item ON ratings (item_idx);

-- what the models are allowed to learn from
CREATE OR REPLACE VIEW train_positives AS
SELECT user_idx, item_idx, rated_at
FROM ratings
WHERE period = 'train' AND rating >= 4;

CREATE TABLE IF NOT EXISTS recommendations (
    run_id        TEXT NOT NULL,
    user_idx      INTEGER NOT NULL REFERENCES users (user_idx),
    rank          SMALLINT NOT NULL,
    item_idx      INTEGER NOT NULL REFERENCES items (item_idx),
    score         REAL,
    created_at    TIMESTAMP NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, user_idx, rank)
);
