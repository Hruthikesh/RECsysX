"""PostgreSQL storage for users, items, ratings and served recommendations.

Training reads the Parquet snapshot in data/processed/ (faster, no server needed), the
database is the store the final pipeline talks to: it fetches a user's profile and
rating history at request time and writes the recommendations back. scripts/db_check.py
cross-checks the split and the main dataset statistics in SQL against pandas.

Connection: DATABASE_URL (or --database-url). If neither is set and the optional
`pgserver` package is installed, a local server is started in .pgdata/ - handy on a
laptop without a PostgreSQL install.
"""
from __future__ import annotations

import io
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import PROJECT_ROOT, project_path

SCHEMA_SQL = PROJECT_ROOT / "db" / "schema.sql"
QUERIES_SQL = PROJECT_ROOT / "db" / "queries.sql"


def database_url(url: str | None = None) -> str:
    url = url or os.environ.get("DATABASE_URL")
    if url:
        return url
    try:
        import pgserver
    except ImportError as exc:
        raise RuntimeError("no database configured: set DATABASE_URL or `pip install pgserver`") from exc
    server = pgserver.get_server(project_path(".pgdata"), cleanup_mode=None)
    return server.get_uri()


def connect(url: str | None = None):
    import psycopg

    return psycopg.connect(database_url(url))


def create_schema(conn, drop: bool = False) -> None:
    with conn.cursor() as cur:
        if drop:
            cur.execute("DROP VIEW IF EXISTS train_positives; "
                        "DROP TABLE IF EXISTS recommendations, ratings, items, users CASCADE;")
        cur.execute(SCHEMA_SQL.read_text())
    conn.commit()


def _copy_df(conn, table: str, df: pd.DataFrame) -> None:
    buf = io.StringIO()
    df.to_csv(buf, index=False, header=False)
    cols = ", ".join(df.columns)
    with conn.cursor() as cur:
        with cur.copy(f"COPY {table} ({cols}) FROM STDIN WITH (FORMAT csv)") as cp:
            cp.write(buf.getvalue())


def _pg_array(values) -> str:
    return "{" + ",".join('"' + v.replace('"', '\\"') + '"' for v in values) + "}"


def load_processed(conn, processed_dir: str | Path) -> dict[str, int]:
    """Replace the users/items/ratings tables with the processed Parquet data."""
    d = Path(processed_dir)
    users = pd.read_parquet(d / "users.parquet")
    items = pd.read_parquet(d / "items.parquet")
    inter = pd.read_parquet(d / "interactions.parquet")
    with conn.cursor() as cur:
        cur.execute("TRUNCATE recommendations, ratings, items, users CASCADE")
    _copy_df(conn, "users", pd.DataFrame({
        "user_idx": users["user_idx"], "movielens_id": users["user_id"], "gender": users["gender"],
        "age": users["age"], "occupation": users["occupation"], "zip": users["zip"],
        "eval_role": users["role"], "user_group": users["group"]}))
    _copy_df(conn, "items", pd.DataFrame({
        "item_idx": items["item_idx"], "movielens_id": items["movie_id"], "title": items["title"],
        "release_year": items["year"].astype("Int64"), "genres": items["genres"].str.split("|").map(_pg_array),
        "is_head": items["is_head"]}))
    _copy_df(conn, "ratings", pd.DataFrame({
        "user_idx": inter["user_idx"], "item_idx": inter["item_idx"], "rating": inter["rating"],
        "rated_at": pd.to_datetime(inter["timestamp"], unit="s").dt.strftime("%Y-%m-%d %H:%M:%S"),
        "period": inter["period"]}))
    conn.commit()
    return {"users": len(users), "items": len(items), "ratings": len(inter)}


def load_interactions(conn, positive_threshold: int = 4) -> pd.DataFrame:
    """Same frame as data/processed/interactions.parquet, read back from the database."""
    with conn.cursor() as cur:
        cur.execute("SELECT user_idx, item_idx, rating, extract(epoch FROM rated_at)::bigint, period "
                    "FROM ratings ORDER BY rated_at, user_idx, item_idx")
        rows = cur.fetchall()
    out = pd.DataFrame(rows, columns=["user_idx", "item_idx", "rating", "timestamp", "period"])
    out["rating"] = out["rating"].astype(np.int8)
    out["timestamp"] = out["timestamp"].astype(np.int64)
    out["positive"] = out["rating"] >= positive_threshold
    return out[["user_idx", "item_idx", "rating", "timestamp", "positive", "period"]]


def named_queries(path: str | Path = QUERIES_SQL) -> dict[str, str]:
    text = Path(path).read_text()
    blocks = re.split(r"^-- name: *(\S+)\s*$", text, flags=re.M)
    return {blocks[i]: blocks[i + 1].strip() for i in range(1, len(blocks), 2)}


def run_query(conn, sql: str):
    with conn.cursor() as cur:
        cur.execute(sql)
        return cur.fetchone()[0]


class PostgresStore:
    """What the serving side needs from the database."""

    def __init__(self, conn):
        self.conn = conn

    def user_profile(self, user_idx: int) -> dict:
        with self.conn.cursor() as cur:
            cur.execute("SELECT gender, age, occupation, eval_role, user_group FROM users WHERE user_idx = %s",
                        (int(user_idx),))
            row = cur.fetchone()
            if row is None:
                raise KeyError(f"user {user_idx} not in database")
            # history = everything the user rated before the cutoff, used to filter
            # already-seen movies and as context for new-user recommendations
            cur.execute("SELECT item_idx, rating FROM ratings WHERE user_idx = %s AND period = 'train' "
                        "ORDER BY rated_at", (int(user_idx),))
            hist = cur.fetchall()
        return {"user_idx": int(user_idx), "gender": row[0], "age": row[1], "occupation": row[2],
                "eval_role": row[3], "group": row[4],
                "history_items": [h[0] for h in hist], "history_ratings": [h[1] for h in hist]}

    def item_titles(self, item_ids) -> dict[int, str]:
        with self.conn.cursor() as cur:
            cur.execute("SELECT item_idx, title FROM items WHERE item_idx = ANY(%s)", ([int(i) for i in item_ids],))
            return dict(cur.fetchall())

    def write_recommendations(self, run_id: str, recs: pd.DataFrame) -> int:
        """recs: columns user_idx, rank, item_idx, score."""
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM recommendations WHERE run_id = %s", (run_id,))
        _copy_df(self.conn, "recommendations", recs.assign(run_id=run_id)[["run_id", "user_idx", "rank", "item_idx", "score"]])
        self.conn.commit()
        return len(recs)
