"""Database round trip on a separate `recsysx_test` database (never touches the real tables).

Skipped when there is no DATABASE_URL and `pgserver` is not installed.
"""
import os

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.postgres


@pytest.fixture(scope="module")
def conn(tiny_cfg):
    psycopg = pytest.importorskip("psycopg")
    if not os.environ.get("DATABASE_URL"):
        pytest.importorskip("pgserver")
    from psycopg.conninfo import make_conninfo

    from recsysx.data.db import database_url

    base = database_url()
    with psycopg.connect(base, autocommit=True) as admin:
        admin.execute("DROP DATABASE IF EXISTS recsysx_test")
        admin.execute("CREATE DATABASE recsysx_test")
    c = psycopg.connect(make_conninfo(base, dbname="recsysx_test"))
    yield c
    c.close()


def test_load_and_roundtrip(conn, tiny_cfg, tiny_data):
    from recsysx.config import project_path
    from recsysx.data.db import create_schema, load_interactions, load_processed, run_query

    create_schema(conn, drop=True)
    counts = load_processed(conn, project_path(tiny_cfg["paths"]["processed_dir"]))
    assert counts["ratings"] == len(tiny_data.interactions)
    back = load_interactions(conn)
    ref = tiny_data.interactions.reset_index(drop=True)
    for col in ["user_idx", "item_idx", "rating", "timestamp", "positive", "period"]:
        assert (back[col].to_numpy() == ref[col].to_numpy()).all(), col
    assert run_query(conn, "SELECT count(*) FROM train_positives") == len(tiny_data.train_pos)


def test_store_profile_and_recommendations(conn, tiny_data):
    from recsysx.data.db import PostgresStore

    store = PostgresStore(conn)
    u = int(np.where(tiny_data.group == "warm")[0][0])
    prof = store.user_profile(u)
    assert sorted(prof["history_items"]) == sorted(tiny_data.history[u].indices.tolist())
    recs = pd.DataFrame({"user_idx": [u] * 3, "rank": [1, 2, 3], "item_idx": [0, 1, 2], "score": [0.9, 0.5, 0.1]})
    assert store.write_recommendations("test_run", recs) == 3
    assert store.write_recommendations("test_run", recs) == 3   # re-writing replaces, not duplicates
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM recommendations WHERE run_id = 'test_run'")
        assert cur.fetchone()[0] == 3
