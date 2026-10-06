"""Cross-check the database against the Parquet files.

Runs every query in db/queries.sql, computes the same number with pandas and writes
results/tables/db_consistency.csv. Also reads the full ratings table back and checks it
is identical to data/processed/interactions.parquet.

    python scripts/db_check.py
"""
import argparse

import numpy as np
import pandas as pd

from recsysx.config import load_config, project_path
from recsysx.data import RecData
from recsysx.data.db import connect, load_interactions, named_queries, run_query


def pandas_values(data: RecData) -> dict:
    inter, users, items = data.interactions, data.users, data.items
    train = inter[inter["period"] == "train"]
    pos = train[train["positive"]]
    ev = inter[(inter["period"] == "eval") & inter["positive"]]
    role = users["role"].to_numpy()
    head = items["is_head"].to_numpy()
    cold_test = users[(users["role"] == "test") & (users["group"] == "cold")]["user_idx"]
    return {
        "n_users": len(users),
        "n_items": len(items),
        "n_ratings": len(inter),
        "n_train_positives": len(pos),
        "n_eval_positives_val_test_users": int(np.isin(role[ev["user_idx"]], ["val", "test"]).sum()),
        "matrix_density": len(inter) / (len(users) * len(items)),
        "latest_train_rating_before_first_eval": bool(train["timestamp"].max() < inter.loc[inter["period"] == "eval", "timestamp"].min()),
        "cold_test_users_with_train_positives": int(pos["user_idx"].isin(cold_test).any()),
        "top_item_by_train_positives": int(pos.groupby("item_idx").size().sort_values(ascending=False, kind="stable").index[0]),
        "share_of_positives_on_head_items": float(head[pos["item_idx"]].mean()),
        "median_user_train_ratings": float(train.groupby("user_idx").size().median()),
        "drama_share_of_items": float(items["genres"].str.contains("Drama").mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--database-url", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="config overrides")
    args = ap.parse_args()
    cfg = load_config(overrides=args.set)
    data = RecData.load(cfg)
    expected = pandas_values(data)
    rows = []
    with connect(args.database_url) as conn:
        for name, sql in named_queries().items():
            got = run_query(conn, sql)
            got = got if isinstance(got, bool) else float(got)  # numeric comes back as Decimal
            exp = expected[name]
            if isinstance(exp, bool):
                ok = bool(got) == exp
            else:
                ok = bool(np.isclose(float(got), float(exp), rtol=1e-9, atol=1e-12))
            rows.append({"check": name, "postgres": got, "pandas": exp, "match": ok})
        db_inter = load_interactions(conn, data.meta["positive_threshold"])
    parquet = data.interactions.reset_index(drop=True)
    same = all((db_inter[c].to_numpy() == parquet[c].to_numpy()).all()
               for c in ["user_idx", "item_idx", "rating", "timestamp", "positive", "period"])
    rows.append({"check": "ratings_table_roundtrip_identical", "postgres": len(db_inter), "pandas": len(parquet),
                 "match": bool(same)})
    df = pd.DataFrame(rows)
    out = project_path(cfg["paths"]["results_dir"]) / "tables" / "db_consistency.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\n{df['match'].sum()}/{len(df)} checks match -> {out}")


if __name__ == "__main__":
    main()
