"""Get recommendations for one or more users from the final pipeline.

    python scripts/recommend.py --user 1 --k 10
    python scripts/recommend.py --user 1 2 3 --to-db --run-id demo     # also store them in PostgreSQL

Users are the internal user_idx (0..6039); the original MovieLens id is shown as well.
"""
import argparse

import pandas as pd

from recsysx.config import load_config
from recsysx.data import RecData
from recsysx.pipeline import RecommendationPipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", nargs="+", type=int, required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--no-mmr", action="store_true")
    ap.add_argument("--to-db", action="store_true", help="write the recommendations to PostgreSQL")
    ap.add_argument("--use-db", action="store_true", help="read the user history from PostgreSQL")
    ap.add_argument("--database-url", default=None)
    ap.add_argument("--run-id", default="cli")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. paths.results_dir=...")
    args = ap.parse_args()
    cfg = load_config(overrides=args.set)
    data = RecData.load(cfg)
    store = None
    if args.to_db or args.use_db:
        from recsysx.data.db import PostgresStore, connect
        store = PostgresStore(connect(args.database_url))
    pipe = RecommendationPipeline.from_artifacts(cfg, data, store=store if args.use_db else None)
    titles = data.items["title"].to_numpy()
    genres = data.items["genres"].to_numpy()
    out = []
    for u in args.user:
        rec = pipe.recommend(u, k=args.k, diversify=not args.no_mmr)
        urow = data.users.iloc[u]
        hist = data.history[u].indices
        liked = data.train_ratings[(data.train_ratings["user_idx"] == u) & (data.train_ratings["rating"] >= 4)]
        print(f"\nuser_idx {u} (MovieLens id {urow['user_id']}, {urow['gender']}, age group {urow['age']}, "
              f"{urow['group']} user, {len(hist)} ratings before the cutoff)")
        for i in liked.sort_values("timestamp")["item_idx"].to_numpy()[-5:]:
            print(f"   liked: {titles[i]}  [{genres[i]}]")
        for r, (i, s) in enumerate(zip(rec.items, rec.scores), 1):
            print(f"  {r:2d}. {titles[i]}  [{genres[i]}]  score={s:.3f}")
        print("  timing (ms): " + ", ".join(f"{k}={v:.1f}" for k, v in rec.timings_ms.items()))
        out += [{"user_idx": u, "rank": r, "item_idx": i, "score": s}
                for r, (i, s) in enumerate(zip(rec.items, rec.scores), 1)]
    if args.to_db:
        n = store.write_recommendations(args.run_id, pd.DataFrame(out))
        print(f"\nwrote {n} rows to recommendations (run_id={args.run_id})")


if __name__ == "__main__":
    main()
