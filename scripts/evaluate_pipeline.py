"""Evaluate the integrated pipeline on the test users, end to end.

Runs retrieval -> features -> ranker -> MMR exactly like a request would, for all test
users in one batch (metrics) and then one user at a time (latency).

    python scripts/evaluate_pipeline.py              # history from the in-memory data
    python scripts/evaluate_pipeline.py --use-db     # history read from PostgreSQL per request
"""
import argparse

import numpy as np
import pandas as pd

from recsysx.config import load_config, project_path
from recsysx.data import RecData
from recsysx.evaluation import Evaluator, flatten_result
from recsysx.evaluation.metrics import beyond_accuracy
from recsysx.pipeline import RecommendationPipeline
from recsysx.utils import save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-db", action="store_true")
    ap.add_argument("--database-url", default=None)
    ap.add_argument("--latency-requests", type=int, default=200)
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. paths.results_dir=...")
    args = ap.parse_args()
    cfg = load_config(overrides=args.set)
    data = RecData.load(cfg)
    tables = project_path(cfg["paths"]["results_dir"]) / "tables"
    store = None
    if args.use_db:
        from recsysx.data.db import PostgresStore, connect
        store = PostgresStore(connect(args.database_url))
    users = data.eval_users("test")
    ev = Evaluator(data, "test", ks=cfg["eval"]["ks"])

    rows = []
    # "temporal" is the real final pipeline (ranker trained only on data before the cutoff), the
    # "validation" ranker is reported next to it as the optimistic same-period version
    for variant in ("temporal", "validation"):
        pipe = RecommendationPipeline.from_artifacts(cfg, data, store=store, ranker_variant=variant)
        for diversify in (False, True):
            if diversify and pipe.mmr_lambda >= 1.0:
                continue
            top, _, timings = pipe.recommend_batch(users, k=20, diversify=diversify)
            res = ev.metrics_from_topk(top)
            label = "Final pipeline" if variant == "temporal" else "Pipeline, same-period ranker"
            name = label + (f" + MMR (lambda={pipe.mmr_lambda})" if diversify else " (no MMR)")
            row = {"pipeline": name, "ranker_variant": variant, "history_source": "postgres" if store else "memory",
                   **flatten_result(res),
                   **beyond_accuracy(top[:, :10], data.n_items, data.genre_matrix(), data.item_popularity(),
                                     int(data.known_users().sum()), data.items["is_head"].to_numpy()),
                   **{f"batched_{k}": v for k, v in timings.items()}}
            rows.append(row)
            print(f"{name}: recall@20 {row['recall@20']:.4f} ndcg@20 {row['ndcg@20']:.4f} "
                  f"cold recall@20 {row['cold/recall@20']:.4f} diversity@10 {row['diversity@10']:.3f}")
    pipe = RecommendationPipeline.from_artifacts(cfg, data, store=store, ranker_variant="temporal")

    # single-request latency (what one API call would cost), CPU
    rng = np.random.default_rng(0)
    lat = []
    for u in rng.choice(users, min(args.latency_requests, len(users)), replace=False):
        rec = pipe.recommend(int(u), k=10)
        lat.append(rec.timings_ms)
    lat = pd.DataFrame(lat)
    summary = {f"{c}_mean": float(lat[c].mean()) for c in lat.columns}
    summary.update({f"{c}_p95": float(lat[c].quantile(0.95)) for c in lat.columns})
    summary.update({"n_requests": len(lat), "history_source": "postgres" if store else "memory",
                    "retriever": pipe.retriever_key, "ranker": "xgb (temporal)", "n_candidates": pipe.n_candidates,
                    "mmr_lambda": pipe.mmr_lambda})
    suffix = "_db" if store else ""
    pd.DataFrame(rows).to_csv(tables / f"final_pipeline_results{suffix}.csv", index=False)
    save_json(summary, tables / f"final_pipeline_latency{suffix}.json")
    print({k: round(v, 2) for k, v in summary.items() if isinstance(v, float)})


if __name__ == "__main__":
    main()
