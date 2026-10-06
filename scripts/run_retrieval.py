"""Candidate generation experiment with FAISS.

For every embedding model from the main suite (seed 42 checkpoints):
  * build an exact inner-product index over the item vectors
  * retrieve the top-N unseen items for val and test users
  * candidate recall@N for N in 20..500, per user group
  * single-query latency (one user at a time, like an online request) and batched cost
Then for the best retriever: exact vs IVF vs HNSW (ANN overlap with exact search, recall, latency).

Retrieval is evaluated on its own here; ranking quality is in scripts/run_ranking.py.

    python scripts/run_retrieval.py
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch

from recsysx.config import load_config, project_path
from recsysx.data import RecData
from recsysx.evaluation import topk_items
from recsysx.experiments.suites import selected_model_cfg
from recsysx.models import build_recommender
from recsysx.retrieval import FaissRetriever, ann_overlap, candidate_recall
from recsysx.utils import get_device, save_json

EMBEDDING_MODELS = ["mf", "two_tower", "node2vec", "graphsage", "gat", "lightgcn", "hybrid"]
NS = [20, 50, 100, 200, 500]


def load_model(cfg, data, key, seed, device):
    m = build_recommender(selected_model_cfg(key, cfg), device)
    return m.load(project_path(cfg["paths"]["checkpoint_dir"]) / "main" / f"{key}_s{seed}", data)


def single_query_latency(retr: FaissRetriever, U: np.ndarray, users: np.ndarray, history, n: int,
                         n_queries: int = 300, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(users), min(n_queries, len(users)), replace=False)
    times = []
    for j in pick:
        t = time.perf_counter()
        retr.search(U[users[j]][None, :], n, history, users[j:j + 1])
        times.append(1000 * (time.perf_counter() - t))
    return {"latency_ms_mean": float(np.mean(times)), "latency_ms_p50": float(np.percentile(times, 50)),
            "latency_ms_p95": float(np.percentile(times, 95))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models", nargs="*", default=EMBEDDING_MODELS)
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. paths.results_dir=...")
    args = ap.parse_args()
    cfg = load_config(overrides=args.set)
    data = RecData.load(cfg)
    device = get_device(cfg["device"])
    tables = project_path(cfg["paths"]["results_dir"]) / "tables"
    rows, lat_rows = [], []

    for key in args.models:
        model = load_model(cfg, data, key, args.seed, device)
        U, I = model.user_vectors(), model.item_vectors()
        retr = FaissRetriever(I, "flat")
        for role in ("val", "test"):
            users = data.eval_users(role)
            targets = [data.targets.indices[data.targets.indptr[u]:data.targets.indptr[u + 1]] for u in users]
            t = time.perf_counter()
            cand, _ = retr.search(U[users], max(NS), data.history, users)
            batch_ms = 1000 * (time.perf_counter() - t) / len(users)
            rec = {"model": model.cfg.get("label", key), "key": key, "role": role, "index": "flat",
                   "batched_ms_per_user": batch_ms, "build_time_s": retr.build_time_s}
            rec.update(candidate_recall(cand, targets, NS))
            groups = data.group[users]
            for g in ("warm", "sparse", "cold"):
                sel = groups == g
                gr = candidate_recall(cand[sel], [targets[j] for j in np.where(sel)[0]], [100, 200])
                rec.update({f"{g}/{k}": v for k, v in gr.items()})
            rows.append(rec)
            if role == "test":
                # sanity check: FAISS exact search == brute-force torch top-k with the same filter
                brute = topk_items(lambda u: torch.as_tensor(U, device=device)[u] @ torch.as_tensor(I, device=device).T,
                                   users, data.history, 100, device)
                rec["faiss_vs_bruteforce_top100_overlap"] = ann_overlap(cand[:, :100], brute)
        for n in NS:
            users = data.eval_users("test")
            lat = single_query_latency(retr, U, users, data.history, n)
            lat_rows.append({"model": model.cfg.get("label", key), "key": key, "index": "flat", "n": n, **lat})
        print(f"{key}: test recall@100 {rows[-1]['recall@100']:.4f} recall@200 {rows[-1]['recall@200']:.4f}")
        torch.cuda.empty_cache()

    res = pd.DataFrame(rows)
    res.to_csv(tables / "retrieval_results.csv", index=False)

    # best retriever by validation recall@200 -> compare FAISS index types on it
    val = res[res["role"] == "val"].sort_values("recall@200", ascending=False)
    best_key = val.iloc[0]["key"]
    model = load_model(cfg, data, best_key, args.seed, device)
    U, I = model.user_vectors(), model.item_vectors()
    users = data.eval_users("test")
    targets = [data.targets.indices[data.targets.indptr[u]:data.targets.indptr[u + 1]] for u in users]
    exact = FaissRetriever(I, "flat")
    exact_c, _ = exact.search(U[users], 200, data.history, users)
    idx_rows = []
    # HNSW: efSearch gets raised to k anyway (k includes the history over-fetch), so the
    # graph size M is the HNSW setting that actually changes something here
    configs = [("flat", {})] + [("ivf", {"nlist": 64, "nprobe": p}) for p in (1, 4, 16)] + \
              [("hnsw", {"hnsw_m": m}) for m in (8, 32)]
    for kind, params in configs:
        r = FaissRetriever(I, kind, **params)
        t = time.perf_counter()
        c, _ = r.search(U[users], 200, data.history, users)
        batch_ms = 1000 * (time.perf_counter() - t) / len(users)
        lat = single_query_latency(r, U, users, data.history, 200)
        idx_rows.append({"model": model.cfg.get("label", best_key), "index": kind, "params": str(params),
                         "build_time_s": r.build_time_s, "index_bytes": r.memory_bytes(),
                         "ann_overlap@200_vs_exact": ann_overlap(c, exact_c), "batched_ms_per_user": batch_ms,
                         **candidate_recall(c, targets, [100, 200]), **lat})
    lat_df = pd.DataFrame(lat_rows)
    lat_df.to_csv(tables / "retrieval_latency.csv", index=False)
    pd.DataFrame(idx_rows).to_csv(tables / "retrieval_index_comparison.csv", index=False)
    save_json({"best_retriever_by_val_recall@200": best_key, "seed": args.seed,
               "faiss_threads": __import__("faiss").omp_get_max_threads(),
               "note": "latency measured on CPU (FAISS flat/IVF/HNSW, faiss-cpu)"},
              tables / "retrieval_meta.json")
    print(res[["model", "role", "recall@50", "recall@100", "recall@200", "recall@500"]].to_string(index=False))
    print(pd.DataFrame(idx_rows)[["index", "params", "ann_overlap@200_vs_exact", "recall@200", "latency_ms_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
