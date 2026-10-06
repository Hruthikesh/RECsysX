"""Second stage: rank the FAISS candidates with LogReg / XGBoost / a neural ranker.

Protocol (no test information is used for training or selection):
  * retrieval models: trained on pre-cutoff data (main suite, seed 42)
  * ranker training data: candidates of the *validation* users, labels = their
    post-cutoff positives. 80% of the validation users fit the ranker, 20% are used for
    early stopping and for choosing the ranker / candidate count.
  * evaluation: candidates of the *test* users, same metrics as every other model.
    Recall is computed against all test positives, not only the ones that were
    retrieved, so "retrieval only" and "retrieval + ranking" are directly comparable.

    python scripts/run_ranking.py
"""
import argparse
import time

import joblib
import numpy as np
import pandas as pd
import torch

from recsysx.config import load_config, project_path
from recsysx.data import RecData
from recsysx.evaluation import Evaluator, flatten_result
from recsysx.experiments.artifacts import load_main_model
from recsysx.features import RankingFeatureBuilder, add_labels
from recsysx.features.ranking import pair_scorer
from recsysx.ranking import RANKERS
from recsysx.retrieval import FaissRetriever
from recsysx.utils import get_device, load_json, save_json, set_seed

SCORE_MODELS = ["mf", "two_tower", "node2vec", "graphsage", "gat", "lightgcn", "hybrid", "itemknn"]
GRAPH_SCORES = ["score_node2vec", "score_graphsage", "score_gat", "score_lightgcn", "score_hybrid"]
PAIR_FEATS = ["genre_affinity", "genre_rating", "pop_gap", "year_gap"]


def rank_lists(df: pd.DataFrame, scores: np.ndarray, n_users: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    """df rows are (user, candidate) in user-major order with the same N per user."""
    N = len(df) // n_users
    s = scores.reshape(n_users, N).astype(np.float64).copy()
    s[~df["valid"].to_numpy().reshape(n_users, N)] = -np.inf
    order = np.argsort(-s, axis=1, kind="stable")
    items = df["item"].to_numpy().reshape(n_users, N)
    ranked = np.take_along_axis(items, order, 1)
    return ranked[:, :k], np.take_along_axis(s, order, 1)


def truncate(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df[df["retrieval_rank"] < n].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retriever", default=None, help="default: best retriever from run_retrieval.py")
    ap.add_argument("--ns", nargs="*", type=int, default=[50, 100, 200])
    ap.add_argument("--rankers", nargs="*", default=["logreg", "xgb", "neural"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. paths.results_dir=...")
    args = ap.parse_args()
    set_seed(args.seed)
    cfg = load_config(overrides=args.set)
    data = RecData.load(cfg)
    device = get_device(cfg["device"])
    tables = project_path(cfg["paths"]["results_dir"]) / "tables"
    out_dir = project_path(cfg["paths"]["checkpoint_dir"]) / "ranking"
    out_dir.mkdir(parents=True, exist_ok=True)
    retriever_key = args.retriever or load_json(tables / "retrieval_meta.json")["best_retriever_by_val_recall@200"]

    models = {k: load_main_model(cfg, data, k, args.seed, device) for k in SCORE_MODELS}
    # pair features are computed on CPU, that is also what the latency numbers below measure
    scorers = {k: pair_scorer(m, "cpu") for k, m in models.items()}
    fb = RankingFeatureBuilder(data, scorers)
    retr_model = models[retriever_key]
    faiss_idx = FaissRetriever(retr_model.item_vectors(), "flat")
    U = retr_model.user_vectors()
    n_max = max(args.ns)

    feats = {}
    for role in ("val", "test"):
        users = data.eval_users(role)
        cand, sc = faiss_idx.search(U[users], n_max, data.history, users)
        t = time.perf_counter()
        df = add_labels(fb.build(users, cand, sc), data)
        print(f"{role}: {len(df)} rows, {df['label'].mean():.3%} positive, features in {time.perf_counter() - t:.1f}s")
        feats[role] = (users, df)
    cols = RankingFeatureBuilder.feature_columns(feats["test"][1])

    val_users, val_df = feats["val"]
    rng = np.random.default_rng(args.seed)
    es_users = rng.choice(val_users, int(0.2 * len(val_users)), replace=False)
    is_es = val_df["user"].isin(es_users).to_numpy()
    test_users, test_df = feats["test"]
    ev = Evaluator(data, "test", ks=cfg["eval"]["ks"], device=device)
    assert (ev.users == test_users).all()

    # held-out validation users are scored exactly like the test users: metrics against all
    # of their positives, including the ones the retriever missed
    es_sorted = np.sort(es_users)
    ev_es = Evaluator(data, "val", ks=cfg["eval"]["ks"], device=device, users=es_sorted)

    def es_metrics(es_df, scores):
        topk, _ = rank_lists(es_df, scores, len(es_sorted), 20)
        m = ev_es.metrics_from_topk(topk, by_group=False)["all"]
        return {"es_recall@20": m["recall@20"], "es_ndcg@20": m["ndcg@20"], "es_ndcg@10": m["ndcg@10"]}

    rows, trained = [], {}
    for n in args.ns:
        tr = truncate(val_df[~is_es], n)
        es = truncate(val_df[is_es], n)
        te = truncate(test_df, n)
        assert (es["user"].to_numpy()[::n] == es_sorted).all()
        X_tr, y_tr, g_tr = tr[cols].to_numpy(np.float32), tr["label"].to_numpy(), tr["user"].to_numpy()
        X_es, y_es, g_es = es[cols].to_numpy(np.float32), es["label"].to_numpy(), es["user"].to_numpy()
        X_te = te[cols].to_numpy(np.float32)

        # retrieval-only ordering
        topk, _ = rank_lists(te, -te["retrieval_rank"].to_numpy(float), len(test_users), 20)
        res = ev.metrics_from_topk(topk)
        rows.append({"n_candidates": n, "ranker": "retrieval only",
                     **es_metrics(es, -es["retrieval_rank"].to_numpy(float)), **flatten_result(res)})

        for name in args.rankers:
            ranker = RANKERS[name](seed=args.seed, device="cuda" if device.type == "cuda" else "cpu")
            t = time.perf_counter()
            # early stopping inside fit uses NDCG over the retrieved candidates of the es users
            ranker.fit(X_tr, y_tr, g_tr, X_es, y_es, g_es)
            fit_s = time.perf_counter() - t
            es_m = es_metrics(es, ranker.predict(X_es))
            t = time.perf_counter()
            s_te = ranker.predict(X_te)
            pred_ms = 1000 * (time.perf_counter() - t) / len(test_users)
            topk, _ = rank_lists(te, s_te, len(test_users), 20)
            res = ev.metrics_from_topk(topk)
            rows.append({"n_candidates": n, "ranker": name, **es_m, "fit_time_s": fit_s,
                         "batched_predict_ms_per_user": pred_ms, **ranker.info(), **flatten_result(res)})
            trained[(name, n)] = ranker
            print(f"N={n} {name:7s} es recall@20 {es_m['es_recall@20']:.4f} | test recall@20 "
                  f"{res['all']['recall@20']:.4f} ndcg@20 {res['all']['ndcg@20']:.4f}")
    res_df = pd.DataFrame(rows)
    res_df.insert(0, "retriever", retriever_key)
    res_df.to_csv(tables / "ranking_results.csv", index=False)

    # pick ranker + candidate count on the held-out validation users only (primary metric recall@20)
    learned = res_df[res_df["ranker"] != "retrieval only"]
    best = learned.sort_values("es_recall@20", ascending=False).iloc[0]
    best_name, best_n = best["ranker"], int(best["n_candidates"])
    best_ranker = trained[(best_name, best_n)]
    print(f"selected ranker: {best_name} with N={best_n} (held-out validation recall@20 {best['es_recall@20']:.4f})")

    # XGBoost feature importance + feature-group ablation at the selected N
    xgb = trained.get(("xgb", best_n))
    if xgb is not None:
        imp = pd.Series(xgb.feature_importance(cols)).sort_values(ascending=False)
        imp.rename("gain").to_frame().to_csv(tables / "ranking_feature_importance.csv", index_label="feature")
    groups = {
        "all features": cols,
        "without graph-model scores": [c for c in cols if c not in GRAPH_SCORES],
        "without any model scores": [c for c in cols if not c.startswith("score_")],
        "without hand-made pair features": [c for c in cols if c not in PAIR_FEATS],
        "only retrieval score + rank": ["retrieval_score", "retrieval_rank"],
    }
    abl = []
    tr, es, te = truncate(val_df[~is_es], best_n), truncate(val_df[is_es], best_n), truncate(test_df, best_n)
    for gname, gcols in groups.items():
        r = RANKERS["xgb"](seed=args.seed, device="cuda" if device.type == "cuda" else "cpu")
        r.fit(tr[gcols].to_numpy(np.float32), tr["label"].to_numpy(), tr["user"].to_numpy(),
              es[gcols].to_numpy(np.float32), es["label"].to_numpy(), es["user"].to_numpy())
        topk, _ = rank_lists(te, r.predict(te[gcols].to_numpy(np.float32)), len(test_users), 20)
        res = ev.metrics_from_topk(topk)
        abl.append({"features": gname, "n_features": len(gcols), "n_candidates": best_n,
                    **{k: v for k, v in flatten_result(res).items() if "@" in k or k == "mrr"}})
    pd.DataFrame(abl).to_csv(tables / "ranking_feature_ablation.csv", index=False)

    # online latency: one user at a time, CPU, retrieval -> features -> ranker
    lat = []
    for u in rng.choice(test_users, 200, replace=False):
        t0 = time.perf_counter()
        cand, sc = faiss_idx.search(U[[u]], best_n, data.history, np.array([u]))
        t1 = time.perf_counter()
        f = fb.build(np.array([u]), cand, sc)
        t2 = time.perf_counter()
        s = best_ranker.predict(f[cols].to_numpy(np.float32))
        top = f["item"].to_numpy()[np.argsort(-s)[:10]]
        t3 = time.perf_counter()
        lat.append({"retrieval_ms": 1000 * (t1 - t0), "features_ms": 1000 * (t2 - t1), "ranking_ms": 1000 * (t3 - t2),
                    "total_ms": 1000 * (t3 - t0), "n_out": len(top)})
    lat = pd.DataFrame(lat)
    lat_summary = {f"{c}_{stat}": float(getattr(lat[c], stat)() if stat != "p95" else lat[c].quantile(0.95))
                   for c in ("retrieval_ms", "features_ms", "ranking_ms", "total_ms") for stat in ("mean", "median", "p95")}
    lat_summary.update({"ranker": best_name, "n_candidates": best_n, "retriever": retriever_key, "device": "cpu",
                        "n_requests": len(lat)})
    save_json(lat_summary, tables / "pipeline_latency.json")

    # artifacts for the final pipeline + re-ranking analysis
    joblib.dump(best_ranker, out_dir / f"ranker_{best_name}.joblib")
    save_json({"retriever": retriever_key, "ranker": best_name, "n_candidates": best_n, "feature_columns": cols,
               "score_models": SCORE_MODELS, "seed": args.seed}, out_dir / "ranking_meta.json")
    for role, df, mask in (("es", val_df, is_es), ("test", test_df, None)):
        sub = truncate(df[mask] if mask is not None else df, best_n)
        order_users = sub["user"].to_numpy()[::best_n]
        ranked, scores = rank_lists(sub, best_ranker.predict(sub[cols].to_numpy(np.float32)), len(order_users), best_n)
        np.savez_compressed(out_dir / f"{role}_ranked.npz", users=order_users, items=ranked, scores=scores)
    print(res_df[["n_candidates", "ranker", "es_recall@20", "recall@10", "recall@20", "ndcg@10", "ndcg@20"]].to_string(index=False))
    print(lat_summary)
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
