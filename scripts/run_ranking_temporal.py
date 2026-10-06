"""Strictly temporal version of the ranking experiment (checks how much of the ranking gain is real).

In scripts/run_ranking.py the ranker learns from validation users whose labels come from the same
period after the cutoff as the test users' labels. Nothing about the test users leaks, but the
ranker can learn how strongly e.g. "recent movie" or "recently popular" predicts a like *in that
period*, information a real system would not have yet.

Here everything the ranker learns from lies before the cutoff T:
  * a second cutoff T0 (70% quantile of the timestamps)
  * all score models are retrained on positives before T0 (selected configs, number of epochs =
    best epoch of the main seed-42 run, no early stopping on anything after T0)
  * ranker training users = users with positives in [T0, T), labels = those positives
  * the ranker is then applied unchanged to the test users with the normal models/features (as of T)

    python scripts/run_ranking_temporal.py
"""
import argparse

import joblib
import numpy as np
import pandas as pd
import torch

from recsysx.config import load_config, project_path
from recsysx.data import RecData, to_csr
from recsysx.data.split import assign_groups
from recsysx.evaluation import Evaluator, flatten_result
from recsysx.experiments.artifacts import load_main_model
from recsysx.experiments.suites import selected_model_cfg
from recsysx.features import RankingFeatureBuilder, add_labels
from recsysx.features.ranking import pair_scorer
from recsysx.models import build_recommender
from recsysx.ranking import RANKERS
from recsysx.retrieval import FaissRetriever
from recsysx.utils import get_device, load_json, set_seed

SCORE_MODELS = ["mf", "two_tower", "node2vec", "graphsage", "gat", "lightgcn", "hybrid", "itemknn"]


def data_as_of(data: RecData, t0: int, sparse_max: int) -> RecData:
    inter = data.interactions
    before = inter[inter["timestamp"] < t0]
    window = inter[(inter["timestamp"] >= t0) & (inter["timestamp"] < data.cutoff) & inter["positive"]]
    pos = before[before["positive"]]
    shape = (data.n_users, data.n_items)
    users_w = np.unique(window["user_idx"].to_numpy())
    role = np.full(data.n_users, "none", dtype=object)
    role[users_w] = "val"
    counts = np.bincount(pos["user_idx"], minlength=data.n_users)
    return data.copy(
        train_pos=pos[["user_idx", "item_idx"]].to_numpy(dtype=np.int64, copy=True),
        train_ratings=before[["user_idx", "item_idx", "rating", "timestamp"]].reset_index(drop=True),
        history=to_csr(before["user_idx"].to_numpy(), before["item_idx"].to_numpy(), shape),
        targets=to_csr(window["user_idx"].to_numpy(), window["item_idx"].to_numpy(), shape),
        role=role, group=assign_groups(counts, sparse_max), meta={**data.meta, "cutoff_timestamp": int(t0)},
        tag="as_of_T0")


def rank_topk(df: pd.DataFrame, scores: np.ndarray, n_users: int, k: int = 20) -> np.ndarray:
    N = len(df) // n_users
    s = scores.reshape(n_users, N).astype(np.float64).copy()
    s[~df["valid"].to_numpy().reshape(n_users, N)] = -np.inf
    order = np.argsort(-s, axis=1, kind="stable")[:, :k]
    return np.take_along_axis(df["item"].to_numpy().reshape(n_users, N), order, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quantile", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    set_seed(args.seed)
    cfg = load_config(overrides=args.set)
    device = get_device(cfg["device"])
    data = RecData.load(cfg)
    tables = project_path(cfg["paths"]["results_dir"]) / "tables"
    meta = load_json(project_path(cfg["paths"]["checkpoint_dir"]) / "ranking" / "ranking_meta.json")
    n_cand, cols = meta["n_candidates"], meta["feature_columns"]
    main_runs = pd.read_csv(tables / "main_runs.csv").set_index("run")

    t0 = int(np.sort(data.interactions["timestamp"].to_numpy())[int(args.quantile * len(data.interactions))])
    early = data_as_of(data, t0, cfg["split"]["user_sparse_max"])
    users_w = early.eval_users("val")
    print(f"T0 = {pd.to_datetime(t0, unit='s')}, {len(early.train_pos)} positives before T0, "
          f"{len(users_w)} ranker-training users with positives in [T0, T)")

    early_models = {}
    for key in SCORE_MODELS:
        mcfg = selected_model_cfg(key, cfg)
        if "epochs" in mcfg:
            mcfg["epochs"] = int(main_runs.loc[f"{key}_s{args.seed}", "best_epoch"])
        mcfg["seed"] = args.seed
        set_seed(args.seed)
        early_models[key] = build_recommender(mcfg, device).fit(early)   # no evaluator -> fixed epochs
        print("trained", key, "on data before T0")
    fb0 = RankingFeatureBuilder(early, {k: pair_scorer(m, "cpu") for k, m in early_models.items()})
    retr0 = early_models[meta["retriever"]]
    cand0, sc0 = FaissRetriever(retr0.item_vectors(), "flat").search(retr0.user_vectors()[users_w], n_cand,
                                                                      early.history, users_w)
    df0 = add_labels(fb0.build(users_w, cand0, sc0), early)
    rng = np.random.default_rng(args.seed)
    es = rng.choice(users_w, int(0.2 * len(users_w)), replace=False)
    is_es = df0["user"].isin(es).to_numpy()
    tr, va = df0[~is_es], df0[is_es]
    temporal = RANKERS["xgb"](seed=args.seed, device="cuda" if device.type == "cuda" else "cpu")
    temporal.fit(tr[cols].to_numpy(np.float32), tr["label"].to_numpy(), tr["user"].to_numpy(),
                 va[cols].to_numpy(np.float32), va["label"].to_numpy(), va["user"].to_numpy())

    ck = project_path(cfg["paths"]["checkpoint_dir"]) / "ranking"
    joblib.dump(temporal, ck / "ranker_xgb_temporal.joblib")

    # validation and test users: exactly the candidates and features of the normal pipeline
    models = {k: load_main_model(cfg, data, k, args.seed, device) for k in SCORE_MODELS}
    fb = RankingFeatureBuilder(data, {k: pair_scorer(m, "cpu") for k, m in models.items()})
    retr = models[meta["retriever"]]
    same_window = joblib.load(ck / f"ranker_{meta['ranker']}.joblib")
    rows = []
    for role in ("val", "test"):
        users = data.eval_users(role)
        cand, sc = FaissRetriever(retr.item_vectors(), "flat").search(retr.user_vectors()[users], n_cand,
                                                                      data.history, users)
        df = fb.build(users, cand, sc)
        X = df[cols].to_numpy(np.float32)
        ev = Evaluator(data, role, ks=cfg["eval"]["ks"], device=device)
        variants = [("retrieval only", -df["retrieval_rank"].to_numpy(float)),
                    ("xgb trained only on data before the cutoff (strictly temporal)", temporal.predict(X))]
        if role == "test":  # the same-period ranker was trained on validation users
            variants.insert(1, (f"{meta['ranker']} trained on validation users (same period as test)", same_window.predict(X)))
        for name, scores in variants:
            res = ev.metrics_from_topk(rank_topk(df, scores, len(users)))
            rows.append({"role": role, "ranker": name,
                         **{k: v for k, v in flatten_result(res).items() if "@" in k or k == "mrr"}})
        # ranked candidate lists of the temporal ranker, used by the re-ranking analysis
        N = len(df) // len(users)
        s_t = temporal.predict(X).reshape(len(users), N).astype(np.float64)
        s_t[~df["valid"].to_numpy().reshape(len(users), N)] = -np.inf
        order = np.argsort(-s_t, axis=1, kind="stable")
        items = np.take_along_axis(df["item"].to_numpy().reshape(len(users), N), order, 1)
        np.savez_compressed(ck / f"temporal_{role}_ranked.npz", users=users, items=items,
                            scores=np.take_along_axis(s_t, order, 1))
    out = pd.DataFrame(rows)
    out.to_csv(tables / "ranking_temporal_check.csv", index=False)
    imp = pd.Series(temporal.feature_importance(cols)).sort_values(ascending=False)
    imp.rename("gain").to_frame().to_csv(tables / "ranking_temporal_feature_importance.csv", index_label="feature")
    print(out[["role", "ranker", "recall@20", "ndcg@20", "warm/recall@20", "sparse/recall@20", "cold/recall@20"]].round(4).to_string(index=False))
    print((imp / imp.sum()).head(8).round(3))
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
