"""Error analysis helpers: where do the models miss, and for which users/items?"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..data.dataset import RecData
from ..evaluation.metrics import ranking_metrics_per_user

ACTIVITY_BINS = [-1, 0, 5, 20, 50, 100, 200, 10_000]
ACTIVITY_LABELS = ["0", "1-5", "6-20", "21-50", "51-100", "101-200", ">200"]


def per_user_recall(topk: np.ndarray, users: np.ndarray, data: RecData, k: int = 20) -> np.ndarray:
    t = data.targets
    hits = np.zeros((len(users), k), dtype=bool)
    n_t = np.zeros(len(users))
    for j, u in enumerate(users):
        tgt = t.indices[t.indptr[u]:t.indptr[u + 1]]
        n_t[j] = len(tgt)
        hits[j] = np.isin(topk[j, :k], tgt)
    return ranking_metrics_per_user(hits, n_t, [k])[f"recall@{k}"]


def user_profile_stats(data: RecData) -> pd.DataFrame:
    pos = data.train_pos
    g = data.genre_matrix()
    prof = np.zeros((data.n_users, g.shape[1]))
    np.add.at(prof, pos[:, 0], g[pos[:, 1]])
    share = prof / prof.sum(1, keepdims=True).clip(1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -(np.where(share > 0, share * np.log(share), 0)).sum(1) / np.log(g.shape[1])
    deg_item = np.bincount(pos[:, 1], minlength=data.n_items)
    n_pos = np.bincount(pos[:, 0], minlength=data.n_users)
    mean_nb_deg = np.zeros(data.n_users)
    np.add.at(mean_nb_deg, pos[:, 0], deg_item[pos[:, 1]])
    mean_nb_deg = np.where(n_pos > 0, mean_nb_deg / n_pos.clip(1), np.nan)
    return pd.DataFrame({"train_pos": n_pos, "genre_entropy": np.where(n_pos > 0, ent, np.nan),
                         "top_genre_share": np.where(n_pos > 0, share.max(1), np.nan),
                         "mean_item_degree": mean_nb_deg})


def item_popularity_bucket(data: RecData) -> np.ndarray:
    pop = data.item_popularity()
    head = data.items["is_head"].to_numpy()
    out = np.where(head, "head (top 20%)", "mid")
    tail_cut = np.quantile(pop[~head], 0.5)
    out = np.where(~head & (pop <= tail_cut), "tail (bottom 40%)", out)
    out = np.where(pop == 0, "cold (0 train positives)", out)
    return out


def target_hit_rate_by_bucket(lists: dict[str, np.ndarray], users: np.ndarray, data: RecData, k: int = 20) -> pd.DataFrame:
    """Share of test targets in each item-popularity bucket that a model puts in its top-k."""
    bucket = item_popularity_bucket(data)
    t = data.targets
    rows = []
    for name, topk in lists.items():
        found = {b: [0, 0] for b in np.unique(bucket)}
        for j, u in enumerate(users):
            tgt = t.indices[t.indptr[u]:t.indptr[u + 1]]
            hit = np.isin(tgt, topk[j, :k])
            for b, h in zip(bucket[tgt], hit):
                found[b][0] += int(h)
                found[b][1] += 1
        for b, (h, n) in found.items():
            rows.append({"model": name, "item_bucket": b, "targets": n, "hit_rate@20": h / max(n, 1)})
    return pd.DataFrame(rows)


def graph_vs_mf(recall_graph: np.ndarray, recall_mf: np.ndarray, users: np.ndarray, data: RecData) -> tuple[pd.DataFrame, dict]:
    """Where does the GNN lose against MF? Relate the per-user difference to how popular the
    user's own items are (a user whose items all have thousands of neighbours gets a very
    averaged-out neighbourhood)."""
    stats = user_profile_stats(data).iloc[users].reset_index(drop=True)
    df = stats.assign(diff=recall_graph - recall_mf, recall_graph=recall_graph, recall_mf=recall_mf)
    df = df[df["train_pos"] > 0]
    df["neighbour_degree_quartile"] = pd.qcut(df["mean_item_degree"], 4, labels=["Q1 (nichest)", "Q2", "Q3", "Q4 (most popular)"])
    table = df.groupby("neighbour_degree_quartile", observed=True).agg(
        users=("diff", "size"), mean_item_degree=("mean_item_degree", "mean"), recall_graph=("recall_graph", "mean"),
        recall_mf=("recall_mf", "mean"), mean_diff=("diff", "mean")).reset_index()
    rho = spearmanr(df["mean_item_degree"], df["diff"])
    rho_deg = spearmanr(df["train_pos"], df["diff"])
    return table, {"spearman_diff_vs_mean_item_degree": float(rho.statistic), "p_value": float(rho.pvalue),
                   "spearman_diff_vs_user_degree": float(rho_deg.statistic), "p_value_user_degree": float(rho_deg.pvalue),
                   "n_users": int(len(df))}
