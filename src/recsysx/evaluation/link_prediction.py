"""Link-prediction view of the same models.

Recommendation here is link prediction on the user-item graph: a model is good if it
gives a higher score to a (user, item) edge that appears after the cutoff than to a
random non-edge. AUC/AP measure exactly that pairwise question. They do not care about
the top of the list, so a model can have a good AUC and still a weak recall@20
(popularity is the usual example). Both views are reported for that reason.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def sample_eval_edges(users: np.ndarray, targets, history, n_items: int, seed: int = 0,
                      neg_per_pos: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Positive edges = eval-window positives of the given users. Negatives are random
    items the user neither rated before the cutoff nor liked after it."""
    rng = np.random.default_rng(seed)
    pu, pi = [], []
    for u in users:
        t = targets.indices[targets.indptr[u]:targets.indptr[u + 1]]
        pu.append(np.full(len(t), u))
        pi.append(t)
    pu, pi = np.concatenate(pu), np.concatenate(pi)
    nu = np.repeat(pu, neg_per_pos)
    ni = rng.integers(0, n_items, len(nu))
    forbidden = []
    for m in (targets, history):
        coo = m[np.unique(nu)].tocoo()
        forbidden.append(np.unique(nu)[coo.row].astype(np.int64) * n_items + coo.col)
    forbidden = np.unique(np.concatenate(forbidden))
    for _ in range(20):
        bad = np.isin(nu.astype(np.int64) * n_items + ni, forbidden)
        if not bad.any():
            break
        ni[bad] = rng.integers(0, n_items, bad.sum())
    u_all = np.concatenate([pu, nu])
    i_all = np.concatenate([pi, ni])
    y = np.concatenate([np.ones(len(pu)), np.zeros(len(nu))])
    return u_all, i_all, y


def link_prediction_scores(score_fn, users: np.ndarray, items: np.ndarray, device, batch: int = 512) -> np.ndarray:
    """Score arbitrary (user, item) pairs with a full-row score function."""
    out = np.empty(len(users), dtype=np.float64)
    order = np.argsort(users, kind="stable")
    su, si = users[order], items[order]
    uniq, start = np.unique(su, return_index=True)
    bounds = list(start) + [len(su)]
    for b in range(0, len(uniq), batch):
        ub = uniq[b:b + batch]
        with torch.no_grad():
            scores = score_fn(torch.as_tensor(ub, device=device, dtype=torch.long)).float().cpu().numpy()
        for j in range(len(ub)):
            lo, hi = bounds[b + j], bounds[b + j + 1]
            out[order[lo:hi]] = scores[j, si[lo:hi]]
    return out


def auc_ap(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    s = np.nan_to_num(scores, nan=0.0, posinf=1e9, neginf=-1e9)
    return {"auc": float(roc_auc_score(labels, s)), "ap": float(average_precision_score(labels, s))}
