"""Full-ranking evaluation shared by every model.

Each model only has to provide ``score_fn(user_ids) -> (B, n_items)`` scores. The
evaluator removes items the user already rated in the training period, takes the top-K
over the *whole* catalog (no sampled negatives, those overestimate quality and can
change model orderings) and computes the metrics per user group.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import scipy.sparse as sp
import torch

from ..data.dataset import RecData
from .metrics import beyond_accuracy, ranking_metrics_per_user

ScoreFn = Callable[[torch.Tensor], torch.Tensor]
GROUPS = ("warm", "sparse", "cold")


def mask_rows(scores: torch.Tensor, csr: sp.csr_matrix, rows: np.ndarray, value: float = -np.inf) -> None:
    """In place: scores[b, j] = value for every j stored in csr row rows[b]."""
    indptr = csr.indptr
    starts, ends = indptr[rows], indptr[rows + 1]
    lens = ends - starts
    if lens.sum() == 0:
        return
    cols = np.concatenate([csr.indices[s:e] for s, e in zip(starts, ends)])
    r = np.repeat(np.arange(len(rows)), lens)
    scores[torch.as_tensor(r, device=scores.device), torch.as_tensor(cols, device=scores.device, dtype=torch.long)] = value


def topk_items(score_fn: ScoreFn, users: np.ndarray, history: sp.csr_matrix | None, k: int,
               device: torch.device, batch_size: int = 1024, item_mask: np.ndarray | None = None) -> np.ndarray:
    """Top-k item ids per user after removing already-seen items.

    item_mask (bool, n_items): if given, only these items can be recommended.
    """
    out = []
    allowed = None
    if item_mask is not None:
        allowed = torch.as_tensor(~item_mask, device=device)
    for start in range(0, len(users), batch_size):
        rows = users[start:start + batch_size]
        with torch.no_grad():
            scores = score_fn(torch.as_tensor(rows, device=device, dtype=torch.long)).float().clone()
        if history is not None:
            mask_rows(scores, history, rows)
        if allowed is not None:
            scores[:, allowed] = -np.inf
        out.append(scores.topk(k, dim=1).indices.cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, k), dtype=np.int64)


class Evaluator:
    def __init__(self, data: RecData, role: str, ks=(5, 10, 20), device: torch.device | str = "cpu",
                 batch_size: int = 1024, users: np.ndarray | None = None,
                 history: sp.csr_matrix | None = None, targets: sp.csr_matrix | None = None,
                 groups: np.ndarray | None = None):
        self.data = data
        self.role = role
        self.ks = sorted(ks)
        self.max_k = max(self.ks)
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.history = data.history if history is None else history
        self.targets = data.targets if targets is None else targets
        self.users = data.eval_users(role) if users is None else np.asarray(users)
        self.groups = data.group[self.users] if groups is None else np.asarray(groups)
        self.target_lists = [self.targets.indices[self.targets.indptr[u]:self.targets.indptr[u + 1]]
                             for u in self.users]
        self.n_targets = np.array([len(t) for t in self.target_lists])

    def topk(self, score_fn: ScoreFn, k: int | None = None, item_mask: np.ndarray | None = None) -> np.ndarray:
        return topk_items(score_fn, self.users, self.history, k or self.max_k, self.device,
                          self.batch_size, item_mask)

    def hits(self, topk: np.ndarray, item_subset: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Hit matrix and per-user number of targets (optionally only targets inside item_subset)."""
        hits = np.zeros(topk.shape, dtype=bool)
        n_t = np.zeros(len(topk), dtype=np.int64)
        for u, (row, tgt) in enumerate(zip(topk, self.target_lists)):
            if item_subset is not None:
                tgt = tgt[item_subset[tgt]]
            n_t[u] = len(tgt)
            if len(tgt):
                hits[u] = np.isin(row, tgt)
        return hits, n_t

    def metrics_from_topk(self, topk: np.ndarray, item_subset: np.ndarray | None = None,
                          by_group: bool = True) -> dict:
        hits, n_t = self.hits(topk, item_subset)
        valid = n_t > 0
        per_user = ranking_metrics_per_user(hits, n_t, self.ks)
        res = {"all": {m: float(v[valid].mean()) if valid.any() else float("nan") for m, v in per_user.items()}}
        res["all"]["n_users"] = int(valid.sum())
        if by_group:
            for g in GROUPS:
                sel = valid & (self.groups == g)
                res[g] = {m: float(v[sel].mean()) if sel.any() else float("nan") for m, v in per_user.items()}
                res[g]["n_users"] = int(sel.sum())
        res["_per_user"] = {m: v for m, v in per_user.items()}
        res["_valid"] = valid
        return res

    def evaluate(self, score_fn: ScoreFn, beyond: bool = True, beyond_k: int = 10) -> tuple[dict, np.ndarray]:
        topk = self.topk(score_fn)
        res = self.metrics_from_topk(topk)
        if beyond:
            d = self.data
            res["beyond"] = beyond_accuracy(
                topk[:, :beyond_k], d.n_items, d.genre_matrix(), d.item_popularity(),
                int(d.known_users().sum()), d.items["is_head"].to_numpy())
        return res, topk


def flatten_result(res: dict, prefix: str = "") -> dict:
    """{'all': {'recall@20': x}, 'warm': {...}} -> {'recall@20': x, 'warm/recall@20': ...}"""
    flat = {}
    for group, vals in res.items():
        if group.startswith("_"):
            continue
        for m, v in vals.items():
            key = m if group in ("all", "beyond") else f"{group}/{m}"
            flat[prefix + key] = v
    return flat
