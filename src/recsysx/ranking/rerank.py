"""Diversity-aware re-ranking with Maximal Marginal Relevance (Carbonell & Goldstein, 1998).

Greedy: pick the next item that maximises
    lambda * relevance(i) - (1 - lambda) * max_{j already picked} sim(i, j)
lambda = 1 keeps the ranker order, smaller lambda trades relevance for diversity.
Relevance is min-max scaled per user so lambda means the same thing for every user.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def mmr_rerank(cand: np.ndarray, relevance: np.ndarray, item_vectors: np.ndarray, k: int, lam: float,
               device: str | torch.device = "cpu") -> np.ndarray:
    """cand, relevance: (n_users, N). Returns (n_users, k) re-ranked item ids."""
    dev = torch.device(device)
    c = torch.as_tensor(cand, device=dev, dtype=torch.long)
    rel = torch.as_tensor(relevance, device=dev, dtype=torch.float32).clone()
    valid = c >= 0
    rel = rel.masked_fill(~valid, float("nan"))
    lo = torch.nan_to_num(rel, nan=float("inf")).min(1, keepdim=True).values
    hi = torch.nan_to_num(rel, nan=float("-inf")).max(1, keepdim=True).values
    rel = torch.nan_to_num((rel - lo) / (hi - lo).clamp_min(1e-12), nan=0.0)
    if lam >= 1.0:
        order = rel.masked_fill(~valid, -1.0).argsort(dim=1, descending=True)[:, :k]
        return torch.gather(c, 1, order).cpu().numpy()

    v = F.normalize(torch.as_tensor(item_vectors, device=dev, dtype=torch.float32), dim=1)
    cv = v[c.clamp_min(0)]                                  # (U, N, d)
    sim = torch.einsum("und,umd->unm", cv, cv)              # (U, N, N)
    n_users, N = c.shape
    picked = torch.zeros((n_users, N), dtype=torch.bool, device=dev)
    max_sim = torch.zeros((n_users, N), device=dev)
    out = torch.empty((n_users, k), dtype=torch.long, device=dev)
    rows = torch.arange(n_users, device=dev)
    for step in range(k):
        score = lam * rel - (1 - lam) * max_sim if step else rel.clone()
        score = score.masked_fill(picked | ~valid, float("-inf"))
        j = score.argmax(1)
        out[:, step] = c[rows, j]
        picked[rows, j] = True
        max_sim = torch.maximum(max_sim, sim[rows, j])
    return out.cpu().numpy()
