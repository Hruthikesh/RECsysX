"""Ranking and beyond-accuracy metrics.

Everything works on a top-K matrix of item ids (n_users, K) plus the target items per
user. Ranking metrics are computed per user and then averaged over users.

Definitions used in the whole project:
  recall@K     hits in top-K / number of targets
  precision@K  hits in top-K / K
  ndcg@K       DCG / ideal DCG, ideal list has min(K, n_targets) hits
  map@K        sum of precision@k at every hit position / min(K, n_targets)
  mrr          1 / rank of the first hit inside the evaluated list (0 if none)
  hit@K        1 if at least one hit in top-K
"""
from __future__ import annotations

import numpy as np


def hit_matrix(topk: np.ndarray, targets: list[set] | list[np.ndarray]) -> np.ndarray:
    """Boolean (n_users, K) matrix: is the item at rank r a target for that user?"""
    hits = np.zeros(topk.shape, dtype=bool)
    for u, (row, tgt) in enumerate(zip(topk, targets)):
        if len(tgt) == 0:
            continue
        tgt = tgt if isinstance(tgt, (set, frozenset)) else set(np.asarray(tgt).tolist())
        hits[u] = [i in tgt for i in row]
    return hits


def ranking_metrics_per_user(hits: np.ndarray, n_targets: np.ndarray, ks: list[int]) -> dict[str, np.ndarray]:
    """Per-user metric arrays from a hit matrix. ``hits`` must have at least max(ks) columns."""
    n_targets = np.asarray(n_targets, dtype=np.float64)
    max_k = hits.shape[1]
    discounts = 1.0 / np.log2(np.arange(2, max_k + 2))
    cum_hits = np.cumsum(hits, axis=1)
    out: dict[str, np.ndarray] = {}
    safe_t = np.maximum(n_targets, 1)
    for k in ks:
        if k > max_k:
            raise ValueError(f"k={k} larger than the ranked list ({max_k})")
        h = hits[:, :k]
        n_hit = cum_hits[:, k - 1]
        out[f"recall@{k}"] = n_hit / safe_t
        out[f"precision@{k}"] = n_hit / k
        dcg = (h * discounts[:k]).sum(axis=1)
        ideal_n = np.minimum(n_targets, k).astype(int)
        idcg = np.cumsum(discounts[:k])[np.maximum(ideal_n - 1, 0)]
        out[f"ndcg@{k}"] = np.where(ideal_n > 0, dcg / idcg, 0.0)
        prec_at = cum_hits[:, :k] / np.arange(1, k + 1)
        out[f"map@{k}"] = (prec_at * h).sum(axis=1) / np.maximum(np.minimum(n_targets, k), 1)
        out[f"hit@{k}"] = (n_hit > 0).astype(np.float64)
    first = np.where(hits.any(axis=1), hits.argmax(axis=1) + 1, 0)
    out["mrr"] = np.where(first > 0, 1.0 / np.maximum(first, 1), 0.0)
    return out


def ranking_metrics(topk: np.ndarray, targets, ks: list[int]) -> dict[str, float]:
    hits = hit_matrix(topk, targets)
    n_t = np.array([len(t) for t in targets])
    per_user = ranking_metrics_per_user(hits, n_t, ks)
    return {k: float(v.mean()) for k, v in per_user.items()}


# ---------------------------------------------------------------------- beyond accuracy

def catalog_coverage(topk: np.ndarray, n_items: int) -> float:
    return len(np.unique(topk)) / n_items


def exposure_counts(topk: np.ndarray, n_items: int) -> np.ndarray:
    return np.bincount(topk.ravel(), minlength=n_items)


def gini(values: np.ndarray) -> float:
    """Gini coefficient of a non-negative vector (0 = equal exposure, 1 = one item gets everything)."""
    x = np.sort(np.asarray(values, dtype=np.float64))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


def intra_list_diversity(topk: np.ndarray, item_vectors: np.ndarray) -> float:
    """Mean pairwise (1 - cosine similarity) inside each list, averaged over users.

    With genre vectors this measures how many different kinds of movies one list contains.
    """
    v = item_vectors / np.linalg.norm(item_vectors, axis=1, keepdims=True).clip(1e-12)
    k = topk.shape[1]
    if k < 2:
        return 0.0
    lists = v[topk]                                  # (U, K, d)
    sims = np.einsum("ukd,ujd->ukj", lists, lists)   # (U, K, K)
    iu = np.triu_indices(k, 1)
    return float((1.0 - sims[:, iu[0], iu[1]]).mean())


def novelty(topk: np.ndarray, item_popularity: np.ndarray, n_users: int) -> float:
    """Mean self-information -log2 p(i), p(i) = share of training users that liked i (+1 smoothing)."""
    p = (item_popularity + 1.0) / (n_users + 1.0)
    return float((-np.log2(p))[topk].mean())


def personalization(topk: np.ndarray, n_pairs: int = 20000, seed: int = 0) -> float:
    """1 - mean Jaccard overlap between the lists of random user pairs.

    0 means everybody gets the same list (pure popularity), 1 means no overlap at all.
    """
    n = len(topk)
    if n < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    a = rng.integers(0, n, n_pairs)
    b = rng.integers(0, n, n_pairs)
    keep = a != b
    a, b = a[keep], b[keep]
    k = topk.shape[1]
    sa = np.sort(topk[a], axis=1)
    sb = np.sort(topk[b], axis=1)
    inter = np.array([len(np.intersect1d(x, y, assume_unique=True)) for x, y in zip(sa, sb)])
    jacc = inter / (2 * k - inter)
    return float(1.0 - jacc.mean())


def head_tail_share(topk: np.ndarray, is_head: np.ndarray) -> dict[str, float]:
    flat = topk.ravel()
    head = is_head[flat].mean()
    return {"head_share": float(head), "tail_share": float(1.0 - head)}


def beyond_accuracy(topk: np.ndarray, n_items: int, genre_vectors: np.ndarray, item_popularity: np.ndarray,
                    n_train_users: int, is_head: np.ndarray, seed: int = 0) -> dict[str, float]:
    k = topk.shape[1]
    exp = exposure_counts(topk, n_items)
    out = {
        f"coverage@{k}": catalog_coverage(topk, n_items),
        f"diversity@{k}": intra_list_diversity(topk, genre_vectors),
        f"novelty@{k}": novelty(topk, item_popularity, n_train_users),
        f"personalization@{k}": personalization(topk, seed=seed),
        f"gini@{k}": gini(exp),
    }
    ht = head_tail_share(topk, is_head)
    out[f"tail_share@{k}"] = ht["tail_share"]
    return out
