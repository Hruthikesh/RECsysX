"""Descriptive statistics of the training graph."""
from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import connected_components

from .build import adjacency


def graph_stats(pairs: np.ndarray, n_users: int, n_items: int) -> dict:
    A = adjacency(pairs, n_users, n_items)
    deg = np.asarray(A.sum(axis=1)).ravel()
    du, di = deg[:n_users], deg[n_users:]
    n_edges = int(A.nnz // 2)
    n_comp, labels = connected_components(A, directed=False)
    sizes = np.bincount(labels)
    # components ignoring isolated nodes (users/items with no training positives)
    non_iso = sizes[sizes > 1]

    # two-hop reach: how many other users share at least one liked item with a user
    R = A[:n_users, n_users:]
    co = (R @ R.T).tocsr()
    co.setdiag(0)
    co.eliminate_zeros()
    two_hop_users = np.diff(co.indptr)

    def summary(x):
        x = np.asarray(x, dtype=np.float64)
        return {"mean": float(x.mean()), "median": float(np.median(x)), "p90": float(np.percentile(x, 90)),
                "max": float(x.max()), "min": float(x.min())}

    return {
        "n_user_nodes": int(n_users),
        "n_item_nodes": int(n_items),
        "n_nodes": int(n_users + n_items),
        "n_edges": n_edges,
        "bipartite_density": n_edges / (n_users * n_items),
        "avg_degree_all": float(deg.mean()),
        "user_degree": summary(du),
        "item_degree": summary(di),
        "isolated_users": int((du == 0).sum()),
        "isolated_items": int((di == 0).sum()),
        "users_degree_le_5": int(((du > 0) & (du <= 5)).sum()),
        "items_degree_le_5": int(((di > 0) & (di <= 5)).sum()),
        "connected_components": int(n_comp),
        "non_trivial_components": int(len(non_iso)),
        "largest_component_nodes": int(sizes.max()),
        "users_two_hop_neighbours": summary(two_hop_users[du > 0]),
    }


def degree_arrays(pairs: np.ndarray, n_users: int, n_items: int) -> tuple[np.ndarray, np.ndarray]:
    du = np.bincount(pairs[:, 0], minlength=n_users)
    di = np.bincount(pairs[:, 1], minlength=n_items)
    return du, di


def sample_subgraph(pairs: np.ndarray, n_users: int, n_seed_users: int = 25, max_items_per_user: int = 8,
                    seed: int = 0) -> np.ndarray:
    """Small ego-style sample for plotting: random users and a few of their items."""
    rng = np.random.default_rng(seed)
    users = rng.choice(np.unique(pairs[:, 0]), n_seed_users, replace=False)
    out = []
    for u in users:
        items = pairs[pairs[:, 0] == u, 1]
        if len(items) > max_items_per_user:
            items = rng.choice(items, max_items_per_user, replace=False)
        out.append(np.stack([np.full(len(items), u), items], axis=1))
    return np.concatenate(out)

