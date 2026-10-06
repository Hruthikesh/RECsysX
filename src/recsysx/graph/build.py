"""User-item bipartite graph.

Node ids: users are 0..n_users-1, items are n_users..n_users+n_items-1.
Edges: one undirected edge per training positive (rating >= 4). Low ratings are not
edges. Treating "watched and disliked" as a connection would make a user look similar
to the people who loved the movie, which is the opposite of what the rating says.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch


def bipartite_edges(pairs: np.ndarray, n_users: int) -> np.ndarray:
    """(E, 2) user-item pairs -> (E, 2) node-id pairs with the item offset applied."""
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    return np.stack([pairs[:, 0], pairs[:, 1] + n_users], axis=1)


def edge_index(pairs: np.ndarray, n_users: int, undirected: bool = True) -> torch.Tensor:
    e = torch.as_tensor(bipartite_edges(pairs, n_users).T)
    if undirected:
        e = torch.cat([e, e.flip(0)], dim=1)
    return e


def adjacency(pairs: np.ndarray, n_users: int, n_items: int) -> sp.csr_matrix:
    """Symmetric (N, N) adjacency of the bipartite graph."""
    N = n_users + n_items
    e = bipartite_edges(pairs, n_users)
    rows = np.concatenate([e[:, 0], e[:, 1]])
    cols = np.concatenate([e[:, 1], e[:, 0]])
    a = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(N, N))
    a.sum_duplicates()
    a.data[:] = 1.0
    return a
