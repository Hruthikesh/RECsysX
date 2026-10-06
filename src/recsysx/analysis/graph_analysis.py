"""Figures for the training graph."""
from __future__ import annotations

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch

from ..data.dataset import RecData
from ..graph.stats import degree_arrays, sample_subgraph
from ..plotting import plt, save


def plot_degree_distribution(data: RecData, results_dir="results"):
    du, di = degree_arrays(data.train_pos, data.n_users, data.n_items)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for deg, label, c in ((du, "user nodes", "#1f77b4"), (di, "item nodes", "#d62728")):
        d = np.sort(deg[deg > 0])
        ccdf = 1.0 - np.arange(len(d)) / len(d)
        ax[0].loglog(d, ccdf, label=label, color=c)
    ax[0].set_xlabel("degree (training positives)")
    ax[0].set_ylabel("P(degree >= x)")
    ax[0].set_title("Degree distribution (CCDF, log-log)")
    ax[0].legend()
    # degree of the users we evaluate, per group
    roles = data.role
    for g, c in (("warm", "#2ca02c"), ("sparse", "#ff7f0e")):
        sel = (roles != "none") & (data.group == g)
        ax[1].hist(du[sel], bins=np.logspace(0, 3.3, 30), alpha=0.7, color=c, label=f"{g} ({sel.sum()})")
    n_cold = int(((roles != "none") & (data.group == "cold")).sum())
    ax[1].set_xscale("log")
    ax[1].set_xlabel("user degree")
    ax[1].set_title(f"Degree of val/test users ({n_cold} cold users have degree 0)")
    ax[1].legend()
    return save(fig, "graph_degree_distribution.png", results_dir)


def plot_sampled_subgraph(data: RecData, results_dir="results", seed: int = 3):
    pairs = sample_subgraph(data.train_pos, data.n_users, n_seed_users=20, max_items_per_user=10, seed=seed)
    G = nx.Graph()
    G.add_edges_from((f"u{u}", f"i{i}") for u, i in pairs)
    pop = data.item_popularity()
    pos = nx.spring_layout(G, seed=seed, k=0.35)
    users = [n for n in G if n.startswith("u")]
    items = [n for n in G if n.startswith("i")]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.25, width=0.6)
    nx.draw_networkx_nodes(G, pos, nodelist=users, node_color="#1f77b4", node_size=60, ax=ax, label="user")
    sizes = [10 + 60 * np.log1p(pop[int(n[1:])]) / np.log1p(pop.max()) for n in items]
    nx.draw_networkx_nodes(G, pos, nodelist=items, node_color="#d62728", node_size=sizes, ax=ax, alpha=0.8,
                           label="item (size = popularity)")
    shared = sum(1 for n in items if G.degree(n) > 1)
    ax.set_title(f"20 random users and up to 10 of their liked movies\n"
                 f"{shared} of {len(items)} movies are shared by 2+ of these users")
    ax.legend(loc="lower left")
    ax.axis("off")
    return save(fig, "graph_sampled_subgraph.png", results_dir)


def oversmoothing_curve(data: RecData, hops=(1, 3, 5), n_sample: int = 1000, seed: int = 0) -> dict:
    """Mean pairwise cosine between users when genre vectors are propagated with plain mean
    aggregation over the training graph (no learning involved)."""
    R = data.train_matrix().astype(np.float64)
    known = np.where(np.diff(R.indptr) > 0)[0]
    p_user = sp.diags(1 / np.maximum(np.asarray(R.sum(1)).ravel(), 1)) @ R
    p_item = sp.diags(1 / np.maximum(np.asarray(R.sum(0)).ravel(), 1)) @ R.T
    rng = np.random.default_rng(seed)
    sample = rng.choice(known, n_sample, replace=False)

    def mean_cos(X):
        X = X / np.linalg.norm(X, axis=1, keepdims=True).clip(1e-12)
        S = X @ X.T
        return float((S.sum() - len(X)) / (len(X) * (len(X) - 1)))

    out, x_item = {}, data.genre_matrix().astype(np.float64)
    for hop in range(1, max(hops) + 1, 2):
        x_user = p_user @ x_item
        if hop in hops:
            out[f"users_after_{hop}_steps"] = mean_cos(x_user[sample])
        x_item = p_item @ x_user
    out["random_movie_pairs"] = mean_cos(data.genre_matrix()[rng.choice(data.n_items, n_sample, replace=False)])
    return out


def walk_return_shares(data: RecData, settings=((1.0, 1.0), (4.0, 1.0), (0.25, 1.0), (1.0, 4.0)),
                       n_start: int = 2000, seed: int = 0) -> dict:
    """Share of node2vec walk steps that go straight back to the previous node, per (p, q)."""
    from torch_cluster import random_walk

    from ..graph import edge_index

    row, col = edge_index(data.train_pos, data.n_users)
    known = np.where(np.bincount(data.train_pos[:, 0], minlength=data.n_users) > 0)[0]
    start = torch.as_tensor(np.random.default_rng(seed).choice(known, n_start, replace=False))
    torch.manual_seed(seed)
    out = {}
    for p, q in settings:
        w = random_walk(row, col, start, walk_length=20, p=p, q=q, num_nodes=data.n_users + data.n_items)
        out[f"p={p:g},q={q:g}"] = float((w[:, 2:] == w[:, :-2]).float().mean())
    return out
