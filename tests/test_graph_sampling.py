import numpy as np
import torch

from recsysx.data import NegativeSampler
from recsysx.graph import adjacency, bipartite_edges, edge_index, graph_stats
from recsysx.models.gnn import GNNModule


def test_negative_sampler_never_returns_positives():
    rng = np.random.default_rng(0)
    pos = np.stack([rng.integers(0, 20, 400), rng.integers(0, 15, 400)], axis=1)
    pos = np.unique(pos, axis=0)
    sampler = NegativeSampler(pos, n_items=15)
    users = torch.as_tensor(np.repeat(np.arange(20), 50))
    neg = sampler.sample(users, num=3)
    assert neg.shape == (1000, 3)
    pos_set = set(map(tuple, pos.tolist()))
    bad = [(u, i) for u, row in zip(users.tolist(), neg.tolist()) for i in row if (u, i) in pos_set]
    assert not bad


def test_bipartite_edges_offset_and_undirected():
    pairs = np.array([[0, 0], [0, 2], [1, 1]])
    e = bipartite_edges(pairs, n_users=2)
    assert e.tolist() == [[0, 2], [0, 4], [1, 3]]
    ei = edge_index(pairs, n_users=2)
    assert ei.shape == (2, 6)
    # every edge appears in both directions
    s = set(map(tuple, ei.T.tolist()))
    assert all((b, a) in s for a, b in s)
    # users only connect to items
    assert ((ei[0] < 2) != (ei[1] < 2)).all()


def test_adjacency_and_stats():
    pairs = np.array([[0, 0], [0, 1], [1, 1]])
    A = adjacency(pairs, 3, 2)          # user 2 has no edges
    assert (A != A.T).nnz == 0
    st = graph_stats(pairs, 3, 2)
    assert st["n_edges"] == 3
    assert st["isolated_users"] == 1
    assert st["bipartite_density"] == 3 / 6


def test_target_edges_removed_from_message_passing():
    pairs = np.array([[0, 0], [0, 1], [1, 1], [2, 0]])
    m = GNNModule(3, 2, torch.as_tensor(bipartite_edges(pairs, 3)), torch.ones(3, dtype=torch.bool),
                  torch.ones(2, dtype=torch.bool), num_layers=1, dim=8)
    full = m.message_edges()
    dropped = m.message_edges(drop=torch.tensor([1]))   # remove edge user0 - item1
    assert full.shape[1] == 8 and dropped.shape[1] == 6
    edges = set(map(tuple, dropped.T.tolist()))
    assert (0, 4) not in edges and (4, 0) not in edges
    assert (0, 3) in edges


def test_unknown_users_use_unknown_embedding():
    pairs = np.array([[0, 0], [1, 1]])
    known = torch.tensor([True, True, False])
    m = GNNModule(3, 2, torch.as_tensor(bipartite_edges(pairs, 3)), known, torch.ones(2, dtype=torch.bool),
                  num_layers=0, dim=4, user_id_dropout=0.1)
    ids = m.node_ids()
    assert ids[2].item() == m.N          # cold user -> unknown-user row
    assert ids[0].item() == 0


def test_dropped_users_lose_their_edges():
    pairs = np.array([[0, 0], [0, 1], [1, 1], [2, 0]])
    m = GNNModule(3, 2, torch.as_tensor(bipartite_edges(pairs, 3)), torch.ones(3, dtype=torch.bool),
                  torch.ones(2, dtype=torch.bool), num_layers=1, dim=8, user_id_dropout=0.5)
    drop = torch.tensor([True, False, False])
    e = m.message_edges(drop_users=drop)
    assert 0 not in e[0].tolist() and 0 not in e[1].tolist()     # user 0 is isolated now
    assert e.shape[1] == 4
    ids = m.node_ids(drop_users=drop)
    assert ids[0].item() == m.N and ids[1].item() == 1
