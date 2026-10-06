import numpy as np
import scipy.sparse as sp
import torch

from recsysx.evaluation import Evaluator, topk_items
from recsysx.retrieval import FaissRetriever, ann_overlap, candidate_recall


def test_seen_items_are_never_recommended():
    rng = np.random.default_rng(0)
    scores = torch.as_tensor(rng.random((5, 30)), dtype=torch.float32)
    history = sp.random(5, 30, density=0.3, format="csr", random_state=0)
    top = topk_items(lambda u: scores[u], np.arange(5), history, 10, torch.device("cpu"))
    for u in range(5):
        assert not np.isin(top[u], history[u].indices).any()
        # and the rest is still sorted by score
        allowed = np.setdiff1d(np.arange(30), history[u].indices)
        expected = allowed[np.argsort(-scores[u].numpy()[allowed], kind="stable")][:10]
        assert set(top[u]) == set(expected)


def test_item_mask_restricts_candidates():
    scores = torch.arange(20, dtype=torch.float32).repeat(2, 1)
    mask = np.zeros(20, dtype=bool)
    mask[[2, 5, 7]] = True
    top = topk_items(lambda u: scores[u], np.arange(2), None, 3, torch.device("cpu"), item_mask=mask)
    assert top.tolist() == [[7, 5, 2], [7, 5, 2]]


def test_evaluator_groups(tiny_data):
    ev = Evaluator(tiny_data, "test", ks=[5, 10])
    rng = np.random.default_rng(0)
    s = torch.as_tensor(rng.random((tiny_data.n_users, tiny_data.n_items)), dtype=torch.float32)
    res, top = ev.evaluate(lambda u: s[u])
    assert top.shape == (len(ev.users), 10)
    n = sum(res[g]["n_users"] for g in ("warm", "sparse", "cold"))
    assert n == res["all"]["n_users"] == len(ev.users)
    assert 0 <= res["all"]["recall@10"] <= 1


def test_faiss_exact_matches_bruteforce_with_filter():
    rng = np.random.default_rng(1)
    items = rng.normal(size=(200, 16)).astype(np.float32)
    users = rng.normal(size=(30, 16)).astype(np.float32)
    history = sp.random(30, 200, density=0.1, format="csr", random_state=1)
    retr = FaissRetriever(items, "flat")
    got, _ = retr.search(users, 20, history, np.arange(30))
    scores = torch.as_tensor(users @ items.T)
    expected = topk_items(lambda u: scores[u], np.arange(30), history, 20, torch.device("cpu"))
    assert ann_overlap(got, expected) == 1.0
    for u in range(30):
        assert not np.isin(got[u], history[u].indices).any()


def test_ivf_with_all_probes_is_exact_and_recall_helper():
    rng = np.random.default_rng(2)
    items = rng.normal(size=(300, 8)).astype(np.float32)
    users = rng.normal(size=(10, 8)).astype(np.float32)
    exact, _ = FaissRetriever(items, "flat").search(users, 10)
    ivf, _ = FaissRetriever(items, "ivf", nlist=8, nprobe=8).search(users, 10)
    assert ann_overlap(ivf, exact) == 1.0
    r = candidate_recall(exact, [exact[u, :2] for u in range(10)], [1, 5])
    assert r["recall@1"] == 0.5 and r["recall@5"] == 1.0
