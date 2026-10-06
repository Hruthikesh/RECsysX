"""Train tiny versions of the models for a couple of epochs and check save/load round trips."""
import numpy as np
import pytest
import torch

from recsysx.config import load_config
from recsysx.evaluation import Evaluator
from recsysx.models import build_recommender

pytestmark = pytest.mark.slow

DEV = torch.device("cpu")


def _cfg(name, **over):
    c = load_config(name)["model_cfg"]
    c.update({"epochs": 2, "patience": 5, "batch_size": 256, "seed": 0, **over})
    return c


@pytest.mark.parametrize("name,over", [
    ("mf", {}),
    ("two_tower", {}),
    ("graphsage", {}),
    ("gat", {}),
    ("lightgcn", {}),
    ("hybrid", {}),
    ("node2vec", {"walks_per_node": 2, "walk_length": 6, "context_size": 3, "batch_size": 32}),
])
def test_train_save_load_same_scores(tiny_data, tmp_path, name, over):
    ev = Evaluator(tiny_data, "val", ks=[5])
    model = build_recommender(_cfg(name, **over), DEV).fit(tiny_data, ev)
    users = torch.as_tensor(ev.users)
    before = model.score(users)
    model.save(tmp_path / name)
    loaded = build_recommender(_cfg(name, **over), DEV).load(tmp_path / name, tiny_data)
    after = loaded.score(users)
    assert torch.allclose(before, after, atol=1e-5)
    assert before.shape == (len(users), tiny_data.n_items)
    assert torch.isfinite(before).all()


def test_cold_users_share_the_unknown_vector(tiny_data):
    model = build_recommender(_cfg("mf"), DEV).fit(tiny_data)
    cold = np.where(~tiny_data.known_users())[0]
    assert len(cold) >= 2
    U = model.user_vectors()
    assert np.allclose(U[cold[0]], U[cold[1]])


def test_popularity_and_itemknn(tiny_data, tmp_path):
    pop = build_recommender({"model": "popularity", "variant": "global"}, DEV).fit(tiny_data)
    s = pop.score(torch.arange(3))
    assert torch.allclose(s[0], s[1])                       # not personalised
    counts = np.bincount(tiny_data.train_pos[:, 1], minlength=tiny_data.n_items)
    assert int(s[0].argmax()) == int(counts.argmax())
    demo = build_recommender({"model": "popularity", "variant": "demographic"}, DEV).fit(tiny_data)
    assert demo.score(torch.arange(4)).shape == (4, tiny_data.n_items)
    knn = build_recommender({"model": "itemknn", "top_k_neighbors": 10, "shrink": 1.0}, DEV).fit(tiny_data)
    knn.save(tmp_path / "knn")
    knn2 = build_recommender({"model": "itemknn"}, DEV).load(tmp_path / "knn", tiny_data)
    assert torch.allclose(knn.score(torch.arange(5)), knn2.score(torch.arange(5)))
