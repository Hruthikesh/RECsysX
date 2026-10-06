import numpy as np
import pandas as pd

from recsysx.data.preprocess import clean_items, clean_users
from recsysx.data.split import assign_groups, split_eval_users, temporal_cutoff


def test_title_and_year_parsing():
    movies = pd.DataFrame({"movie_id": [1, 2], "title": ["Matrix, The (1999)", "Toy Story (1995)"],
                           "genres": ["Action|Sci-Fi", "Animation"]})
    items = clean_items(movies)
    assert items["year"].tolist() == [1999, 1995]
    assert items["title_clean"].tolist() == ["The Matrix", "Toy Story"]
    assert items["genre_list"][0] == ["Action", "Sci-Fi"]


def test_zip_region():
    users = pd.DataFrame({"zip": ["98107-2117", "02139", "T8H1N"]})
    assert clean_users(users)["zip_region"].tolist() == ["9", "0", "x"]


def test_temporal_cutoff_and_groups():
    ts = np.arange(100)
    assert temporal_cutoff(ts, 0.8) == 80
    groups = assign_groups(np.array([0, 1, 20, 21]), sparse_max=20)
    assert groups.tolist() == ["cold", "sparse", "sparse", "warm"]


def test_val_test_users_disjoint_and_stratified():
    groups = np.array(["warm"] * 40 + ["cold"] * 20 + ["sparse"] * 10)
    eligible = np.arange(70)
    roles = split_eval_users(eligible, groups, 0.5, seed=1)
    assert len(np.intersect1d(roles["val"], roles["test"])) == 0
    assert len(roles["val"]) + len(roles["test"]) == 70
    for g, n in (("warm", 40), ("cold", 20), ("sparse", 10)):
        assert (groups[roles["val"]] == g).sum() == n // 2


def test_processed_split_has_no_time_leakage(tiny_data):
    inter = tiny_data.interactions
    train_max = inter.loc[inter["period"] == "train", "timestamp"].max()
    eval_min = inter.loc[inter["period"] == "eval", "timestamp"].min()
    assert train_max < eval_min
    # models only ever see training positives
    pos = set(map(tuple, tiny_data.train_pos.tolist()))
    tr = inter[(inter["period"] == "train") & inter["positive"]]
    assert pos == set(zip(tr["user_idx"], tr["item_idx"]))


def test_unrated_movie_dropped_and_ids_contiguous(tiny_data):
    assert tiny_data.meta["preprocessing"]["movies_without_ratings"] == 1
    assert tiny_data.items["item_idx"].tolist() == list(range(tiny_data.n_items))
    assert tiny_data.users["user_idx"].tolist() == list(range(tiny_data.n_users))


def test_groups_match_training_counts(tiny_data):
    counts = np.bincount(tiny_data.train_pos[:, 0], minlength=tiny_data.n_users)
    g = tiny_data.group
    assert (counts[g == "cold"] == 0).all()
    assert ((counts[g == "sparse"] >= 1) & (counts[g == "sparse"] <= 10)).all()
    assert (counts[g == "warm"] > 10).all()


def test_content_matrices(tiny_data):
    assert tiny_data.item_content.shape[0] == tiny_data.n_items
    assert tiny_data.user_content.shape[0] == tiny_data.n_users
    # every movie has at least one genre
    assert (tiny_data.genre_matrix().sum(1) >= 1).all()


def test_train_fraction_keeps_evaluation(tiny_data):
    sub = tiny_data.with_train_fraction(0.5, seed=0)
    assert len(sub.train_pos) < len(tiny_data.train_pos)
    assert (sub.targets != tiny_data.targets).nnz == 0
    assert (sub.history != tiny_data.history).nnz == 0
    assert (sub.eval_users("test") == tiny_data.eval_users("test")).all()


def test_cold_items_removed_from_training(tiny_data):
    cold_data = tiny_data.with_cold_items(0.2, seed=0)
    cold = cold_data.notes["cold_items"]
    assert not np.isin(cold_data.train_pos[:, 1], cold).any()
    assert cold_data.history[:, cold].nnz == 0
    assert not cold_data.train_ratings["item_idx"].isin(cold).any()
    assert (cold_data.item_group[cold] == "cold").all()


def test_onboarding_targets_fixed_across_k(tiny_data):
    out = {k: tiny_data.onboarding("test", k, max_k=3, min_eval_pos=4) for k in (0, 1, 3)}
    users0, hist0, tgt0, rev0 = out[0]
    for k, (users, hist, tgt, rev) in out.items():
        assert (users == users0).all()
        assert (tgt != tgt0).nnz == 0
        assert (hist != hist0).nnz == 0
        assert all(len(r) == k for r in rev)
        # revealed items are never targets
        for u, r in zip(users, rev):
            assert not np.isin(r, tgt[u].indices).any()
