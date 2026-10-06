import numpy as np
import pandas as pd

from recsysx.features import RankingFeatureBuilder, add_labels
from recsysx.ranking import LogRegRanker, group_sizes, mmr_rerank, ndcg_from_scores


def _candidates(data, n=8):
    users = data.eval_users("test")
    rng = np.random.default_rng(0)
    cand = np.stack([rng.choice(data.n_items, n, replace=False) for _ in users])
    return users, cand, rng.random(cand.shape).astype(np.float32)


def test_feature_rows_and_labels(tiny_data):
    users, cand, sc = _candidates(tiny_data)
    fb = RankingFeatureBuilder(tiny_data, {"rand": lambda u, c: np.ones(c.shape)})
    df = add_labels(fb.build(users, cand, sc), tiny_data)
    assert len(df) == cand.size
    assert (df["user"].to_numpy()[::8] == users).all()
    t = tiny_data.targets
    expected = [int(t[u, i] != 0) for u, i in zip(df["user"], df["item"])]
    assert df["label"].tolist() == expected
    assert not df[RankingFeatureBuilder.feature_columns(df)].isna().any().any()


def test_features_do_not_use_eval_period(tiny_data):
    """Changing every rating after the cutoff must not change any feature."""
    users, cand, sc = _candidates(tiny_data)
    a = RankingFeatureBuilder(tiny_data).build(users, cand, sc)
    inter = tiny_data.interactions.copy()
    ev = inter["period"] == "eval"
    inter.loc[ev, "rating"] = 1
    inter.loc[ev, "positive"] = False
    b = RankingFeatureBuilder(tiny_data.copy(interactions=inter)).build(users, cand, sc)
    pd.testing.assert_frame_equal(a, b)


def test_cold_user_features_are_defined(tiny_data):
    cold = np.where(tiny_data.group == "cold")[0][:3]
    fb = RankingFeatureBuilder(tiny_data)
    df = fb.build(cold, np.tile(np.arange(5), (len(cold), 1)), np.zeros((len(cold), 5)))
    assert (df["u_is_cold"] == 1).all()
    assert not df.isna().any().any()


def test_mmr_lambda_one_keeps_order_and_no_duplicates():
    rng = np.random.default_rng(0)
    cand = np.stack([rng.permutation(30)[:20] for _ in range(4)])
    rel = rng.random(cand.shape)
    vec = rng.random((30, 5))
    out = mmr_rerank(cand, rel, vec, 10, 1.0)
    expected = np.take_along_axis(cand, np.argsort(-rel, axis=1), 1)[:, :10]
    assert (out == expected).all()
    out2 = mmr_rerank(cand, rel, vec, 10, 0.3)
    for row, c in zip(out2, cand):
        assert len(set(row)) == 10 and np.isin(row, c).all()


def test_mmr_increases_diversity():
    # items 0-9 are near-duplicates and most relevant, items 10-19 are all different
    vec = np.vstack([np.tile([1.0, 0, 0, 0], (10, 1)), np.eye(10, 4)[np.arange(10) % 4] + 0.01])
    cand = np.arange(20)[None, :]
    rel = np.r_[np.linspace(1, 0.9, 10), np.linspace(0.5, 0.4, 10)][None, :]
    base = mmr_rerank(cand, rel, vec, 5, 1.0)[0]
    div = mmr_rerank(cand, rel, vec, 5, 0.3)[0]
    assert (base < 10).all()
    assert (div >= 10).sum() >= 2


def test_group_sizes_and_ranker_ndcg():
    groups = np.array([5, 5, 5, 9, 9])
    assert group_sizes(groups).tolist() == [3, 2]
    y = np.array([0, 0, 1, 1, 0])
    perfect = np.array([0.1, 0.2, 0.9, 0.8, 0.1])
    assert ndcg_from_scores(perfect, y, groups, k=3) == 1.0
    X = np.c_[perfect, np.random.default_rng(0).random(5)]
    r = LogRegRanker().fit(X, y, groups)
    assert r.predict(X).shape == (5,)
