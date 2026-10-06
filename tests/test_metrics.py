import numpy as np
import pytest

from recsysx.evaluation.metrics import (catalog_coverage, gini, hit_matrix, intra_list_diversity, novelty,
                                        personalization, ranking_metrics, ranking_metrics_per_user)


def test_hand_computed_ranking_metrics():
    # one user, targets {1, 3, 9}; ranked list 1, 2, 3, 4, 5 -> hits at ranks 1 and 3
    topk = np.array([[1, 2, 3, 4, 5]])
    m = ranking_metrics(topk, [{1, 3, 9}], ks=[3, 5])
    assert m["recall@3"] == pytest.approx(2 / 3)
    assert m["precision@3"] == pytest.approx(2 / 3)
    assert m["precision@5"] == pytest.approx(2 / 5)
    dcg = 1 / np.log2(2) + 1 / np.log2(4)
    idcg3 = 1 / np.log2(2) + 1 / np.log2(3) + 1 / np.log2(4)
    assert m["ndcg@3"] == pytest.approx(dcg / idcg3)
    # AP@5 = (1/1 + 2/3) / min(5, 3)
    assert m["map@5"] == pytest.approx((1 + 2 / 3) / 3)
    assert m["mrr"] == pytest.approx(1.0)
    assert m["hit@3"] == 1.0


def test_ndcg_ideal_uses_min_k_targets():
    # a single target found at rank 1 is a perfect list even though K = 5
    m = ranking_metrics(np.array([[7, 1, 2, 3, 4]]), [{7}], ks=[5])
    assert m["ndcg@5"] == pytest.approx(1.0)
    assert m["recall@5"] == 1.0


def test_no_hits_and_mrr_rank():
    m = ranking_metrics(np.array([[1, 2, 3], [4, 5, 6]]), [{9}, {6}], ks=[3])
    assert m["recall@3"] == pytest.approx(0.5)
    assert m["mrr"] == pytest.approx((0 + 1 / 3) / 2)


def test_per_user_matches_aggregate():
    rng = np.random.default_rng(0)
    topk = rng.integers(0, 30, (50, 20))
    targets = [set(rng.choice(30, 5, replace=False).tolist()) for _ in range(50)]
    hits = hit_matrix(topk, targets)
    per = ranking_metrics_per_user(hits, np.array([5] * 50), [10, 20])
    agg = ranking_metrics(topk, targets, [10, 20])
    for k in agg:
        assert per[k].mean() == pytest.approx(agg[k])


def test_gini_extremes():
    assert gini(np.ones(10)) == pytest.approx(0.0)
    x = np.zeros(10)
    x[3] = 5
    assert gini(x) == pytest.approx(0.9)


def test_coverage_novelty_diversity_personalization():
    topk = np.array([[0, 1], [0, 1]])
    assert catalog_coverage(topk, 4) == 0.5
    assert personalization(topk, n_pairs=100) == pytest.approx(0.0)
    assert personalization(np.array([[0, 1], [2, 3]]), n_pairs=100) == pytest.approx(1.0)
    vec = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=float)
    assert intra_list_diversity(np.array([[0, 1]]), vec) == pytest.approx(0.0)
    assert intra_list_diversity(np.array([[0, 2]]), vec) == pytest.approx(1.0)
    pop = np.array([99.0, 0.0, 0.0, 0.0])
    # the unpopular item is more novel
    assert novelty(np.array([[1]]), pop, 100) > novelty(np.array([[0]]), pop, 100)
