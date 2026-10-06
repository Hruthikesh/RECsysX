"""The integrated recommender: FAISS retrieval -> ranking features -> ranker -> MMR.

Which retriever, ranker, candidate count and MMR setting are used is not hard-coded:
they are read from the files the experiments wrote (ranking_meta.json from
run_ranking.py, diversity_selected.json from run_reranking.py), so the pipeline is
whatever the validation results picked.

    pipe = RecommendationPipeline.from_artifacts(cfg, data)
    recs = pipe.recommend(user_idx=42, k=10)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import joblib
import numpy as np
import scipy.sparse as sp

from .config import project_path
from .data.dataset import RecData, to_csr
from .experiments.artifacts import load_main_model
from .features.ranking import RankingFeatureBuilder, pair_scorer
from .ranking.rerank import mmr_rerank
from .retrieval.faiss_index import FaissRetriever
from .utils import load_json


@dataclass
class Recommendation:
    user_idx: int
    items: list[int]
    scores: list[float]
    timings_ms: dict = field(default_factory=dict)


class RecommendationPipeline:
    def __init__(self, data: RecData, retriever, score_models: dict, ranker, feature_columns: list[str],
                 n_candidates: int, mmr_lambda: float = 1.0, mmr_vectors: np.ndarray | None = None, store=None):
        self.data = data
        self.retriever = retriever
        self.U = retriever.user_vectors()
        self.index = FaissRetriever(retriever.item_vectors(), "flat")
        self.features = RankingFeatureBuilder(data, {k: pair_scorer(m, "cpu") for k, m in score_models.items()})
        self.ranker = ranker
        self.cols = feature_columns
        self.n_candidates = n_candidates
        self.mmr_lambda = mmr_lambda
        self.mmr_vectors = mmr_vectors if mmr_vectors is not None else data.genre_matrix()
        self.store = store   # optional PostgresStore: history is then read from the database

    @classmethod
    def from_artifacts(cls, cfg: dict, data: RecData, seed: int = 42, diversify: bool = True, store=None,
                       ranker_variant: str = "temporal"):
        """ranker_variant: "temporal" = XGBoost trained only on data before the cutoff
        (scripts/run_ranking_temporal.py, what a deployed system can do), "validation" = the ranker
        selected in scripts/run_ranking.py, trained on validation users from the test period."""
        ck = project_path(cfg["paths"]["checkpoint_dir"]) / "ranking"
        tables = project_path(cfg["paths"]["results_dir"]) / "tables"
        meta = load_json(ck / "ranking_meta.json")
        models = {k: load_main_model(cfg, data, k, seed, "cpu") for k in meta["score_models"]}
        path = ck / ("ranker_xgb_temporal.joblib" if ranker_variant == "temporal" else f"ranker_{meta['ranker']}.joblib")
        if not path.exists():
            raise FileNotFoundError(f"{path} missing, run scripts/run_ranking.py and scripts/run_ranking_temporal.py")
        ranker = joblib.load(path)
        lam, vectors = 1.0, None
        sel_path = tables / "diversity_selected.json"
        if diversify and sel_path.exists():
            sel = load_json(sel_path)
            lam = sel["lambda"]
            if sel["similarity"] == "mf_embedding":
                vectors = models["mf"].item_vectors()
        pipe = cls(data, models[meta["retriever"]], models, ranker, meta["feature_columns"], meta["n_candidates"],
                   lam, vectors, store)
        pipe.ranker_variant = ranker_variant
        pipe.retriever_key = meta["retriever"]
        return pipe

    # ------------------------------------------------------------------
    def _history(self, users: np.ndarray) -> sp.csr_matrix:
        if self.store is None:
            return self.data.history
        rows, cols = [], []
        for u in users:
            items = self.store.user_profile(int(u))["history_items"]
            rows += [u] * len(items)
            cols += items
        return to_csr(np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64),
                      (self.data.n_users, self.data.n_items))

    def recommend_batch(self, users: np.ndarray, k: int = 10, diversify: bool = True
                        ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Returns (items (n, k), ranker scores of those items (n, k), per-user timings in ms)."""
        users = np.asarray(users)
        t0 = time.perf_counter()
        history = self._history(users)
        t1 = time.perf_counter()
        cand, sc = self.index.search(self.U[users], self.n_candidates, history, users)
        t2 = time.perf_counter()
        feats = self.features.build(users, cand, sc)
        t3 = time.perf_counter()
        s = self.ranker.predict(feats[self.cols].to_numpy(np.float32)).reshape(len(users), -1).astype(np.float64)
        s[cand < 0] = -np.inf
        order = np.argsort(-s, axis=1, kind="stable")
        ranked = np.take_along_axis(cand, order, 1)
        ranked_s = np.take_along_axis(s, order, 1)
        t4 = time.perf_counter()
        if diversify and self.mmr_lambda < 1.0:
            top = mmr_rerank(ranked, np.nan_to_num(ranked_s, neginf=-1e9), self.mmr_vectors, k, self.mmr_lambda)
        else:
            top = ranked[:, :k]
        t5 = time.perf_counter()
        # ranker score of every chosen item (MMR changes the order, not the scores)
        top_s = np.full(top.shape, -np.inf)
        for r in range(len(users)):
            lookup = dict(zip(ranked[r].tolist(), ranked_s[r].tolist()))
            top_s[r] = [lookup.get(int(i), -np.inf) for i in top[r]]
        n = len(users)
        timings = {"history_ms": 1000 * (t1 - t0) / n, "retrieval_ms": 1000 * (t2 - t1) / n,
                   "features_ms": 1000 * (t3 - t2) / n, "ranking_ms": 1000 * (t4 - t3) / n,
                   "rerank_ms": 1000 * (t5 - t4) / n, "total_ms": 1000 * (t5 - t0) / n}
        return top, top_s, timings

    def recommend(self, user_idx: int, k: int = 10, diversify: bool = True) -> Recommendation:
        top, top_s, timings = self.recommend_batch(np.array([user_idx]), k, diversify)
        keep = top[0] >= 0
        return Recommendation(int(user_idx), top[0][keep].tolist(), top_s[0][keep].tolist(), timings)
