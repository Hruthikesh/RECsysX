"""Features for the second-stage ranker.

Leakage rule: every statistic is computed from the training period (ratings before the
cutoff). The labels come from after the cutoff. The model-score features come from
retrieval models that were also trained only on pre-cutoff positives.

Feature groups
  user   activity, rating behaviour, recency, genre breadth, mainstreamness, demographics
  item   popularity (all-time and last 30 days), average rating, release year, cold flag
  pair   retrieval score/rank, scores of the other models, genre affinity, the user's
         average rating for the item's genres, popularity gap, release-year gap
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import torch

from ..data.dataset import RecData

DAY = 86400
PairScorer = Callable[[np.ndarray, np.ndarray], np.ndarray]   # (users (n,), cand (n, N)) -> scores (n, N)


def pair_scorer(model, device: torch.device | str = "cpu") -> PairScorer:
    """Score only the candidate pairs. Embedding models: dot product of cached vectors,
    ItemKNN: the user's history row times the similarity columns of the candidates."""
    dev = torch.device(device)
    if getattr(model, "is_embedding_model", False):
        U = torch.as_tensor(model.user_vectors(), device=dev)
        I = torch.as_tensor(model.item_vectors(), device=dev)

        def fn(users, cand):
            u = U[torch.as_tensor(users, device=dev)]
            it = I[torch.as_tensor(cand, device=dev)]
            return torch.einsum("nd,nkd->nk", u, it).cpu().numpy()
        return fn
    if hasattr(model, "sim"):
        sim = model.sim.to(dev)

        def fn(users, cand):
            hist = torch.as_tensor(model.R[users].toarray(), device=dev)
            full = hist @ sim                                    # (n, n_items)
            return full.gather(1, torch.as_tensor(cand, device=dev)).cpu().numpy()
        return fn

    def fn(users, cand):
        with torch.no_grad():
            full = model.score(torch.as_tensor(users, device=model.device)).float()
        return full.gather(1, torch.as_tensor(cand, device=full.device)).cpu().numpy()
    return fn


def _entropy(rows: np.ndarray) -> np.ndarray:
    p = rows / rows.sum(axis=1, keepdims=True).clip(1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -(np.where(p > 0, p * np.log(p), 0.0)).sum(axis=1)
    return h / np.log(rows.shape[1])


class RankingFeatureBuilder:
    def __init__(self, data: RecData, pair_scorers: dict[str, PairScorer] | None = None, recent_days: int = 30):
        self.data = data
        self.pair_scorers = pair_scorers or {}
        thr = data.meta["positive_threshold"]
        tr = data.train_ratings
        nU, nI = data.n_users, data.n_items
        genres = data.genre_matrix().astype(np.float64)
        self.genres = genres
        cutoff = data.cutoff

        # ---- user statistics
        g = tr.groupby("user_idx")
        u_n = np.zeros(nU)
        u_mean = np.full(nU, np.nan)
        u_std = np.zeros(nU)
        u_first = np.full(nU, np.nan)
        u_last = np.full(nU, np.nan)
        agg = g.agg(n=("rating", "size"), mean=("rating", "mean"), std=("rating", "std"),
                    first=("timestamp", "min"), last=("timestamp", "max"))
        idx = agg.index.to_numpy()
        u_n[idx] = agg["n"].to_numpy()
        u_mean[idx] = agg["mean"].to_numpy()
        u_std[idx] = agg["std"].fillna(0).to_numpy()
        u_first[idx] = agg["first"].to_numpy()
        u_last[idx] = agg["last"].to_numpy()
        global_mean = float(tr["rating"].mean())

        pos = tr[tr["rating"] >= thr]
        u_npos = np.bincount(pos["user_idx"], minlength=nU).astype(np.float64)
        i_pop = np.bincount(pos["item_idx"], minlength=nI).astype(np.float64)
        logpop = np.log1p(i_pop)
        # genre profile of what the user liked
        prof = np.zeros((nU, genres.shape[1]))
        np.add.at(prof, pos["user_idx"].to_numpy(), genres[pos["item_idx"].to_numpy()])
        self.user_genre_profile = prof
        # average rating the user gives per genre (all ratings, not only positives)
        gsum = np.zeros((nU, genres.shape[1]))
        gcnt = np.zeros((nU, genres.shape[1]))
        gi = genres[tr["item_idx"].to_numpy()]
        np.add.at(gsum, tr["user_idx"].to_numpy(), gi * tr["rating"].to_numpy()[:, None])
        np.add.at(gcnt, tr["user_idx"].to_numpy(), gi)
        user_mean_filled = np.where(np.isnan(u_mean), global_mean, u_mean)
        # shrink towards the user's mean when the user rated few movies of a genre
        self.user_genre_rating = (gsum + 3 * user_mean_filled[:, None]) / (gcnt + 3)

        mean_logpop = np.zeros(nU)
        np.add.at(mean_logpop, pos["user_idx"].to_numpy(), logpop[pos["item_idx"].to_numpy()])
        mean_logpop = np.where(u_npos > 0, mean_logpop / u_npos.clip(1), np.nan)
        years = data.items["year"].to_numpy(dtype=np.float64)
        mean_year = np.zeros(nU)
        np.add.at(mean_year, pos["user_idx"].to_numpy(), years[pos["item_idx"].to_numpy()])
        mean_year = np.where(u_npos > 0, mean_year / u_npos.clip(1), np.nan)

        users = data.users
        cold_user = u_npos == 0
        self.user_feats = pd.DataFrame({
            "u_n_ratings_log": np.log1p(u_n),
            "u_n_pos_log": np.log1p(u_npos),
            "u_mean_rating": user_mean_filled,
            "u_rating_std": u_std,
            # cold users have no last rating, 1000 days is longer than the whole dataset
            "u_days_since_last": np.where(np.isnan(u_last), 1000.0, (cutoff - u_last) / DAY),
            "u_active_days": np.where(np.isnan(u_last), 0.0, (u_last - u_first) / DAY),
            "u_genre_entropy": np.where(cold_user, 0.0, _entropy(prof + 1e-12)),
            "u_mean_logpop": np.where(np.isnan(mean_logpop), np.nanmean(mean_logpop), mean_logpop),
            "u_is_cold": cold_user.astype(float),
            "u_gender_m": (users["gender"].to_numpy() == "M").astype(float),
            "u_age": users["age"].to_numpy(dtype=float),
        })
        self._user_mean_year = np.where(np.isnan(mean_year), np.nanmean(mean_year), mean_year)

        # ---- item statistics
        i_n = np.bincount(tr["item_idx"], minlength=nI).astype(np.float64)
        i_sum = np.bincount(tr["item_idx"], weights=tr["rating"].to_numpy(dtype=float), minlength=nI)
        recent = pos[pos["timestamp"] >= cutoff - recent_days * DAY]
        i_recent = np.bincount(recent["item_idx"], minlength=nI).astype(np.float64)
        self.item_feats = pd.DataFrame({
            "i_logpop": logpop,
            "i_n_ratings_log": np.log1p(i_n),
            # Bayesian average with 10 pseudo-ratings, otherwise one 5-star rating = best movie
            "i_mean_rating": (i_sum + 10 * global_mean) / (i_n + 10),
            "i_recent_logpop": np.log1p(i_recent),
            "i_year": years,
            "i_is_cold": (i_pop == 0).astype(float),
            "i_n_genres": genres.sum(axis=1),
        })
        self._years = years

    # ------------------------------------------------------------------
    def model_scores(self, users: np.ndarray, cand: np.ndarray) -> dict[str, np.ndarray]:
        c = np.maximum(cand, 0)
        return {name: np.asarray(fn(users, c), dtype=np.float32) for name, fn in self.pair_scorers.items()}

    def build(self, users: np.ndarray, cand: np.ndarray, cand_scores: np.ndarray) -> pd.DataFrame:
        """One row per (user, candidate). cand/cand_scores: (n_users, N) from the retriever."""
        if not hasattr(self, "_U"):
            # numpy copies of the stat tables: indexing DataFrames row by row cost ~10 ms per
            # request in the online path, building one frame at the end is much cheaper
            self._U, self._I = self.user_feats.to_numpy(), self.item_feats.to_numpy()
        n, N = cand.shape
        uu = np.repeat(users, N)
        ii = np.maximum(cand.ravel(), 0)
        cols = {"user": uu, "item": ii}
        U, I = self._U[uu], self._I[ii]
        cols.update({c: U[:, j] for j, c in enumerate(self.user_feats.columns)})
        cols.update({c: I[:, j] for j, c in enumerate(self.item_feats.columns)})
        cols["retrieval_score"] = cand_scores.ravel()
        cols["retrieval_rank"] = np.tile(np.arange(N), n)
        for name, sc in self.model_scores(users, cand).items():
            cols[f"score_{name}"] = sc.ravel()
        prof = self.user_genre_profile[uu]
        g = self.genres[ii]
        denom = np.linalg.norm(prof, axis=1) * np.linalg.norm(g, axis=1)
        cols["genre_affinity"] = np.where(denom > 0, (prof * g).sum(1) / denom.clip(1e-12), 0.0)
        cols["genre_rating"] = (self.user_genre_rating[uu] * g).sum(1) / g.sum(1).clip(1)
        cols["pop_gap"] = cols["i_logpop"] - cols["u_mean_logpop"]
        cols["year_gap"] = self._years[ii] - self._user_mean_year[uu]
        cols["valid"] = cand.ravel() >= 0
        return pd.DataFrame(cols)

    @staticmethod
    def feature_columns(df: pd.DataFrame) -> list[str]:
        return [c for c in df.columns if c not in ("user", "item", "label", "valid")]


def add_labels(df: pd.DataFrame, data: RecData, targets=None) -> pd.DataFrame:
    t = data.targets if targets is None else targets
    keys = set((t.tocoo().row.astype(np.int64) * data.n_items + t.tocoo().col).tolist())
    k = df["user"].to_numpy(np.int64) * data.n_items + df["item"].to_numpy(np.int64)
    df["label"] = np.fromiter((x in keys for x in k), dtype=np.int8, count=len(k))
    return df
