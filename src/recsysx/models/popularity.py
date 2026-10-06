"""Non-personalised baselines."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..data.dataset import RecData
from .base import Recommender


class Popularity(Recommender):
    """Rank items by number of positive ratings in the training period.

    variant="global": all training positives
    variant="recent": only positives in the last ``recent_days`` before the cutoff
    variant="demographic": popularity inside the user's (gender, age group) segment,
        the simplest thing that gives a brand-new user something other than the global list
    """
    name = "popularity"

    def fit(self, data: RecData, evaluator=None, tracker=None) -> "Popularity":
        variant = self.cfg.get("variant", "global")
        self.variant = variant
        pos = data.train_pos
        n_items = data.n_items
        global_pop = np.bincount(pos[:, 1], minlength=n_items).astype(np.float64)
        # tiny global term breaks ties in the sparse variants with the overall popularity
        tie = global_pop / (global_pop.max() * 1e3)
        if variant == "global":
            table = global_pop[None, :]
            self.segment = np.zeros(data.n_users, dtype=np.int64)
        elif variant == "recent":
            start = data.cutoff - int(self.cfg.get("recent_days", 30)) * 86400
            tr = data.train_ratings
            keep = (tr["timestamp"] >= start) & (tr["rating"] >= data.meta["positive_threshold"])
            # a fraction-subsampled run has fewer positives than train_ratings, restrict to them
            pos_keys = set((pos[:, 0] * n_items + pos[:, 1]).tolist())
            keys = tr["user_idx"].to_numpy() * n_items + tr["item_idx"].to_numpy()
            keep &= np.fromiter((k in pos_keys for k in keys), dtype=bool, count=len(keys))
            recent = np.bincount(tr.loc[keep, "item_idx"], minlength=n_items).astype(np.float64)
            table = (recent + tie)[None, :]
            self.segment = np.zeros(data.n_users, dtype=np.int64)
        elif variant == "demographic":
            seg_key = data.users["gender"].astype(str) + "_" + data.users["age"].astype(str)
            codes, uniq = seg_key.factorize()
            self.segment = np.asarray(codes, dtype=np.int64)
            table = np.zeros((len(uniq), n_items))
            np.add.at(table, (self.segment[pos[:, 0]], pos[:, 1]), 1.0)
            table = table / table.sum(axis=1, keepdims=True).clip(1) + tie
        else:
            raise ValueError(f"unknown popularity variant {variant}")
        self.table = torch.as_tensor(table, dtype=torch.float32, device=self.device)
        self.segment_t = torch.as_tensor(self.segment, device=self.device)
        return self

    def score(self, users: torch.Tensor) -> torch.Tensor:
        return self.table[self.segment_t[users.to(self.device)]]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, table=self.table.cpu().numpy(), segment=self.segment, variant=self.variant)

    def load(self, path: Path, data: RecData) -> "Popularity":
        z = np.load(path, allow_pickle=True)
        self.variant = str(z["variant"])
        self.segment = z["segment"]
        self.table = torch.as_tensor(z["table"], device=self.device)
        self.segment_t = torch.as_tensor(self.segment, device=self.device)
        return self
