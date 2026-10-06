"""Item-based nearest neighbours on the training positives.

score(u, i) = sum over items j the user liked of sim(i, j). In graph terms this counts
(weighted) user -> item -> user -> item paths of length 3, so it is also a simple
"graph" method without any learning. Known to be a hard baseline on MovieLens.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from ..data.dataset import RecData
from .base import Recommender


class ItemKNN(Recommender):
    name = "itemknn"

    def fit(self, data: RecData, evaluator=None, tracker=None):
        R = data.train_matrix()                       # users x items, binary
        k = int(self.cfg.get("top_k_neighbors", 200))
        shrink = float(self.cfg.get("shrink", 10.0))
        co = (R.T @ R).toarray().astype(np.float32)   # 3.7k x 3.7k fits in memory
        norms = np.sqrt(np.diag(co))
        sim = co / (np.outer(norms, norms) + shrink + 1e-12)
        np.fill_diagonal(sim, 0.0)
        if k < sim.shape[0]:
            # keep the k strongest neighbours of each item (column-wise)
            thresh = -np.partition(-sim, k - 1, axis=0)[k - 1]
            sim[sim < thresh[None, :]] = 0.0
        self.sim = torch.as_tensor(sim, device=self.device)
        self.R = R
        pop = data.item_popularity()
        # users without history get all-zero scores, the tiny popularity term turns that
        # into the popularity ranking instead of an arbitrary order
        self.tie = torch.as_tensor(pop / pop.max() * 1e-6, dtype=torch.float32, device=self.device)
        return self

    def score(self, users: torch.Tensor) -> torch.Tensor:
        rows = users.cpu().numpy()
        sub = self.R[rows]
        dense = torch.as_tensor(sub.toarray(), device=self.device)
        return dense @ self.sim + self.tie

    def score_with_history(self, history_rows: sp.csr_matrix) -> torch.Tensor:
        dense = torch.as_tensor(history_rows.toarray(), dtype=torch.float32, device=self.device)
        return dense @ self.sim + self.tie

    def num_parameters(self) -> int:
        return 0

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "sim.npy", self.sim.cpu().numpy())

    def load(self, path: Path, data: RecData):
        self.sim = torch.as_tensor(np.load(Path(path) / "sim.npy"), device=self.device)
        self.R = data.train_matrix()
        pop = data.item_popularity()
        self.tie = torch.as_tensor(pop / pop.max() * 1e-6, dtype=torch.float32, device=self.device)
        return self
