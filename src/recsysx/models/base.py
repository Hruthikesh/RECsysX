"""Common interface for all recommenders.

A recommender is fitted on a RecData object and then exposes ``score(users)`` which
returns a (len(users), n_items) tensor. Embedding-based models additionally expose
user/item vectors, which is what the FAISS retrieval stage and the ranking features use.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..data.dataset import RecData


class Recommender:
    name = "base"
    is_embedding_model = False

    def __init__(self, cfg: dict, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.train_log: list[dict] = []

    def fit(self, data: RecData, evaluator=None, tracker=None) -> "Recommender":
        raise NotImplementedError

    def score(self, users: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def num_parameters(self) -> int:
        return 0

    def save(self, path: Path) -> None:
        raise NotImplementedError

    def load(self, path: Path, data: RecData) -> "Recommender":
        raise NotImplementedError


class EmbeddingRecommender(Recommender):
    """score(u, i) = <user_vec[u], item_vec[i]> on top of cached final embeddings."""
    is_embedding_model = True

    def __init__(self, cfg: dict, device: torch.device):
        super().__init__(cfg, device)
        self.user_vec: torch.Tensor | None = None
        self.item_vec: torch.Tensor | None = None

    def set_embeddings(self, user_vec: torch.Tensor, item_vec: torch.Tensor) -> None:
        self.user_vec = user_vec.detach().float().to(self.device)
        self.item_vec = item_vec.detach().float().to(self.device)

    def score(self, users: torch.Tensor) -> torch.Tensor:
        return self.user_vec[users.to(self.device)] @ self.item_vec.T

    def user_vectors(self) -> np.ndarray:
        return self.user_vec.cpu().numpy()

    def item_vectors(self) -> np.ndarray:
        return self.item_vec.cpu().numpy()

    def save_embeddings(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, user=self.user_vectors(), item=self.item_vectors())

    def load_embeddings(self, path: Path) -> None:
        z = np.load(path)
        self.set_embeddings(torch.as_tensor(z["user"]), torch.as_tensor(z["item"]))
