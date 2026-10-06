"""Two-tower retrieval model.

user tower: [user id embedding, demographics] -> MLP -> L2-normalised 64-d vector
item tower: [item id embedding, genres/year/title features] -> MLP -> L2-normalised vector

Trained with an in-batch sampled softmax: the other items in the batch are the
negatives. Popular items show up in batches more often, so without correction the model
learns to push them down too hard. The logQ correction (Yi et al., 2019) subtracts
log(sampling probability) from each logit to undo that.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..data.dataset import RecData
from .layers import id_dropout
from .trainer import TorchEmbeddingRecommender


def mlp(d_in: int, hidden: int, d_out: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, d_out))


class TwoTowerModule(nn.Module):
    def __init__(self, n_users: int, n_items: int, user_content: torch.Tensor, item_content: torch.Tensor,
                 known_users: torch.Tensor, known_items: torch.Tensor, item_log_q: torch.Tensor,
                 id_dim: int = 64, hidden: int = 256, out_dim: int = 64, dropout: float = 0.1,
                 temperature: float = 0.1, user_id_dropout: float = 0.1, item_id_dropout: float = 0.1,
                 use_log_q: bool = True):
        super().__init__()
        self.n_users, self.n_items = n_users, n_items
        self.user_id = nn.Embedding(n_users + 1, id_dim)
        self.item_id = nn.Embedding(n_items + 1, id_dim)
        nn.init.normal_(self.user_id.weight, std=0.1)
        nn.init.normal_(self.item_id.weight, std=0.1)
        self.register_buffer("user_content", user_content)
        self.register_buffer("item_content", item_content)
        self.register_buffer("known_users", known_users)
        self.register_buffer("known_items", known_items)
        self.register_buffer("item_log_q", item_log_q)
        self.user_mlp = mlp(id_dim + user_content.shape[1], hidden, out_dim, dropout)
        self.item_mlp = mlp(id_dim + item_content.shape[1], hidden, out_dim, dropout)
        self.temperature = temperature
        self.p_user, self.p_item = user_id_dropout, item_id_dropout
        self.use_log_q = use_log_q

    def user_tower(self, users: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.user_id(ids), self.user_content[users]], dim=1)
        return F.normalize(self.user_mlp(x), dim=1)

    def item_tower(self, items: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.item_id(ids), self.item_content[items]], dim=1)
        return F.normalize(self.item_mlp(x), dim=1)

    def batch_loss(self, edge_ids, users, items, sampler):
        eu = self.user_tower(users, id_dropout(users, self.p_user, self.n_users, self.training))
        ei = self.item_tower(items, id_dropout(items, self.p_item, self.n_items, self.training))
        logits = eu @ ei.T / self.temperature
        if self.use_log_q:
            logits = logits - self.item_log_q[items][None, :]
        # if an item appears twice in the batch its other copy is not a real negative
        dup = items[None, :] == items[:, None]
        dup.fill_diagonal_(False)
        logits = logits.masked_fill(dup, float("-inf"))
        return F.cross_entropy(logits, torch.arange(len(users), device=users.device))

    def _ids(self, n: int, known: torch.Tensor, unk: int, use_unk: bool) -> torch.Tensor:
        ids = torch.arange(n, device=known.device)
        return torch.where(known, ids, torch.full_like(ids, unk)) if use_unk else ids

    def compute_embeddings(self):
        users = torch.arange(self.n_users, device=self.known_users.device)
        items = torch.arange(self.n_items, device=self.known_items.device)
        U = self.user_tower(users, self._ids(self.n_users, self.known_users, self.n_users, self.p_user > 0))
        I = self.item_tower(items, self._ids(self.n_items, self.known_items, self.n_items, self.p_item > 0))
        return U, I


class TwoTowerRecommender(TorchEmbeddingRecommender):
    name = "two_tower"

    def build_module(self, data: RecData) -> nn.Module:
        c = self.cfg
        pop = np.bincount(data.train_pos[:, 1], minlength=data.n_items).astype(np.float64)
        log_q = np.log((pop + 1.0) / (pop.sum() + data.n_items))
        return TwoTowerModule(
            data.n_users, data.n_items,
            torch.as_tensor(data.user_content, dtype=torch.float32),
            torch.as_tensor(data.item_content, dtype=torch.float32),
            torch.as_tensor(data.known_users()), torch.as_tensor(data.known_items()),
            torch.as_tensor(log_q, dtype=torch.float32),
            id_dim=int(c.get("id_dim", 64)), hidden=int(c.get("hidden_dim", 256)), out_dim=int(c["embedding_dim"]),
            dropout=float(c.get("dropout", 0.1)), temperature=float(c.get("temperature", 0.1)),
            user_id_dropout=float(c.get("user_id_dropout", 0.1)), item_id_dropout=float(c.get("item_id_dropout", 0.1)),
            use_log_q=bool(c.get("log_q_correction", True)),
        )
