"""Matrix factorisation trained with BPR on implicit feedback."""
from __future__ import annotations

import torch
from torch import nn

from ..data.dataset import RecData
from .layers import bpr_loss, id_dropout, l2_batch
from .trainer import TorchEmbeddingRecommender


class MFModule(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int, known_users: torch.Tensor, known_items: torch.Tensor,
                 l2_reg: float = 1e-4, num_negatives: int = 1, user_id_dropout: float = 0.0,
                 item_id_dropout: float = 0.0):
        super().__init__()
        self.n_users, self.n_items = n_users, n_items
        # last row of each table is the unknown-id embedding
        self.user = nn.Embedding(n_users + 1, dim)
        self.item = nn.Embedding(n_items + 1, dim)
        nn.init.normal_(self.user.weight, std=0.1)
        nn.init.normal_(self.item.weight, std=0.1)
        self.register_buffer("known_users", known_users)
        self.register_buffer("known_items", known_items)
        self.l2_reg, self.num_negatives = l2_reg, num_negatives
        self.p_user, self.p_item = user_id_dropout, item_id_dropout

    def batch_loss(self, edge_ids, users, items, sampler):
        neg = sampler.sample(users, self.num_negatives)
        u = self.user(id_dropout(users, self.p_user, self.n_users, self.training))
        p = self.item(id_dropout(items, self.p_item, self.n_items, self.training))
        n = self.item(id_dropout(neg, self.p_item, self.n_items, self.training))
        return bpr_loss(u, p, n) + self.l2_reg * l2_batch(u, p, n)

    def compute_embeddings(self):
        U = self.user.weight[:-1].clone()
        I = self.item.weight[:-1].clone()
        if self.p_user > 0:
            U[~self.known_users] = self.user.weight[-1]
        if self.p_item > 0:
            I[~self.known_items] = self.item.weight[-1]
        return U, I


class MFRecommender(TorchEmbeddingRecommender):
    name = "mf"

    def build_module(self, data: RecData) -> nn.Module:
        c = self.cfg
        return MFModule(
            data.n_users, data.n_items, int(c["embedding_dim"]),
            torch.as_tensor(data.known_users()), torch.as_tensor(data.known_items()),
            l2_reg=float(c.get("l2_reg", 1e-4)), num_negatives=int(c.get("num_negatives", 1)),
            user_id_dropout=float(c.get("user_id_dropout", 0.0)), item_id_dropout=float(c.get("item_id_dropout", 0.0)),
        )
