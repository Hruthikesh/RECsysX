from __future__ import annotations

import torch
import torch.nn.functional as F


def id_dropout(ids: torch.Tensor, p: float, unk_index: int, training: bool) -> torch.Tensor:
    """Randomly replace ids with the unknown-id row.

    This is what trains the "unknown" embedding. At inference every user/item that never
    appeared in training is mapped to that row instead of an untrained random vector.
    """
    if not training or p <= 0:
        return ids
    drop = torch.rand(ids.shape, device=ids.device) < p
    return torch.where(drop, torch.full_like(ids, unk_index), ids)


def bpr_loss(u: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
    """u, pos: (B, d); neg: (B, n_neg, d). Mean of -log sigmoid(s_pos - s_neg)."""
    s_pos = (u * pos).sum(-1, keepdim=True)
    s_neg = torch.einsum("bd,bnd->bn", u, neg)
    return F.softplus(s_neg - s_pos).mean()


def l2_batch(*tensors: torch.Tensor) -> torch.Tensor:
    batch = tensors[0].shape[0]
    return sum(t.pow(2).sum() for t in tensors) / (2 * batch)
