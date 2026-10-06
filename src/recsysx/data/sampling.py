"""Negative sampling for implicit feedback.

Negatives are drawn uniformly from the whole catalog and re-drawn when they hit one of
the user's training positives. Items the user rated low (< 4) are allowed as negatives:
for this dataset a low rating is real evidence that the user did not like the item.
"""
from __future__ import annotations

import numpy as np
import torch


class NegativeSampler:
    def __init__(self, train_pos: np.ndarray, n_items: int, device: torch.device | str = "cpu",
                 max_tries: int = 20):
        self.n_items = n_items
        self.max_tries = max_tries
        self.device = torch.device(device)
        keys = np.unique(train_pos[:, 0].astype(np.int64) * n_items + train_pos[:, 1].astype(np.int64))
        self.pos_keys = torch.as_tensor(keys, device=self.device)

    def is_positive(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        keys = users.to(torch.int64) * self.n_items + items.to(torch.int64)
        idx = torch.searchsorted(self.pos_keys, keys).clamp(max=len(self.pos_keys) - 1)
        return self.pos_keys[idx] == keys

    def sample(self, users: torch.Tensor, num: int = 1, generator: torch.Generator | None = None) -> torch.Tensor:
        """Return a (len(users), num) tensor of negative item ids."""
        users = users.to(self.device)
        rep = users.repeat_interleave(num)
        neg = torch.randint(0, self.n_items, (len(rep),), device=self.device, generator=generator)
        for _ in range(self.max_tries):
            bad = self.is_positive(rep, neg)
            n_bad = int(bad.sum())
            if n_bad == 0:
                break
            neg[bad] = torch.randint(0, self.n_items, (n_bad,), device=self.device, generator=generator)
        bad = self.is_positive(rep, neg)
        if bad.any():
            # only happens for users who liked most of the catalog: sample exactly from
            # the items they did not like
            for j in torch.nonzero(bad).flatten().tolist():
                u = int(rep[j])
                lo = torch.searchsorted(self.pos_keys, torch.tensor(u * self.n_items, device=self.device))
                hi = torch.searchsorted(self.pos_keys, torch.tensor((u + 1) * self.n_items, device=self.device))
                liked = (self.pos_keys[lo:hi] - u * self.n_items).cpu().numpy()
                allowed = np.setdiff1d(np.arange(self.n_items), liked)
                if len(allowed):
                    neg[j] = int(np.random.choice(allowed))
        return neg.view(len(users), num)
