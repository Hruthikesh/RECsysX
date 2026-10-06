"""In-memory view of the processed data that every model and evaluator works with."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..config import project_path
from ..utils import load_json


def to_csr(rows: np.ndarray, cols: np.ndarray, shape: tuple[int, int]) -> sp.csr_matrix:
    m = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=shape)
    m.sum_duplicates()
    m.data[:] = 1.0
    return m


@dataclass
class RecData:
    n_users: int
    n_items: int
    interactions: pd.DataFrame       # every rating, with period train/eval
    users: pd.DataFrame
    items: pd.DataFrame
    item_content: np.ndarray
    user_content: np.ndarray
    meta: dict
    train_pos: np.ndarray            # (E, 2) user, item pairs the models learn from
    train_ratings: pd.DataFrame      # train-period ratings (any value) used for features
    history: sp.csr_matrix           # items each user already rated -> filtered at eval time
    targets: sp.csr_matrix           # eval-window positives
    role: np.ndarray                 # per user: val / test / none
    group: np.ndarray                # per user: cold / sparse / warm (fixed from the main split)
    item_group: np.ndarray           # per item: cold / sparse / warm
    tag: str = "main"
    notes: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, cfg: dict) -> "RecData":
        d = project_path(cfg["paths"]["processed_dir"])
        if not (d / "interactions.parquet").exists():
            raise FileNotFoundError(f"{d} has no processed data, run scripts/prepare_data.py first")
        inter = pd.read_parquet(d / "interactions.parquet")
        users = pd.read_parquet(d / "users.parquet")
        items = pd.read_parquet(d / "items.parquet")
        meta = load_json(d / "meta.json")
        n_users, n_items = len(users), len(items)

        train = inter[inter["period"] == "train"]
        pos = train[train["positive"]]
        ev = inter[(inter["period"] == "eval") & inter["positive"]]
        ev = ev[users["role"].to_numpy()[ev["user_idx"].to_numpy()] != "none"]

        return cls(
            n_users=n_users,
            n_items=n_items,
            interactions=inter,
            users=users,
            items=items,
            item_content=np.load(d / "item_content.npy"),
            user_content=np.load(d / "user_content.npy"),
            meta=meta,
            train_pos=pos[["user_idx", "item_idx"]].to_numpy(dtype=np.int64, copy=True),
            train_ratings=train[["user_idx", "item_idx", "rating", "timestamp"]].reset_index(drop=True),
            history=to_csr(train["user_idx"].to_numpy(), train["item_idx"].to_numpy(), (n_users, n_items)),
            targets=to_csr(ev["user_idx"].to_numpy(), ev["item_idx"].to_numpy(), (n_users, n_items)),
            role=users["role"].to_numpy(),
            group=users["group"].to_numpy(),
            item_group=items["group"].to_numpy(),
        )

    # ------------------------------------------------------------------ helpers
    @property
    def cutoff(self) -> int:
        return int(self.meta["cutoff_timestamp"])

    def eval_users(self, role: str) -> np.ndarray:
        users = np.where(self.role == role)[0]
        has_target = np.diff(self.targets.indptr)[users] > 0
        return users[has_target]

    def known_users(self) -> np.ndarray:
        return np.bincount(self.train_pos[:, 0], minlength=self.n_users) > 0

    def known_items(self) -> np.ndarray:
        return np.bincount(self.train_pos[:, 1], minlength=self.n_items) > 0

    def train_matrix(self) -> sp.csr_matrix:
        return to_csr(self.train_pos[:, 0], self.train_pos[:, 1], (self.n_users, self.n_items))

    def item_popularity(self) -> np.ndarray:
        return np.bincount(self.train_pos[:, 1], minlength=self.n_items).astype(np.float64)

    def genre_matrix(self) -> np.ndarray:
        return self.item_content[:, :18]

    def copy(self, **changes) -> "RecData":
        new = copy.copy(self)
        for k, v in changes.items():
            setattr(new, k, v)
        new.notes = dict(self.notes)
        return new

    # ------------------------------------------------------------------ protocol variants
    def with_train_fraction(self, fraction: float, seed: int) -> "RecData":
        """Keep a random fraction of the training positives.

        Evaluation users, targets and the seen-item filter are not touched, so results
        at different fractions are directly comparable.
        """
        if fraction >= 1.0:
            return self
        rng = np.random.default_rng(seed)
        keep = rng.random(len(self.train_pos)) < fraction
        new = self.copy(train_pos=self.train_pos[keep], tag=f"frac{fraction:g}")
        new.notes["train_fraction"] = fraction
        return new

    def with_cold_items(self, fraction: float, seed: int, min_train_pos: int = 1) -> "RecData":
        """Simulate new items: pick a random subset of items and delete everything
        the training period knows about them.

        The natural split only has ~60 items that first appear after the cutoff, which is
        too few to say anything about item cold-start.
        """
        rng = np.random.default_rng(seed)
        candidates = np.where(self.item_popularity() >= min_train_pos)[0]
        n_cold = int(round(fraction * self.n_items))
        cold = np.sort(rng.choice(candidates, size=n_cold, replace=False))
        is_cold = np.zeros(self.n_items, dtype=bool)
        is_cold[cold] = True

        keep_pos = ~is_cold[self.train_pos[:, 1]]
        tr = self.train_ratings
        tr = tr[~is_cold[tr["item_idx"].to_numpy()]].reset_index(drop=True)
        # a brand-new item cannot be in anyone's history either
        history = self.history.tolil(copy=True)
        history[:, cold] = 0
        history = history.tocsr()
        history.eliminate_zeros()
        item_group = self.item_group.copy()
        item_group[cold] = "cold"
        new = self.copy(train_pos=self.train_pos[keep_pos], train_ratings=tr, history=history,
                        item_group=item_group, tag=f"colditems{fraction:g}")
        new.notes["cold_items"] = cold
        return new

    def onboarding(self, role: str, k: int, max_k: int, min_eval_pos: int) -> tuple[np.ndarray, sp.csr_matrix, sp.csr_matrix, list[np.ndarray]]:
        """New-user scenario for users with no training history.

        For every cold user the eval-window ratings are put in time order. The first
        ``max_k`` positives are the "onboarding" ratings, the positives after them are the
        targets. For a given k only the first k onboarding positives are revealed to the
        model. Targets, the seen-item filter (everything up to the max_k-th positive) and
        the set of users are identical for every k, so the only thing that changes with k
        is how much the model knows about the user.

        Returns (users, history, targets, revealed_items_per_user).
        """
        if not 0 <= k <= max_k:
            raise ValueError("need 0 <= k <= max_k")
        users = np.where((self.role == role) & (self.group == "cold"))[0]
        ev = self.interactions[(self.interactions["period"] == "eval")
                               & self.interactions["user_idx"].isin(users)]
        ev = ev.sort_values(["user_idx", "timestamp"], kind="stable")
        out_users, revealed, seen_rows, seen_cols, tgt_rows, tgt_cols = [], [], [], [], [], []
        for u, g in ev.groupby("user_idx", sort=True):
            pos_flags = g["positive"].to_numpy()
            if pos_flags.sum() < max(min_eval_pos, max_k + 1):
                continue
            items = g["item_idx"].to_numpy()
            pos_items = items[pos_flags]
            cut = np.where(pos_flags)[0][max_k - 1] + 1 if max_k > 0 else 0
            seen = items[:cut]
            after = items[cut:][pos_flags[cut:]]
            out_users.append(u)
            revealed.append(pos_items[:k])
            seen_rows += [u] * len(seen)
            seen_cols += list(seen)
            tgt_rows += [u] * len(after)
            tgt_cols += list(after)
        shape = (self.n_users, self.n_items)
        history = to_csr(np.array(seen_rows, dtype=np.int64), np.array(seen_cols, dtype=np.int64), shape)
        targets = to_csr(np.array(tgt_rows, dtype=np.int64), np.array(tgt_cols, dtype=np.int64), shape)
        return np.array(out_users), history, targets, revealed
