"""Evaluation split.

ML-1M users rate most of their movies in one burst right after signing up, so a plain
train/val/test split on time ends up with a validation window full of brand-new users
and a test window (2001-2003) of long-time returning users. Model selection on one
population and testing on another made the numbers hard to interpret.

What is used instead:
  * one global time cutoff T (80% quantile of all rating timestamps)
  * everything before T is training data, nothing after T is ever trained on
  * users with positives after T are split into disjoint validation and test users,
    stratified by their cold/sparse/warm group so both sets have the same mix.
"""
from __future__ import annotations

import numpy as np


def temporal_cutoff(timestamps: np.ndarray, quantile: float) -> int:
    ts = np.sort(np.asarray(timestamps))
    return int(ts[int(quantile * len(ts))])


def assign_groups(train_counts: np.ndarray, sparse_max: int) -> np.ndarray:
    counts = np.asarray(train_counts)
    return np.where(counts == 0, "cold", np.where(counts <= sparse_max, "sparse", "warm"))


def split_eval_users(eligible: np.ndarray, groups: np.ndarray, val_fraction: float, seed: int) -> dict:
    """Stratified random split of the eligible users into val and test."""
    rng = np.random.default_rng(seed)
    val, test = [], []
    eligible = np.asarray(eligible)
    for g in np.unique(groups[eligible]):
        members = eligible[groups[eligible] == g].copy()
        rng.shuffle(members)
        n_val = int(round(val_fraction * len(members)))
        val.append(members[:n_val])
        test.append(members[n_val:])
    return {"val": np.sort(np.concatenate(val)), "test": np.sort(np.concatenate(test))}
