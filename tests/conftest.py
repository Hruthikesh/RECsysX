"""A tiny fake MovieLens-1M (same file format) so the tests run the real preprocessing code."""
from __future__ import annotations

import numpy as np
import pytest

from recsysx.config import load_config
from recsysx.data import RecData
from recsysx.data.movielens import GENRES
from recsysx.data.preprocess import prepare_dataset


def write_fake_ml1m(folder, n_users=60, n_items=50, seed=0):
    rng = np.random.default_rng(seed)
    d = folder / "ml-1m"
    d.mkdir(parents=True)
    ages = [1, 18, 25, 35, 45, 50, 56]
    with open(d / "users.dat", "w", encoding="latin-1") as f:
        for u in range(1, n_users + 1):
            f.write(f"{u}::{'MF'[u % 2]}::{ages[u % 7]}::{u % 21}::{10000 + u}\n")
    words = ["night", "dark", "love", "story", "return", "star", "city", "lost", "king", "dead", "summer", "war"]
    with open(d / "movies.dat", "w", encoding="latin-1") as f:
        for i in range(1, n_items + 2):  # one extra movie nobody rates
            g = "|".join(sorted(set(rng.choice(GENRES, size=rng.integers(1, 3)))))
            title = " ".join(rng.choice(words, 2, replace=False)).title()
            f.write(f"{i}::{title}, The ({1950 + i % 50})::{g}\n")
    rows = []
    t0 = 970_000_000
    for u in range(1, n_users + 1):
        start = t0 + rng.integers(0, 3_000_000)
        items = rng.choice(np.arange(1, n_items + 1), size=rng.integers(20, 35), replace=False)
        for k, i in enumerate(items):
            rows.append((u, i, int(rng.integers(1, 6)), int(start + 60 * k)))
    with open(d / "ratings.dat", "w") as f:
        for r in rows:
            f.write("::".join(map(str, r)) + "\n")
    return d


@pytest.fixture(scope="session")
def tiny_cfg(tmp_path_factory):
    root = tmp_path_factory.mktemp("recsysx")
    write_fake_ml1m(root / "raw")
    cfg = load_config(overrides=[f"paths.raw_dir={(root / 'raw').as_posix()}",
                                 f"paths.processed_dir={(root / 'processed').as_posix()}",
                                 f"paths.results_dir={(root / 'results').as_posix()}",
                                 f"paths.checkpoint_dir={(root / 'ckpt').as_posix()}",
                                 "split.user_sparse_max=10", "device=cpu"])
    prepare_dataset(cfg)
    return cfg


@pytest.fixture(scope="session")
def tiny_data(tiny_cfg):
    return RecData.load(tiny_cfg)
