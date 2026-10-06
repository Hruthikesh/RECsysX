"""Loading the trained models that later stages (retrieval, ranking, pipeline) build on."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..config import project_path
from ..data.dataset import RecData
from ..models import build_recommender
from .suites import selected_model_cfg


def checkpoint_dir(cfg: dict, key: str, seed: int = 42, group: str = "main") -> Path:
    return project_path(cfg["paths"]["checkpoint_dir"]) / group / f"{key}_s{seed}"


def load_main_model(cfg: dict, data: RecData, key: str, seed: int = 42, device: torch.device | str = "cpu"):
    """Rebuild a model from the main-suite checkpoint (popularity models are refitted, they are just counts)."""
    model = build_recommender(selected_model_cfg(key, cfg), torch.device(device))
    if key.startswith("popularity"):
        return model.fit(data)
    path = checkpoint_dir(cfg, key, seed)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing, run `python scripts/run_experiments.py main` first")
    return model.load(path, data)


def load_topk(cfg: dict, key: str, seed: int = 42, role: str = "test", group: str = "main") -> tuple[np.ndarray, np.ndarray]:
    z = np.load(checkpoint_dir(cfg, key, seed, group) / "topk.npz")
    return z[f"{role}_users"], z[role]
