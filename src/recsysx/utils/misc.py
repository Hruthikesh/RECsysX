from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def count_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


_LOGGERS: dict[str, logging.Logger] = {}


def get_logger(name: str = "recsysx") -> logging.Logger:
    if name in _LOGGERS:
        return _LOGGERS[name]
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    _LOGGERS[name] = logger
    return logger


class _NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, cls=_NumpyEncoder)


def load_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class Timer:
    """Context manager that records wall-clock seconds; syncs CUDA so GPU work is counted."""

    def __init__(self, sync_cuda: bool = True):
        self.sync_cuda = sync_cuda
        self.elapsed = 0.0

    def __enter__(self):
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self._start
        return False
