from __future__ import annotations

import torch

from .gnn import GNNRecommender
from .itemknn import ItemKNN
from .mf import MFRecommender
from .node2vec import Node2VecRecommender
from .popularity import Popularity
from .two_tower import TwoTowerRecommender

MODELS = {
    "popularity": Popularity,
    "itemknn": ItemKNN,
    "mf": MFRecommender,
    "two_tower": TwoTowerRecommender,
    "node2vec": Node2VecRecommender,
    "gnn": GNNRecommender,
}


def build_recommender(model_cfg: dict, device: torch.device):
    name = model_cfg["model"]
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}, choose from {sorted(MODELS)}")
    return MODELS[name](model_cfg, device)
