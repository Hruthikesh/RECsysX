"""Train GNN variants for a fixed number of epochs without early stopping and print the
validation curve. This is how the GraphSAGE early-stopping / overfitting problem was found
(results/logs/gnn_training_diagnostics.txt).

    python scripts/debug_gnn_training.py --epochs 40 default "{lr: 0.005}" "{conv: lgc}"

Each positional argument is a YAML dict of overrides on top of configs/models/graphsage.yaml.
"""
import argparse
import json
import logging
import time

import torch
import yaml

from recsysx.config import load_config
from recsysx.data import RecData
from recsysx.evaluation import Evaluator
from recsysx.models import build_recommender
from recsysx.utils import get_device, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variants", nargs="+")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--base", default="graphsage")
    ap.add_argument("--every", type=int, default=4, help="print every n-th epoch of the curve")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config()
    data = RecData.load(cfg)
    device = get_device(cfg["device"])
    ev = Evaluator(data, "val", device=device)
    logging.getLogger("recsysx").setLevel(logging.WARNING)
    for spec in args.variants:
        over = {} if spec == "default" else yaml.safe_load(spec)
        mcfg = load_config(args.base)["model_cfg"]
        mcfg.update({"epochs": args.epochs, "patience": 10_000, "seed": args.seed, **over})
        set_seed(args.seed)
        t = time.time()
        model = build_recommender(mcfg, device).fit(data, ev)
        hist = model.fit_info["history"]
        curve = [round(h["val/recall@20"], 4) for h in hist][args.every - 1::args.every]
        groups = {g: [round(h.get(f"val/{g}/recall@20", float("nan")), 3) for h in hist][args.every - 1::args.every]
                  for g in ("warm", "cold")}
        print(json.dumps(over), f"{time.time() - t:.0f}s best val recall@20 {model.fit_info['best_val']:.4f} "
              f"at epoch {model.fit_info['best_epoch']}, final train loss {hist[-1]['loss']:.3f}", curve, flush=True)
        print("    warm", groups["warm"], flush=True)
        print("    cold", groups["cold"], flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
