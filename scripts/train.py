"""Train and evaluate a single model.

    python scripts/train.py --model mf
    python scripts/train.py --model graphsage --seed 1 --set num_layers=3 aggr=max
    python scripts/train.py --model gat --train-frac 0.2 --wandb

Metrics for validation and test users are printed and appended to
results/tables/single_runs.csv, the checkpoint goes to checkpoints/single/<run name>/.
"""
import argparse

from recsysx.config import apply_overrides, load_config
from recsysx.data import RecData
from recsysx.experiments import run_model, upsert_csv
from recsysx.experiments.suites import selected_model_cfg, table_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="config name in configs/models/ or a path to a yaml file")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--set", nargs="*", default=[], help="model config overrides key=value")
    ap.add_argument("--train-frac", type=float, default=1.0, help="use a random fraction of the training positives")
    ap.add_argument("--no-selected", action="store_true", help="ignore configs/selected.yaml (tuned values)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    if args.wandb:
        cfg["tracking"]["wandb"] = True
    data = RecData.load(cfg)
    if args.train_frac < 1:
        data = data.with_train_fraction(args.train_frac, seed=args.seed)
    mcfg = load_config(args.model)["model_cfg"] if args.no_selected else selected_model_cfg(args.model, cfg)
    mcfg = apply_overrides(mcfg, args.set)
    name = args.name or f"{args.model}_s{args.seed}" + (f"_frac{args.train_frac:g}" if args.train_frac < 1 else "")
    record, _, _ = run_model(cfg, mcfg, data, name, group="single", seed=args.seed, label=args.model)
    upsert_csv([record], table_path(cfg, "single_runs.csv"))
    keys = ["val/recall@20", "val/ndcg@20", "test/recall@10", "test/recall@20", "test/ndcg@10", "test/ndcg@20",
            "test/warm/recall@20", "test/sparse/recall@20", "test/cold/recall@20", "test/coverage@10",
            "train_time_s", "best_epoch", "n_params"]
    for k in keys:
        v = record.get(k)
        print(f"{k:28s} {v:.4f}" if isinstance(v, float) else f"{k:28s} {v}")


if __name__ == "__main__":
    main()
