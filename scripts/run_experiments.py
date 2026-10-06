"""Run one experiment suite (see src/recsysx/experiments/suites.py).

    python scripts/run_experiments.py tune
    python scripts/run_experiments.py main
    python scripts/run_experiments.py gnn_ablation
    python scripts/run_experiments.py hybrid
    python scripts/run_experiments.py sparsity
    python scripts/run_experiments.py item_cold
    python scripts/run_experiments.py onboarding     # needs the checkpoints from `main`

Options: --models mf gat  (subset), --seeds 42 (fewer seeds for a quick run), --wandb
"""
import argparse
import inspect

from recsysx.config import load_config
from recsysx.data import RecData
from recsysx.experiments.suites import SUITES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("suite", choices=sorted(SUITES))
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--fractions", nargs="*", type=float, default=None, help="sparsity suite only")
    ap.add_argument("--wandb", action="store_true", help="also log runs to Weights & Biases")
    ap.add_argument("--max-epochs", type=int, default=None, help="cap epochs of every model (quick/smoke runs)")
    ap.add_argument("--set", nargs="*", default=[], help="base config overrides, e.g. paths.results_dir=/tmp/r")
    args = ap.parse_args()

    cfg = load_config(overrides=args.set)
    if args.wandb:
        cfg["tracking"]["wandb"] = True
    if args.max_epochs:
        cfg["max_epochs"] = args.max_epochs
    data = RecData.load(cfg)
    fn = SUITES[args.suite]
    params = inspect.signature(fn).parameters
    kwargs = {}
    if args.models is not None and "models" in params:
        kwargs["models"] = args.models
    if args.seeds is not None:
        if "seeds" in params:
            kwargs["seeds"] = args.seeds
        elif "seed" in params:
            kwargs["seed"] = args.seeds[0]
    if args.fractions is not None and "fractions" in params:
        kwargs["fractions"] = args.fractions
    df = fn(cfg, data, **kwargs)
    print(f"{args.suite}: {len(df)} runs finished")


if __name__ == "__main__":
    main()
