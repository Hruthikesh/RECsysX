"""Build data/processed/ from the raw files: id mapping, cleaning, temporal split,
val/test users, user/item groups and content feature matrices.

    python scripts/prepare_data.py
"""
import argparse
import json

from recsysx.config import load_config
from recsysx.data.preprocess import prepare_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. split.cutoff_quantile=0.85")
    args = ap.parse_args()
    cfg = load_config(overrides=args.set)
    meta = prepare_dataset(cfg)
    summary = {k: v for k, v in meta.items() if not k.endswith("_features")}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
