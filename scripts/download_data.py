"""Download MovieLens-1M from GroupLens into data/raw/ and verify the checksum.

    python scripts/download_data.py
"""
import argparse

from recsysx.config import load_config, project_path
from recsysx.data.movielens import download_ml1m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="download again even if the zip exists")
    args = ap.parse_args()
    cfg = load_config()
    folder = download_ml1m(project_path(cfg["paths"]["raw_dir"]), force=args.force)
    print(f"MovieLens-1M ready in {folder}")


if __name__ == "__main__":
    main()
