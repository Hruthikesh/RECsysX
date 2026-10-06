"""Create the schema and load the processed MovieLens tables into PostgreSQL.

    set DATABASE_URL=postgresql://user:pass@localhost:5432/recsysx   (or export on Linux/macOS)
    python scripts/db_load.py

Without DATABASE_URL a local server is started with `pgserver` if it is installed.
"""
import argparse
import time

from recsysx.config import load_config, project_path
from recsysx.data.db import connect, create_schema, load_processed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--database-url", default=None)
    ap.add_argument("--drop", action="store_true", help="drop and recreate all tables first")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides")
    args = ap.parse_args()
    cfg = load_config(overrides=args.set)
    with connect(args.database_url) as conn:
        create_schema(conn, drop=args.drop)
        t = time.perf_counter()
        counts = load_processed(conn, project_path(cfg["paths"]["processed_dir"]))
        print(f"loaded {counts} in {time.perf_counter() - t:.1f}s")


if __name__ == "__main__":
    main()
