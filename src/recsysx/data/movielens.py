"""Download and parse the raw MovieLens-1M files.

The dataset cannot be redistributed (GroupLens license), so it is fetched from the
official URL and checked against the published MD5.
"""
from __future__ import annotations

import hashlib
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
ML1M_MD5 = "c4d9eecfca2ab87c1945afe126590906"

GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary",
    "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western",
]

# codes from the ML-1M README
AGE_GROUPS = {1: "Under 18", 18: "18-24", 25: "25-34", 35: "35-44", 45: "45-49", 50: "50-55", 56: "56+"}
OCCUPATIONS = {
    0: "other", 1: "academic/educator", 2: "artist", 3: "clerical/admin", 4: "college/grad student",
    5: "customer service", 6: "doctor/health care", 7: "executive/managerial", 8: "farmer",
    9: "homemaker", 10: "K-12 student", 11: "lawyer", 12: "programmer", 13: "retired",
    14: "sales/marketing", 15: "scientist", 16: "self-employed", 17: "technician/engineer",
    18: "tradesman/craftsman", 19: "unemployed", 20: "writer",
}


def md5sum(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def download_ml1m(raw_dir: str | Path, force: bool = False) -> Path:
    """Download + extract ml-1m.zip into raw_dir. Returns the extracted folder."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "ml-1m.zip"
    if force or not zip_path.exists():
        print(f"downloading {ML1M_URL}")
        urllib.request.urlretrieve(ML1M_URL, zip_path)
    digest = md5sum(zip_path)
    if digest != ML1M_MD5:
        raise RuntimeError(f"MD5 mismatch for {zip_path}: {digest} (expected {ML1M_MD5})")
    out_dir = raw_dir / "ml-1m"
    if force or not (out_dir / "ratings.dat").exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(raw_dir)
    return out_dir


def _read_dat(path: Path, names: list[str]) -> pd.DataFrame:
    # movies.dat contains latin-1 characters (e.g. accented titles)
    return pd.read_csv(path, sep="::", engine="python", names=names, encoding="latin-1")


def load_raw(raw_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (ratings, movies, users) with the original MovieLens ids."""
    folder = Path(raw_dir) / "ml-1m"
    if not (folder / "ratings.dat").exists():
        raise FileNotFoundError(
            f"{folder} not found. Run `python scripts/download_data.py` first."
        )
    ratings = _read_dat(folder / "ratings.dat", ["user_id", "movie_id", "rating", "timestamp"])
    movies = _read_dat(folder / "movies.dat", ["movie_id", "title", "genres"])
    users = _read_dat(folder / "users.dat", ["user_id", "gender", "age", "occupation", "zip"])
    users["zip"] = users["zip"].astype(str)
    return ratings, movies, users
