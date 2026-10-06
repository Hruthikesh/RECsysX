"""Turn the raw MovieLens files into the processed tables used everywhere else.

Output (in data/processed/):
  interactions.parquet  user_idx, item_idx, rating, timestamp, positive, period
  users.parquet         demographics + role (val/test/none) + group (cold/sparse/warm)
  items.parquet         title, year, genres, training popularity, group, head flag
  item_content.npy      content vectors (genres, year, title tf-idf)
  user_content.npy      demographic vectors
  meta.json             cutoff timestamp, counts, preprocessing log
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import project_path
from ..utils import get_logger, save_json
from .movielens import GENRES, load_raw
from .split import assign_groups, split_eval_users, temporal_cutoff

log = get_logger()

YEAR_RE = re.compile(r"\((\d{4})\)\s*$")


def clean_items(movies: pd.DataFrame) -> pd.DataFrame:
    items = movies.copy()
    items["year"] = items["title"].str.extract(YEAR_RE, expand=False).astype(float)
    items["title_clean"] = items["title"].str.replace(YEAR_RE, "", regex=True).str.strip()
    # "Matrix, The" -> "The Matrix" so title tokens are consistent
    items["title_clean"] = items["title_clean"].str.replace(r"^(.*), (The|A|An)$", r"\2 \1", regex=True)
    items["genre_list"] = items["genres"].str.split("|")
    return items


def clean_users(users: pd.DataFrame) -> pd.DataFrame:
    users = users.copy()
    first = users["zip"].str[0]
    users["zip_region"] = np.where(first.str.isdigit(), first, "x")
    return users


def build_tables(raw_dir: Path, positive_threshold: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    ratings, movies, users = load_raw(raw_dir)
    report: dict = {
        "raw_ratings": len(ratings), "raw_movies": len(movies), "raw_users": len(users),
        "missing_values": {
            "ratings": int(ratings.isna().sum().sum()),
            "movies": int(movies.isna().sum().sum()),
            "users": int(users.isna().sum().sum()),
        },
    }

    dup_mask = ratings.duplicated(["user_id", "movie_id"], keep="last")
    report["duplicate_user_item_pairs"] = int(dup_mask.sum())
    ratings = ratings[~dup_mask]

    # The catalog is every movie with at least one rating. 177 movies in movies.dat were
    # never rated by anyone, they can never be a target so they are dropped.
    rated = np.sort(ratings["movie_id"].unique())
    report["movies_without_ratings"] = int(len(movies) - len(rated))
    movies = movies[movies["movie_id"].isin(rated)]

    user_ids = np.sort(users["user_id"].unique())
    user_map = pd.Series(np.arange(len(user_ids)), index=user_ids)
    item_map = pd.Series(np.arange(len(rated)), index=rated)

    inter = pd.DataFrame({
        "user_idx": user_map.loc[ratings["user_id"]].to_numpy(),
        "item_idx": item_map.loc[ratings["movie_id"]].to_numpy(),
        "rating": ratings["rating"].to_numpy().astype(np.int8),
        "timestamp": ratings["timestamp"].to_numpy().astype(np.int64),
    })
    inter["positive"] = inter["rating"] >= positive_threshold
    inter = inter.sort_values(["timestamp", "user_idx", "item_idx"], kind="stable").reset_index(drop=True)

    items = clean_items(movies)
    items["item_idx"] = item_map.loc[items["movie_id"]].to_numpy()
    items = items.sort_values("item_idx").reset_index(drop=True)
    report["items_missing_year"] = int(items["year"].isna().sum())

    users = clean_users(users)
    users["user_idx"] = user_map.loc[users["user_id"]].to_numpy()
    users = users.sort_values("user_idx").reset_index(drop=True)
    return inter, items, users, report


def item_content_matrix(items: pd.DataFrame, title_dims: int = 16, seed: int = 0) -> tuple[np.ndarray, list[str]]:
    """Genres (multi-hot) + release year + decade one-hot + a small SVD of title tf-idf."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    genre = np.zeros((len(items), len(GENRES)), dtype=np.float32)
    gidx = {g: j for j, g in enumerate(GENRES)}
    for row, glist in enumerate(items["genre_list"]):
        for g in glist:
            genre[row, gidx[g]] = 1.0

    year = items["year"].fillna(items["year"].median()).to_numpy()
    year_norm = ((year - year.mean()) / year.std()).astype(np.float32)[:, None]
    decades = np.arange(1910, 2010, 10)
    decade = np.zeros((len(items), len(decades) + 1), dtype=np.float32)  # last column: before 1910
    d_idx = np.clip(((year - 1910) // 10).astype(int), -1, len(decades) - 1)
    decade[np.arange(len(items)), np.where(d_idx < 0, len(decades), d_idx)] = 1.0

    tfidf = TfidfVectorizer(min_df=2, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9']+\b", stop_words="english")
    title_vec = tfidf.fit_transform(items["title_clean"].fillna(""))
    title = np.zeros((len(items), title_dims), dtype=np.float32)
    n_comp = min(title_dims, title_vec.shape[1] - 1)
    if n_comp >= 1:  # only fails on toy data with a tiny vocabulary
        svd = TruncatedSVD(n_components=n_comp, random_state=seed)
        title[:, :n_comp] = svd.fit_transform(title_vec)
    title /= np.linalg.norm(title, axis=1, keepdims=True).clip(1e-8)

    names = [f"genre:{g}" for g in GENRES] + ["year_norm"] + [f"decade:{d}" for d in decades] + ["decade:<1910"]
    names += [f"title_svd:{k}" for k in range(title_dims)]
    return np.hstack([genre, year_norm, decade, title]), names


def user_content_matrix(users: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """One-hot demographics: gender, age group, occupation, first zip digit."""
    parts, names = [], []
    for col in ["gender", "age", "occupation", "zip_region"]:
        dummies = pd.get_dummies(users[col].astype(str), prefix=col)
        parts.append(dummies.to_numpy(dtype=np.float32))
        names += list(dummies.columns)
    return np.hstack(parts), names


def prepare_dataset(cfg: dict) -> dict:
    raw_dir = project_path(cfg["paths"]["raw_dir"])
    out_dir = project_path(cfg["paths"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    scfg = cfg["split"]

    inter, items, users, report = build_tables(raw_dir, cfg["data"]["positive_threshold"])

    cutoff = temporal_cutoff(inter["timestamp"].to_numpy(), scfg["cutoff_quantile"])
    inter["period"] = np.where(inter["timestamp"] < cutoff, "train", "eval")

    train_pos = inter[(inter["period"] == "train") & inter["positive"]]
    user_train_pos = np.bincount(train_pos["user_idx"], minlength=len(users))
    item_train_pos = np.bincount(train_pos["item_idx"], minlength=len(items))

    users["train_positives"] = user_train_pos
    users["train_ratings"] = np.bincount(inter.loc[inter["period"] == "train", "user_idx"], minlength=len(users))
    users["group"] = assign_groups(user_train_pos, scfg["user_sparse_max"])

    eval_pos = inter[(inter["period"] == "eval") & inter["positive"]]
    eval_counts = np.bincount(eval_pos["user_idx"], minlength=len(users))
    users["eval_positives"] = eval_counts
    eligible = np.where(eval_counts >= cfg["eval"]["min_user_positives"])[0]
    roles = split_eval_users(eligible, users["group"].to_numpy(), scfg["val_user_fraction"], scfg["seed"])
    users["role"] = "none"
    users.loc[roles["val"], "role"] = "val"
    users.loc[roles["test"], "role"] = "test"

    items["train_positives"] = item_train_pos
    items["group"] = assign_groups(item_train_pos, scfg["item_sparse_max"])
    order = np.argsort(-item_train_pos, kind="stable")
    n_head = int(round(scfg["head_item_share"] * len(items)))
    head = np.zeros(len(items), dtype=bool)
    head[order[:n_head]] = True
    items["is_head"] = head

    item_content, item_names = item_content_matrix(items, seed=cfg["seed"])
    user_content, user_names = user_content_matrix(users)

    items_out = items.drop(columns=["genre_list"]).copy()
    inter.to_parquet(out_dir / "interactions.parquet", index=False)
    items_out.to_parquet(out_dir / "items.parquet", index=False)
    users.to_parquet(out_dir / "users.parquet", index=False)
    np.save(out_dir / "item_content.npy", item_content)
    np.save(out_dir / "user_content.npy", user_content)

    meta = {
        "dataset": cfg["data"]["name"],
        "positive_threshold": cfg["data"]["positive_threshold"],
        "cutoff_quantile": scfg["cutoff_quantile"],
        "cutoff_timestamp": int(cutoff),
        "cutoff_date": str(pd.to_datetime(cutoff, unit="s")),
        "n_users": int(len(users)),
        "n_items": int(len(items)),
        "n_interactions": int(len(inter)),
        "n_positive": int(inter["positive"].sum()),
        "n_train_interactions": int((inter["period"] == "train").sum()),
        "n_train_positive": int(len(train_pos)),
        "n_eval_interactions": int((inter["period"] == "eval").sum()),
        "n_eval_positive": int(len(eval_pos)),
        "n_val_users": int(len(roles["val"])),
        "n_test_users": int(len(roles["test"])),
        "val_groups": users.loc[roles["val"], "group"].value_counts().to_dict(),
        "test_groups": users.loc[roles["test"], "group"].value_counts().to_dict(),
        "item_groups": items["group"].value_counts().to_dict(),
        "item_content_features": item_names,
        "user_content_features": user_names,
        "preprocessing": report,
    }
    save_json(meta, out_dir / "meta.json")
    log.info("processed data written to %s", out_dir)
    return meta
