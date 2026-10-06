"""Exploratory analysis of MovieLens-1M (numbers + figures).

Used by scripts/run_eda.py and notebooks/01_eda.ipynb.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.dataset import RecData
from ..data.movielens import AGE_GROUPS, GENRES, OCCUPATIONS
from ..plotting import plt, save

DAY = 86400


def dataset_summary(data: RecData) -> dict:
    inter, users, items = data.interactions, data.users, data.items
    per_user = inter.groupby("user_idx").size()
    per_item = inter.groupby("item_idx").size()
    item_sorted = np.sort(per_item.to_numpy())[::-1]
    top20 = item_sorted[: int(0.2 * len(item_sorted))].sum() / item_sorted.sum()
    ts = pd.to_datetime(inter["timestamp"], unit="s")
    return {
        "users": int(len(users)),
        "items": int(len(items)),
        "ratings": int(len(inter)),
        "positive_ratings (>=4)": int(inter["positive"].sum()),
        "positive_share": float(inter["positive"].mean()),
        "density": float(len(inter) / (len(users) * len(items))),
        "sparsity": float(1 - len(inter) / (len(users) * len(items))),
        "ratings_per_user_mean": float(per_user.mean()),
        "ratings_per_user_median": float(per_user.median()),
        "ratings_per_user_min": int(per_user.min()),
        "ratings_per_user_max": int(per_user.max()),
        "ratings_per_item_mean": float(per_item.mean()),
        "ratings_per_item_median": float(per_item.median()),
        "items_with_lt_10_ratings": int((per_item < 10).sum()),
        "rating_share_of_top20pct_items": float(top20),
        "first_rating": str(ts.min()),
        "last_rating": str(ts.max()),
        "mean_rating": float(inter["rating"].mean()),
        "duplicate_pairs": int(data.meta["preprocessing"]["duplicate_user_item_pairs"]),
        "missing_values": data.meta["preprocessing"]["missing_values"],
        "movies_without_ratings_dropped": int(data.meta["preprocessing"]["movies_without_ratings"]),
    }


def burst_stats(inter: pd.DataFrame) -> dict:
    """How concentrated is a user's rating activity around signup?"""
    first = inter.groupby("user_idx")["timestamp"].transform("min")
    within_day = (inter["timestamp"] - first) <= DAY
    share = within_day.groupby(inter["user_idx"]).mean()
    span_days = inter.groupby("user_idx")["timestamp"].agg(lambda t: (t.max() - t.min()) / DAY)
    return {
        "median_share_of_ratings_in_first_24h": float(share.median()),
        "users_with_half_their_ratings_in_first_24h": float((share >= 0.5).mean()),
        "users_with_all_ratings_in_first_24h": float((share == 1).mean()),
        "median_active_span_days": float(span_days.median()),
    }


def split_summary(data: RecData) -> pd.DataFrame:
    u = data.users
    rows = []
    for role in ("val", "test"):
        sub = u[u["role"] == role]
        for g in ("warm", "sparse", "cold"):
            s = sub[sub["group"] == g]
            rows.append({"role": role, "group": g, "users": len(s),
                         "median_train_positives": float(s["train_positives"].median()) if len(s) else np.nan,
                         "median_eval_positives": float(s["eval_positives"].median()) if len(s) else np.nan,
                         "eval_positives": int(s["eval_positives"].sum())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- figures

def plot_user_activity(data: RecData, results_dir="results"):
    inter = data.interactions
    per_user = inter.groupby("user_idx").size()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    bins = np.logspace(np.log10(per_user.min()), np.log10(per_user.max()), 40)
    ax[0].hist(per_user, bins=bins, color="#1f77b4")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("ratings per user (log)")
    ax[0].set_ylabel("users")
    ax[0].set_title("User activity")
    ax[0].axvline(per_user.median(), color="k", ls="--", lw=1, label=f"median = {per_user.median():.0f}")
    ax[0].legend()
    pos_train = data.users["train_positives"]
    groups = data.users["group"]
    for g, c in (("warm", "#2ca02c"), ("sparse", "#ff7f0e")):
        ax[1].hist(pos_train[groups == g], bins=np.logspace(0, 3.4, 30), alpha=0.7, label=g, color=c)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("training positives per user (log)")
    ax[1].set_title(f"Users at the cutoff ({(groups == 'cold').sum()} have no training positives)")
    ax[1].legend()
    return save(fig, "eda_user_activity.png", results_dir)


def plot_item_popularity(data: RecData, results_dir="results"):
    per_item = data.interactions.groupby("item_idx").size().sort_values(ascending=False).to_numpy()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ranks = np.arange(1, len(per_item) + 1)
    ax[0].loglog(ranks, per_item, color="#d62728")
    ax[0].set_xlabel("item rank by popularity (log)")
    ax[0].set_ylabel("ratings (log)")
    ax[0].set_title("Item popularity (rank-frequency)")
    cum = np.cumsum(per_item) / per_item.sum()
    ax[1].plot(ranks / len(ranks), cum, color="#d62728")
    ax[1].plot([0, 1], [0, 1], color="grey", ls=":", lw=1)
    i20 = int(0.2 * len(ranks)) - 1
    ax[1].scatter([0.2], [cum[i20]], color="k", zorder=3)
    ax[1].annotate(f"top 20% of items\n= {cum[i20]:.0%} of ratings", (0.2, cum[i20]), (0.35, 0.55),
                   arrowprops=dict(arrowstyle="->"))
    ax[1].set_xlabel("share of items (most popular first)")
    ax[1].set_ylabel("share of ratings")
    ax[1].set_title("Long tail: cumulative share of ratings")
    return save(fig, "eda_item_popularity_long_tail.png", results_dir)


def plot_ratings_and_time(data: RecData, results_dir="results"):
    inter = data.interactions
    ts = pd.to_datetime(inter["timestamp"], unit="s")
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    counts = inter["rating"].value_counts().sort_index()
    ax[0].bar(counts.index, counts.to_numpy(), color=["#bbbbbb"] * 3 + ["#2ca02c"] * 2)
    ax[0].set_xlabel("rating")
    ax[0].set_ylabel("count")
    ax[0].set_title(f"Ratings (green = positive, {inter['positive'].mean():.0%})")
    weekly = ts.dt.to_period("W").value_counts().sort_index()
    ax[1].plot(weekly.index.to_timestamp(), weekly.to_numpy(), color="#1f77b4")
    cutoff = pd.to_datetime(data.cutoff, unit="s")
    ax[1].axvline(cutoff, color="k", ls="--", lw=1)
    ax[1].text(cutoff, weekly.max() * 0.9, "  cutoff", fontsize=9)
    ax[1].set_yscale("log")
    ax[1].set_title("Ratings per week")
    ax[1].tick_params(axis="x", rotation=30)
    first = inter.groupby("user_idx")["timestamp"].min()
    share = ((inter["timestamp"] - inter["user_idx"].map(first)) <= DAY).groupby(inter["user_idx"]).mean()
    ax[2].hist(share, bins=20, color="#9467bd")
    ax[2].set_xlabel("share of a user's ratings within 24h of their first")
    ax[2].set_ylabel("users")
    ax[2].set_title("Users rate in one burst after signing up")
    return save(fig, "eda_ratings_time.png", results_dir)


def plot_genres(data: RecData, results_dir="results"):
    inter = data.interactions
    g = data.genre_matrix()
    n_movies = g.sum(0)
    rated = g[inter["item_idx"].to_numpy()]
    n_ratings = rated.sum(0)
    pos_rate = (rated * inter["positive"].to_numpy()[:, None]).sum(0) / n_ratings
    order = np.argsort(-n_ratings)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].barh(np.array(GENRES)[order][::-1], n_ratings[order][::-1] / 1e3, color="#1f77b4", label="ratings (k)")
    ax[0].set_xlabel("ratings (thousands)")
    ax[0].set_title("Ratings per genre (a movie can have several)")
    ax[1].barh(np.array(GENRES)[order][::-1], pos_rate[order][::-1], color="#2ca02c")
    ax[1].axvline(inter["positive"].mean(), color="k", ls="--", lw=1)
    ax[1].set_xlabel("share of ratings >= 4")
    ax[1].set_title("How often a genre is rated positively")
    ax[1].set_xlim(0.3, 0.8)
    fig.tight_layout()
    return save(fig, "eda_genres.png", results_dir), dict(zip(GENRES, n_movies.astype(int)))


def plot_demographics(data: RecData, results_dir="results"):
    u = data.users
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4), gridspec_kw={"width_ratios": [1, 2, 3]})
    gc = u["gender"].value_counts()
    ax[0].bar(gc.index, gc.to_numpy(), color="#7f7f7f")
    ax[0].set_title("Gender")
    ac = u["age"].value_counts().sort_index()
    ax[1].bar([AGE_GROUPS[a] for a in ac.index], ac.to_numpy(), color="#7f7f7f")
    ax[1].set_title("Age group")
    ax[1].tick_params(axis="x", rotation=30)
    oc = u["occupation"].value_counts()
    ax[2].bar([OCCUPATIONS[o] for o in oc.index], oc.to_numpy(), color="#7f7f7f")
    ax[2].set_title("Occupation")
    ax[2].tick_params(axis="x", rotation=75, labelsize=7)
    return save(fig, "eda_demographics.png", results_dir)


def extra_facts(data: RecData) -> dict:
    """Numbers quoted in the docs that are not part of the other EDA tables."""
    inter, users = data.interactions, data.users
    ts = np.sort(inter["timestamp"].to_numpy())
    t80, t90 = ts[int(0.8 * len(ts))], ts[int(0.9 * len(ts))]
    pos = inter[inter["positive"]]
    # the plain 80/10/10 time split that was considered first
    tr_users = set(inter.loc[inter["timestamp"] < t80, "user_idx"])
    trval_users = set(inter.loc[inter["timestamp"] < t90, "user_idx"])
    val_u = set(pos.loc[(pos["timestamp"] >= t80) & (pos["timestamp"] < t90), "user_idx"])
    test_u = set(pos.loc[pos["timestamp"] >= t90, "user_idx"])
    before = pos[pos["period"] == "train"]["item_idx"].value_counts().head(100).index
    after = pos[pos["period"] == "eval"]["item_idx"].value_counts().head(100).index
    new = [i for i in after if i not in set(before)]
    years = data.items.set_index("item_idx").loc[new, "year"]
    monthly = pd.to_datetime(inter["timestamp"], unit="s").dt.to_period("M").value_counts()
    first_pos = pos.groupby("item_idx")["timestamp"].min()
    n_tgt = np.diff(data.targets.indptr)[data.eval_users("test")]
    return {
        "plain_80_10_10_split": {"val_users": len(val_u), "val_users_without_history": len(val_u - tr_users),
                                 "test_users": len(test_u), "test_users_without_history": len(test_u - trval_users)},
        "top100_liked_overlap_before_after_cutoff": int(len(set(before) & set(after))),
        "top100_newcomers": len(new), "top100_newcomers_released_2000": int((years == 2000).sum()),
        "busiest_month": str(monthly.idxmax()), "busiest_month_ratings": int(monthly.max()),
        "movies_with_first_positive_after_cutoff_and_eval_positives": int(
            (first_pos >= data.cutoff).sum()),
        "warm_users_median_train_positives": float(users.loc[users["group"] == "warm", "train_positives"].median()),
        "test_users_targets_median": float(np.median(n_tgt)), "test_users_targets_mean": float(n_tgt.mean()),
        "simulated_new_items": int(len(data.with_cold_items(0.1, seed=42).notes["cold_items"])),
    }


def positive_rate_by_age_gender(data: RecData) -> pd.DataFrame:
    m = data.interactions.merge(data.users[["user_idx", "gender", "age"]], on="user_idx")
    return m.pivot_table(index="age", columns="gender", values="positive", aggfunc="mean")
