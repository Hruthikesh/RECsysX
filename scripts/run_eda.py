"""EDA + graph statistics -> results/tables/eda_*.{json,csv}, results/figures/eda_*.png, graph_*.png

    python scripts/run_eda.py
"""
import argparse
import json

from recsysx.analysis.eda import (burst_stats, dataset_summary, extra_facts, plot_demographics, plot_genres,
                                  plot_item_popularity, plot_ratings_and_time, plot_user_activity,
                                  positive_rate_by_age_gender, split_summary)
from recsysx.analysis.graph_analysis import (oversmoothing_curve, plot_degree_distribution, plot_sampled_subgraph,
                                             walk_return_shares)
from recsysx.config import load_config, project_path
from recsysx.data import RecData
from recsysx.graph import graph_stats
from recsysx.utils import save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. paths.results_dir=...")
    cfg = load_config(overrides=ap.parse_args().set)
    data = RecData.load(cfg)
    tables = project_path(cfg["paths"]["results_dir"]) / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    summary = dataset_summary(data)
    summary.update(burst_stats(data.interactions))
    rdir = cfg["paths"]["results_dir"]
    _, genre_counts = plot_genres(data, rdir)
    summary["movies_per_genre"] = genre_counts
    save_json(summary, tables / "eda_summary.json")

    split = split_summary(data)
    split.to_csv(tables / "eda_split_summary.csv", index=False)

    gstats = graph_stats(data.train_pos, data.n_users, data.n_items)
    save_json(gstats, tables / "graph_stats.json")
    save_json(extra_facts(data), tables / "eda_extra.json")
    positive_rate_by_age_gender(data).to_csv(tables / "eda_positive_rate_by_age_gender.csv")
    save_json({"mean_pairwise_cosine_genre_propagation": oversmoothing_curve(data),
               "node2vec_walk_return_share": walk_return_shares(data)}, tables / "graph_extra.json")

    for fn in (plot_user_activity, plot_item_popularity, plot_ratings_and_time, plot_demographics,
               plot_degree_distribution, plot_sampled_subgraph):
        print("saved", fn(data, rdir))
    print(json.dumps({k: v for k, v in summary.items() if k != "movies_per_genre"}, indent=2))
    print(split.to_string(index=False))
    print(json.dumps(gstats, indent=2))


if __name__ == "__main__":
    main()
