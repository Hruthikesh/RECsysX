"""Diversity-aware re-ranking (MMR) on top of the ranker, plus exposure analysis.

Input: the ranked candidate lists written by scripts/run_ranking.py for the held-out
validation users ("es") and the test users.

1. MMR sweep over lambda with two similarity definitions (genre vectors, MF item vectors)
   -> results/tables/diversity_results.csv
2. Operating point: the most diverse lambda whose validation NDCG@10 stays within 2% of
   the un-diversified ranker -> results/tables/diversity_selected.json
3. Exposure of every main model, the ranker and the ranker + MMR
   -> results/tables/exposure_results.csv, exposure_by_popularity_decile.csv

    python scripts/run_reranking.py
"""
import argparse

import numpy as np
import pandas as pd

from recsysx.config import load_config, project_path
from recsysx.data import RecData
from recsysx.evaluation import Evaluator, gini, intra_list_diversity
from recsysx.evaluation.metrics import beyond_accuracy, exposure_counts
from recsysx.experiments.artifacts import load_main_model, load_topk
from recsysx.experiments.suites import MAIN_ORDER, MODEL_SPECS
from recsysx.ranking import mmr_rerank
from recsysx.utils import get_device, load_json, save_json

LAMBDAS = [1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3]
K = 10


def evaluate_lists(data, ev, topk, emb_vectors):
    res = ev.metrics_from_topk(topk, by_group=False)["all"]
    out = {m: res[m] for m in (f"recall@{K}", f"ndcg@{K}", f"precision@{K}", "mrr")}
    out.update(beyond_accuracy(topk, data.n_items, data.genre_matrix(), data.item_popularity(),
                               int(data.known_users().sum()), data.items["is_head"].to_numpy()))
    out[f"emb_diversity@{K}"] = intra_list_diversity(topk, emb_vectors)
    return out


def exposure_row(name, topk, data):
    exp = exposure_counts(topk, data.n_items)
    pop = data.item_popularity()
    head = data.items["is_head"].to_numpy()
    total = exp.sum()
    order = np.argsort(-exp)
    return {
        "list": name,
        "gini": gini(exp),
        "head_exposure_share": float(exp[head].sum() / total),
        "tail_exposure_share": float(exp[~head].sum() / total),
        "items_recommended": int((exp > 0).sum()),
        "catalog_coverage": float((exp > 0).mean()),
        "tail_items_recommended": int(((exp > 0) & ~head).sum()),
        "tail_coverage": float(((exp > 0) & ~head).sum() / (~head).sum()),
        "top10_items_share": float(exp[order[:10]].sum() / total),
        "max_item_share_of_users": float(exp.max() / len(topk)),
        # movies without a single training positive (mostly isolated graph nodes)
        "slots_on_items_without_train_positives": float(exp[pop == 0].sum() / total),
        "items_without_train_positives_shown": int(((exp > 0) & (pop == 0)).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranker-variant", default="temporal", choices=["temporal", "validation"],
                    help="which ranker's lists to re-rank (the final pipeline uses the temporal one)")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. paths.results_dir=...")
    args = ap.parse_args()
    cfg = load_config(overrides=args.set)
    data = RecData.load(cfg)
    device = get_device(cfg["device"])
    tables = project_path(cfg["paths"]["results_dir"]) / "tables"
    ck = project_path(cfg["paths"]["checkpoint_dir"]) / "ranking"
    mf = load_main_model(cfg, data, "mf", 42, device)
    sims = {"genre": data.genre_matrix(), "mf_embedding": mf.item_vectors()}

    # the temporal ranker never saw any validation user, so all of them can be used to pick lambda;
    # the "validation" ranker was fitted on 80% of them, only its held-out 20% ("es") can be used
    if args.ranker_variant == "temporal":
        files = {"es": ck / "temporal_val_ranked.npz", "test": ck / "temporal_test_ranked.npz"}
        ranker_name = "xgb, temporal"
    else:
        files = {"es": ck / "es_ranked.npz", "test": ck / "test_ranked.npz"}
        ranker_name = load_json(ck / "ranking_meta.json")["ranker"]
    rows, lists = [], {}
    for role, path in files.items():
        z = np.load(path)
        users, items, scores = z["users"], z["items"], z["scores"]
        ev = Evaluator(data, "val" if role == "es" else "test", ks=[5, 10], device=device, users=users)
        for sim_name, vecs in sims.items():
            for lam in LAMBDAS:
                if lam == 1.0 and sim_name != "genre":
                    continue  # identical to the genre run with lambda=1
                topk = mmr_rerank(items, np.nan_to_num(scores, neginf=-1e9), vecs, K, lam, device)
                r = {"role": "validation" if role == "es" else "test", "similarity": sim_name if lam < 1 else "none",
                     "lambda": lam, "n_users": len(users), **evaluate_lists(data, ev, topk, sims["mf_embedding"])}
                rows.append(r)
                lists[(role, sim_name, lam)] = topk
    div = pd.DataFrame(rows)
    div.to_csv(tables / "diversity_results.csv", index=False)

    # operating point chosen on validation users
    val = div[div["role"] == "validation"]
    base_ndcg = float(val[val["lambda"] == 1.0][f"ndcg@{K}"].iloc[0])
    ok = val[val[f"ndcg@{K}"] >= 0.98 * base_ndcg]
    pick = ok.sort_values(f"diversity@{K}", ascending=False).iloc[0]
    selected = {"lambda": float(pick["lambda"]), "similarity": pick["similarity"] if pick["lambda"] < 1 else "genre",
                "rule": "max genre diversity@10 with validation ndcg@10 >= 98% of lambda=1",
                "ranker": ranker_name, "n_validation_users": int(val["n_users"].iloc[0]),
                "val_ndcg@10": float(pick[f"ndcg@{K}"]), "val_ndcg@10_no_mmr": base_ndcg}
    save_json(selected, tables / "diversity_selected.json")
    print("selected MMR operating point:", selected)

    # ---------------------------------------------------------------- exposure
    exp_rows, decile_rows = [], []
    pop = data.item_popularity()
    deciles = np.zeros(data.n_items, dtype=int)
    deciles[np.argsort(-pop, kind="stable")] = np.arange(data.n_items) * 10 // data.n_items
    test_users = data.eval_users("test")
    sources = {}
    for key in MAIN_ORDER:
        u, topk = load_topk(cfg, key, 42, "test")
        sources[MODEL_SPECS[key][1]] = topk[:, :K]
    sources[f"Ranker ({ranker_name})"] = lists[("test", "genre", 1.0)]
    sel_key = ("test", selected["similarity"], selected["lambda"])
    sources[f"Ranker ({ranker_name}) + MMR (lambda={selected['lambda']})"] = lists.get(sel_key, lists[("test", "genre", 1.0)])
    # what test users actually liked after the cutoff, as a reference distribution
    tgt = data.targets[test_users].tocoo()
    actual = np.bincount(tgt.col, minlength=data.n_items)
    for name, topk in sources.items():
        exp_rows.append(exposure_row(name, topk, data))
        exp = exposure_counts(topk, data.n_items)
        for d in range(10):
            decile_rows.append({"list": name, "popularity_decile": d + 1,
                                "exposure_share": float(exp[deciles == d].sum() / exp.sum())})
    head = data.items["is_head"].to_numpy()
    exp_rows.append({"list": "Actual test positives", "gini": gini(actual),
                     "head_exposure_share": float(actual[head].sum() / actual.sum()),
                     "tail_exposure_share": float(actual[~head].sum() / actual.sum()),
                     "items_recommended": int((actual > 0).sum()), "catalog_coverage": float((actual > 0).mean()),
                     "tail_items_recommended": int(((actual > 0) & ~head).sum()),
                     "tail_coverage": float(((actual > 0) & ~head).sum() / (~head).sum()),
                     "top10_items_share": float(np.sort(actual)[::-1][:10].sum() / actual.sum()),
                     "max_item_share_of_users": float(actual.max() / len(test_users))})
    for d in range(10):
        decile_rows.append({"list": "Actual test positives", "popularity_decile": d + 1,
                            "exposure_share": float(actual[deciles == d].sum() / actual.sum())})
    pd.DataFrame(exp_rows).to_csv(tables / "exposure_results.csv", index=False)
    pd.DataFrame(decile_rows).to_csv(tables / "exposure_by_popularity_decile.csv", index=False)
    print(div[div["role"] == "test"][["similarity", "lambda", f"recall@{K}", f"ndcg@{K}", f"diversity@{K}",
                                      f"coverage@{K}", f"novelty@{K}"]].to_string(index=False))
    print(pd.DataFrame(exp_rows).to_string(index=False))


if __name__ == "__main__":
    main()
