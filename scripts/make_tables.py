"""Turn the raw run files in results/tables/*_runs.csv into the summary tables.

Every number in the summary tables is computed here from the run files, nothing is typed
by hand. Seeds are aggregated as mean and standard deviation.

    python scripts/make_tables.py
"""
import argparse
import json
import warnings

import numpy as np
import pandas as pd

from recsysx.config import load_config, project_path
from recsysx.experiments.suites import MAIN_ORDER, MODEL_SPECS

# adding columns to the wide run tables triggers a harmless fragmentation warning
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

METRICS = ["recall@5", "recall@10", "recall@20", "ndcg@5", "ndcg@10", "ndcg@20", "precision@10", "map@20", "mrr"]
BEYOND = ["coverage@10", "diversity@10", "novelty@10", "personalization@10", "gini@10", "tail_share@10"]
GROUP_COLS = [f"{g}/{m}" for g in ("warm", "sparse", "cold") for m in ("recall@20", "ndcg@20")]


def agg(df: pd.DataFrame, by: list[str], cols: list[str]) -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns]
    g = df.groupby(by, sort=False)[cols]
    return pd.concat([g.mean(), g.std(ddof=1).add_suffix("_std"), g.size().rename("n_seeds")], axis=1).reset_index()


def label_order(df: pd.DataFrame, col: str = "model") -> pd.DataFrame:
    order = {MODEL_SPECS[k][1]: j for j, k in enumerate(MAIN_ORDER)}
    return df.sort_values(col, key=lambda s: s.map(order).fillna(99), kind="stable")


def fmt(mean, std=None, digits=4):
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return "n/a"
    if std is None or np.isnan(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r.to_numpy()) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", nargs="*", default=[])
    cfg = load_config(overrides=ap.parse_args().set)
    T = project_path(cfg["paths"]["results_dir"]) / "tables"
    md = {}

    main_runs = pd.read_csv(T / "main_runs.csv")
    test_cols = [f"test/{m}" for m in METRICS + BEYOND + GROUP_COLS]
    other = ["train_time_s", "n_params", "best_epoch", "peak_gpu_mem_mb", "val/recall@20"]
    main = label_order(agg(main_runs, ["model"], test_cols + other))
    main.columns = [c.replace("test/", "") for c in main.columns]
    main.to_csv(T / "main_results.csv", index=False)

    # ---------------------------------------------------------------- per-family tables
    base_models = [MODEL_SPECS[k][1] for k in ("popularity", "popularity_recent", "popularity_demo", "itemknn")]
    main[main["model"].isin(base_models)].to_csv(T / "baseline_results.csv", index=False)
    mf = main[main["model"] == "MF-BPR"].copy()
    mf.to_csv(T / "mf_results.csv", index=False)
    lp = pd.read_csv(T / "link_prediction_results.csv") if (T / "link_prediction_results.csv").exists() else None
    graph = main[main["model"].isin(["MF-BPR", "Node2Vec", "GraphSAGE", "GAT", "LightGCN", "Hybrid (id+content+graph)"])].copy()
    if lp is not None:
        graph = graph.merge(lp[["model", "auc", "ap"]].rename(columns={"auc": "link_auc", "ap": "link_ap"}),
                            on="model", how="left")
    graph.to_csv(T / "graph_results.csv", index=False)

    md["main"] = to_markdown(pd.DataFrame({
        "Model": main["model"],
        "Recall@10": [fmt(a, b) for a, b in zip(main["recall@10"], main["recall@10_std"])],
        "Recall@20": [fmt(a, b) for a, b in zip(main["recall@20"], main["recall@20_std"])],
        "NDCG@10": [fmt(a, b) for a, b in zip(main["ndcg@10"], main["ndcg@10_std"])],
        "NDCG@20": [fmt(a, b) for a, b in zip(main["ndcg@20"], main["ndcg@20_std"])],
        "Seeds": main["n_seeds"],
    }))

    # ---------------------------------------------------------------- cold-start
    rows = []
    for _, r in main.iterrows():
        for g in ("warm", "sparse", "cold"):
            rows.append({"section": "user group (natural split)", "model": r["model"], "group": g,
                         "recall@20": r[f"{g}/recall@20"], "recall@20_std": r[f"{g}/recall@20_std"],
                         "ndcg@20": r[f"{g}/ndcg@20"], "ndcg@20_std": r[f"{g}/ndcg@20_std"]})
    user_groups = pd.DataFrame(rows)
    sizes = {g: int(main_runs[f"test/{g}/n_users"].iloc[0]) for g in ("warm", "sparse", "cold")}
    md["user_groups"] = to_markdown(pd.DataFrame({
        "Model": main["model"],
        **{f"{g} (n={sizes[g]})": [fmt(a, b) for a, b in zip(main[f"{g}/recall@20"], main[f"{g}/recall@20_std"])]
           for g in ("warm", "sparse", "cold")}}))
    cold_frames = [user_groups]
    if (T / "item_cold_runs.csv").exists():
        ic = pd.read_csv(T / "item_cold_runs.csv")
        ic_cols = ["coldtgt/recall@20_among_new", "coldtgt/ndcg@20_among_new", "coldtgt/recall@20_full_catalog",
                   "coldtgt/recall@100_full_catalog", "cold_item_share_in_top20", "test/recall@20"]
        ica = label_order(agg(ic, ["model"], ic_cols))
        ica.to_csv(T / "item_cold_results.csv", index=False)
        cold_frames.append(ica.assign(section="item cold-start (10% of items held out)"))
        md["item_cold"] = to_markdown(pd.DataFrame({
            "Model": ica["model"],
            "Recall@20 among new items": [fmt(a, b) for a, b in zip(ica["coldtgt/recall@20_among_new"], ica["coldtgt/recall@20_among_new_std"])],
            "NDCG@20 among new items": [fmt(a, b) for a, b in zip(ica["coldtgt/ndcg@20_among_new"], ica["coldtgt/ndcg@20_among_new_std"])],
            "Recall@100 of new items, full catalog": [fmt(a, b) for a, b in zip(ica["coldtgt/recall@100_full_catalog"], ica["coldtgt/recall@100_full_catalog_std"])],
            "Share of new items in top-20": [fmt(a, b, 3) for a, b in zip(ica["cold_item_share_in_top20"], ica["cold_item_share_in_top20_std"])],
            "Overall recall@20": [fmt(a, b) for a, b in zip(ica["test/recall@20"], ica["test/recall@20_std"])],
        }))
    if (T / "onboarding_runs.csv").exists():
        ob = pd.read_csv(T / "onboarding_runs.csv")
        cold_frames.append(ob.assign(section="new-user onboarding (k revealed ratings)"))
        piv = ob.pivot(index="model", columns="k_revealed", values="recall@20")
        piv = label_order(piv.reset_index())
        md["onboarding"] = to_markdown(pd.DataFrame({
            "Model": piv["model"], **{f"k={k}": [fmt(v) for v in piv[k]] for k in piv.columns if k != "model"}}))
        ob.to_csv(T / "onboarding_results.csv", index=False)
    pd.concat(cold_frames, ignore_index=True).to_csv(T / "cold_start_results.csv", index=False)

    # ---------------------------------------------------------------- sparsity
    if (T / "sparsity_runs.csv").exists():
        sp = pd.read_csv(T / "sparsity_runs.csv")
        full = main_runs.assign(train_fraction=1.0)
        full = full[full["key"].isin(sp["key"].unique())]
        both = pd.concat([sp, full], ignore_index=True)
        spa = agg(both, ["model", "train_fraction"], [f"test/{m}" for m in ("recall@20", "ndcg@20", "warm/recall@20",
                                                                             "sparse/recall@20", "cold/recall@20", "coverage@10")]
                  + ["train_time_s"])
        spa.columns = [c.replace("test/", "") for c in spa.columns]
        spa = label_order(spa.sort_values("train_fraction", ascending=False))
        spa.to_csv(T / "sparsity_results.csv", index=False)
        # .pivot, not .pivot_table: pivot_table silently drops all-NaN columns (std with 1 seed)
        piv = spa.pivot(index="model", columns="train_fraction", values="recall@20")
        pstd = spa.pivot(index="model", columns="train_fraction", values="recall@20_std")
        piv = label_order(piv.reset_index())
        fr = sorted([c for c in piv.columns if c != "model"], reverse=True)
        md["sparsity"] = to_markdown(pd.DataFrame({
            "Model": piv["model"],
            **{f"{int(f * 100)}%": [fmt(piv.loc[i, f], pstd.loc[piv.loc[i, 'model'], f]) for i in piv.index] for f in fr},
            "kept at 10%": [f"{piv.loc[i, min(fr)] / piv.loc[i, 1.0]:.0%}" for i in piv.index]}))

    # ---------------------------------------------------------------- hybrid ablation
    if (T / "hybrid_runs.csv").exists():
        hy = pd.read_csv(T / "hybrid_runs.csv")
        hya = agg(hy, ["variant"], [f"test/{m}" for m in ("recall@20", "ndcg@20", "warm/recall@20", "sparse/recall@20",
                                                          "cold/recall@20", "coverage@10")] + ["n_params"])
        hya.columns = [c.replace("test/", "") for c in hya.columns]
        hya.to_csv(T / "hybrid_results.csv", index=False)
        md["hybrid"] = to_markdown(pd.DataFrame({
            "Variant": hya["variant"],
            "Recall@20": [fmt(a, b) for a, b in zip(hya["recall@20"], hya["recall@20_std"])],
            "NDCG@20": [fmt(a, b) for a, b in zip(hya["ndcg@20"], hya["ndcg@20_std"])],
            "Warm R@20": [fmt(a, b) for a, b in zip(hya["warm/recall@20"], hya["warm/recall@20_std"])],
            "Sparse R@20": [fmt(a, b) for a, b in zip(hya["sparse/recall@20"], hya["sparse/recall@20_std"])],
            "Cold R@20": [fmt(a, b) for a, b in zip(hya["cold/recall@20"], hya["cold/recall@20_std"])],
        }))

    # ---------------------------------------------------------------- retrieval / ranking
    if (T / "retrieval_results.csv").exists():
        rr = pd.read_csv(T / "retrieval_results.csv")
        rr = label_order(rr[rr["role"] == "test"])
        lat = pd.read_csv(T / "retrieval_latency.csv")
        l200 = lat[lat["n"] == 200].set_index("model")["latency_ms_mean"]
        md["retrieval"] = to_markdown(pd.DataFrame({
            "Retriever": rr["model"],
            **{f"Recall@{n}": rr[f"recall@{n}"].map(lambda x: f"{x:.3f}") for n in (50, 100, 200, 500)},
            "Cold users Recall@200": rr["cold/recall@200"].map(lambda x: f"{x:.3f}"),
            "ms/query (N=200)": [f"{l200.get(m, np.nan):.2f}" for m in rr["model"]],
        }))
    if (T / "ranking_results.csv").exists():
        rk = pd.read_csv(T / "ranking_results.csv")
        md["ranking"] = to_markdown(pd.DataFrame({
            "N": rk["n_candidates"], "Ranker": rk["ranker"],
            "Val. recall@20 (selection)": rk["es_recall@20"].map(lambda x: f"{x:.4f}"),
            "Test recall@20": rk["recall@20"].map(lambda x: f"{x:.4f}"),
            "Test NDCG@10": rk["ndcg@10"].map(lambda x: f"{x:.4f}"),
            "Test NDCG@20": rk["ndcg@20"].map(lambda x: f"{x:.4f}"),
            "Cold users R@20": rk["cold/recall@20"].map(lambda x: f"{x:.4f}"),
        }))

    # ---------------------------------------------------------------- analysis tables for the docs
    def f(x, d=4):
        return "n/a" if pd.isna(x) else f"{x:.{d}f}"
    if (T / "significance_tests.csv").exists():
        sg = pd.read_csv(T / "significance_tests.csv")
        sg = sg[sg["users"].isin(["all"]) | ((sg["model_a"] == "LightGCN") & (sg["model_b"] == "MF-BPR"))]
        md["significance"] = to_markdown(pd.DataFrame({
            "A vs. B": sg["model_a"] + " vs. " + sg["model_b"], "users": sg["users"] + " (" + sg["n"].astype(str) + ")",
            "mean diff. recall@20": sg["diff"].map(lambda x: f"{x:+.4f}"),
            "95% CI": [f"[{a:+.4f}, {b:+.4f}]" for a, b in zip(sg["ci95_low"], sg["ci95_high"])],
            "Wilcoxon p": sg["wilcoxon_p"].map(lambda x: f(x, 4)),
            "users A better / B better": sg["a_better_users"].astype(str) + " / " + sg["b_better_users"].astype(str)}))
    if lp is not None:
        lpo = label_order(lp)
        md["link"] = to_markdown(pd.DataFrame({
            "Model": lpo["model"], "AUC": lpo["auc"].map(f), "AP": lpo["ap"].map(f),
            "AUC warm": lpo["warm/auc"].map(f), "AUC sparse": lpo["sparse/auc"].map(f), "AUC cold": lpo["cold/auc"].map(f),
            "test recall@20": [f(main.set_index("model").loc[x, "recall@20"]) for x in lpo["model"]]}))
    if (T / "ranking_temporal_check.csv").exists():
        tc = pd.read_csv(T / "ranking_temporal_check.csv")
        md["temporal"] = to_markdown(pd.DataFrame({
            "users": tc["role"], "ranking": tc["ranker"], "recall@20": tc["recall@20"].map(f), "NDCG@20": tc["ndcg@20"].map(f),
            "warm R@20": tc["warm/recall@20"].map(f), "sparse R@20": tc["sparse/recall@20"].map(f),
            "cold R@20": tc["cold/recall@20"].map(f)}))
    if (T / "gnn_ablation_runs.csv").exists():
        ga = pd.read_csv(T / "gnn_ablation_runs.csv")
        mr = main_runs.set_index("run")
        sel_rows = [{"base": k, "variant": "selected config (main run, seed 42)", "val/recall@20": mr.loc[f"{k}_s42", "val/recall@20"],
                     "test/recall@20": mr.loc[f"{k}_s42", "test/recall@20"], "test/warm/recall@20": mr.loc[f"{k}_s42", "test/warm/recall@20"],
                     "test/cold/recall@20": mr.loc[f"{k}_s42", "test/cold/recall@20"], "best_epoch": mr.loc[f"{k}_s42", "best_epoch"]}
                    for k in ("graphsage", "gat", "lightgcn") if f"{k}_s42" in mr.index]
        gt = pd.concat([pd.DataFrame(sel_rows), ga], ignore_index=True)
        gt["order"] = gt["base"].map({"graphsage": 0, "gat": 1, "lightgcn": 2})
        gt = gt.sort_values(["order"], kind="stable")
        md["gnn_ablation"] = to_markdown(pd.DataFrame({
            "Model": gt["base"].map({"graphsage": "GraphSAGE", "gat": "GAT", "lightgcn": "LightGCN"}), "Variant": gt["variant"],
            "Val R@20": gt["val/recall@20"].map(f), "Test R@20": gt["test/recall@20"].map(f),
            "Test warm R@20": gt["test/warm/recall@20"].map(f), "Test cold R@20": gt["test/cold/recall@20"].map(f),
            "Best epoch": gt["best_epoch"].map(lambda x: f"{x:.0f}")}))
    if (T / "diversity_results.csv").exists():
        dv = pd.read_csv(T / "diversity_results.csv")
        dt = dv[dv["role"] == "test"]
        md["diversity"] = to_markdown(pd.DataFrame({
            "similarity": dt["similarity"], "lambda": dt["lambda"], "recall@10": dt["recall@10"].map(f),
            "NDCG@10": dt["ndcg@10"].map(f), "genre diversity@10": dt["diversity@10"].map(lambda x: f(x, 3)),
            "embedding diversity@10": dt["emb_diversity@10"].map(lambda x: f(x, 3)), "coverage@10": dt["coverage@10"].map(lambda x: f(x, 3)),
            "novelty@10": dt["novelty@10"].map(lambda x: f(x, 2)), "tail share@10": dt["tail_share@10"].map(lambda x: f(x, 3))}))
    if (T / "exposure_results.csv").exists():
        ex = pd.read_csv(T / "exposure_results.csv")
        md["exposure"] = to_markdown(pd.DataFrame({
            "List (top-10, 881 test users)": ex["list"], "Gini": ex["gini"].map(lambda x: f(x, 3)),
            "share of slots on head items": ex["head_exposure_share"].map(lambda x: f(x, 3)),
            "movies shown at least once": ex["items_recommended"], "tail movies shown": ex["tail_items_recommended"],
            "share of slots on the 10 most shown movies": ex["top10_items_share"].map(lambda x: f(x, 3))}))
    if (T / "personalization_results.csv").exists():
        pr = pd.read_csv(T / "personalization_results.csv")
        md["personalization"] = to_markdown(pd.DataFrame({
            "Model": pr["model"], "personalization@10": pr["personalization@10"].map(lambda x: f(x, 3)),
            "mean Jaccard with popularity list": pr["mean_jaccard_with_popularity_list"].map(lambda x: f(x, 3)),
            "distinct lists (of 881)": pr["distinct_lists"], "coverage@10": pr["coverage@10"].map(lambda x: f(x, 3)),
            "novelty@10": pr["novelty@10"].map(lambda x: f(x, 2)), "genre diversity@10": pr["diversity@10"].map(lambda x: f(x, 3))}))

    # ---------------------------------------------------------------- ablation summary
    abl = []

    def add(ablation, variant, src, rec20, ndcg20, std=None, note=""):
        abl.append({"ablation": ablation, "variant": variant, "test_recall@20": rec20, "test_ndcg@20": ndcg20,
                    "recall@20_std": std, "source": src, "note": note})

    m = main.set_index("model")
    for name, models in (("1. MF vs GraphSAGE", ["MF-BPR", "GraphSAGE"]), ("2. GraphSAGE vs GAT", ["GraphSAGE", "GAT"]),
                         ("3. Node2Vec vs GraphSAGE vs GAT", ["Node2Vec", "GraphSAGE", "GAT"])):
        for mo in models:
            add(name, mo, "main_results.csv", m.loc[mo, "recall@20"], m.loc[mo, "ndcg@20"], m.loc[mo, "recall@20_std"])
    if (T / "hybrid_results.csv").exists():
        for _, r in hya.iterrows():
            add("4. graph / content / hybrid", r["variant"], "hybrid_results.csv", r["recall@20"], r["ndcg@20"], r["recall@20_std"],
                f"cold users recall@20 {r['cold/recall@20']:.4f}")
    if (T / "diversity_results.csv").exists():
        dv = pd.read_csv(T / "diversity_results.csv")
        sel = json.loads((T / "diversity_selected.json").read_text())
        dt = dv[dv["role"] == "test"]
        for lam, sim in ((1.0, "none"), (sel["lambda"], sel["similarity"])):
            r = dt[(dt["lambda"] == lam) & (dt["similarity"] == sim)].iloc[0]
            abl.append({"ablation": "5. diversity re-ranking", "variant": f"MMR lambda={lam} ({sim})",
                        "test_recall@10": r["recall@10"], "test_ndcg@10": r["ndcg@10"], "source": "diversity_results.csv",
                        "note": f"diversity@10 {r['diversity@10']:.3f}, coverage@10 {r['coverage@10']:.3f}, novelty@10 {r['novelty@10']:.2f}"})
    if (T / "sparsity_results.csv").exists():
        for _, r in spa[spa["model"].isin(["MF-BPR", "GraphSAGE", "GAT", "Hybrid (id+content+graph)"])].iterrows():
            add("6. interaction-data size", f"{r['model']} @ {r['train_fraction']:.0%}", "sparsity_results.csv",
                r["recall@20"], r["ndcg@20"], r["recall@20_std"])
    for mo in ("MF-BPR", "GraphSAGE", "Hybrid (id+content+graph)"):
        for g in ("warm", "sparse", "cold"):
            add("7. warm vs cold users", f"{mo} / {g}", "main_results.csv", m.loc[mo, f"{g}/recall@20"],
                m.loc[mo, f"{g}/ndcg@20"], m.loc[mo, f"{g}/recall@20_std"])
    if (T / "ranking_results.csv").exists():
        rk = pd.read_csv(T / "ranking_results.csv")
        meta = json.loads((project_path(cfg["paths"]["checkpoint_dir"]) / "ranking" / "ranking_meta.json").read_text())
        for _, r in rk.iterrows():
            if r["n_candidates"] == meta["n_candidates"]:
                add("8. retrieval only vs retrieval + ranking", f"{r['ranker']} (N={r['n_candidates']})", "ranking_results.csv",
                    r["recall@20"], r["ndcg@20"], note=f"held-out val recall@20 {r['es_recall@20']:.4f}")
            if r["ranker"] == meta["ranker"]:
                add("9. candidate count", f"{r['ranker']} N={r['n_candidates']}", "ranking_results.csv", r["recall@20"], r["ndcg@20"],
                    note=f"held-out val recall@20 {r['es_recall@20']:.4f}")
    if (T / "ranking_temporal_check.csv").exists():
        tc = pd.read_csv(T / "ranking_temporal_check.csv")
        for _, r in tc[tc["role"] == "test"].iterrows():
            add("8b. where the ranker's training labels come from", r["ranker"], "ranking_temporal_check.csv",
                r["recall@20"], r["ndcg@20"], note=f"warm {r['warm/recall@20']:.4f}, cold {r['cold/recall@20']:.4f}")
    if (T / "gnn_ablation_runs.csv").exists():
        ga = pd.read_csv(T / "gnn_ablation_runs.csv")
        for _, r in ga.iterrows():
            add("10. GNN design choices", f"{r['base']}: {r['variant']}", "gnn_ablation_runs.csv", r["test/recall@20"],
                r["test/ndcg@20"], note=f"val recall@20 {r['val/recall@20']:.4f}, cold recall@20 {r['test/cold/recall@20']:.4f}")
        for key, lab in (("graphsage", "GraphSAGE"), ("gat", "GAT")):
            add("10. GNN design choices", f"{key}: selected config", "main_results.csv", m.loc[lab, "recall@20"],
                m.loc[lab, "ndcg@20"], m.loc[lab, "recall@20_std"], note=f"val recall@20 {m.loc[lab, 'val/recall@20']:.4f}")
    if (T / "ranking_feature_ablation.csv").exists():
        fa = pd.read_csv(T / "ranking_feature_ablation.csv")
        for _, r in fa.iterrows():
            add("11. ranking features (XGBoost)", r["features"], "ranking_feature_ablation.csv", r["recall@20"], r["ndcg@20"],
                note=f"{r['n_features']} features")
    pd.DataFrame(abl).to_csv(T / "ablation_results.csv", index=False)

    # ---------------------------------------------------------------- final comparison table
    lat = pd.read_csv(T / "retrieval_latency.csv") if (T / "retrieval_latency.csv").exists() else None
    meta = json.loads((project_path(cfg["paths"]["checkpoint_dir"]) / "ranking" / "ranking_meta.json").read_text()) \
        if (project_path(cfg["paths"]["checkpoint_dir"]) / "ranking" / "ranking_meta.json").exists() else None
    item_cold = ica.set_index("model") if (T / "item_cold_results.csv").exists() else None
    final = []
    for _, r in main.iterrows():
        row = {"Model": r["model"]}
        for c in ["recall@5", "recall@10", "recall@20", "ndcg@5", "ndcg@10", "ndcg@20", "precision@10", "map@20", "mrr",
                  "coverage@10", "diversity@10", "novelty@10"]:
            row[c] = r[c]
        row["cold_users_recall@20"] = r["cold/recall@20"]
        row["sparse_users_recall@20"] = r["sparse/recall@20"]
        row["new_items_recall@20"] = item_cold.loc[r["model"], "coldtgt/recall@20_among_new"] if item_cold is not None and r["model"] in item_cold.index else np.nan
        row["train_time_s"] = r["train_time_s"]
        row["params"] = r["n_params"]
        if lat is not None and meta is not None and r["model"] in set(lat["model"]):
            row["retrieval_latency_ms"] = float(lat[(lat["model"] == r["model"]) & (lat["n"] == meta["n_candidates"])]["latency_ms_mean"].iloc[0])
        else:
            row["retrieval_latency_ms"] = np.nan
        row["ranking_latency_ms"] = np.nan
        final.append(row)
    if (T / "final_pipeline_results.csv").exists():
        fp = pd.read_csv(T / "final_pipeline_results.csv")
        fl = json.loads((T / "final_pipeline_latency.json").read_text())
        for _, r in fp.iterrows():
            row = {"Model": r["pipeline"]}
            for c in ["recall@5", "recall@10", "recall@20", "ndcg@5", "ndcg@10", "ndcg@20", "precision@10", "map@20", "mrr",
                      "coverage@10", "diversity@10", "novelty@10"]:
                row[c] = r[c]
            row["cold_users_recall@20"] = r["cold/recall@20"]
            row["sparse_users_recall@20"] = r["sparse/recall@20"]
            row["new_items_recall@20"] = np.nan
            row["train_time_s"] = np.nan
            row["params"] = np.nan
            row["retrieval_latency_ms"] = fl["retrieval_ms_mean"]
            row["ranking_latency_ms"] = fl["features_ms_mean"] + fl["ranking_ms_mean"] + (fl["rerank_ms_mean"] if "+ MMR" in r["pipeline"] else 0)
            final.append(row)
    final = pd.DataFrame(final)
    final.to_csv(T / "final_results.csv", index=False)

    def f4(x):
        return "n/a" if pd.isna(x) else f"{x:.4f}"
    md["final"] = to_markdown(pd.DataFrame({
        "Model": final["Model"],
        "R@5": final["recall@5"].map(f4), "R@10": final["recall@10"].map(f4), "R@20": final["recall@20"].map(f4),
        "N@5": final["ndcg@5"].map(f4), "N@10": final["ndcg@10"].map(f4), "N@20": final["ndcg@20"].map(f4),
        "P@10": final["precision@10"].map(f4),
        "Cov@10": final["coverage@10"].map(lambda x: f"{x:.3f}"), "Div@10": final["diversity@10"].map(lambda x: f"{x:.3f}"),
        "Nov@10": final["novelty@10"].map(lambda x: f"{x:.2f}"),
        "Cold users R@20": final["cold_users_recall@20"].map(f4),
        "New items R@20": final["new_items_recall@20"].map(f4),
        "Train (s)": final["train_time_s"].map(lambda x: "n/a" if pd.isna(x) else f"{x:.0f}"),
        "Retr. ms": final["retrieval_latency_ms"].map(lambda x: "n/a" if pd.isna(x) else f"{x:.2f}"),
        "Rank ms": final["ranking_latency_ms"].map(lambda x: "n/a" if pd.isna(x) else f"{x:.1f}"),
        "Params": final["params"].map(lambda x: "n/a" if pd.isna(x) else (f"{x / 1e3:.0f}k" if x > 0 else "0")),
    }))
    (T / "summary_tables.md").write_text("\n\n".join(f"<!-- {k} -->\n{v}" for k, v in md.items()) + "\n", encoding="utf-8")
    print((T / "summary_tables.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
