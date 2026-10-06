"""Result figures -> results/figures/. Reads only files in results/tables and the run logs.

    python scripts/make_figures.py
(EDA/graph figures come from scripts/run_eda.py, the t-SNE plot from scripts/run_analysis.py)
"""
import argparse

import numpy as np
import pandas as pd

from recsysx.config import load_config, project_path
from recsysx.experiments.suites import MAIN_ORDER, MODEL_SPECS
from recsysx.plotting import color, plt, save
from recsysx.utils import load_json

LABELS = [MODEL_SPECS[k][1] for k in MAIN_ORDER]
TRAINED = ["MF-BPR", "Two-Tower", "Node2Vec", "GraphSAGE", "GAT", "LightGCN", "Hybrid (id+content+graph)"]
SHORT = {"Popularity (recent)": "Pop. (recent)", "Popularity (demographic)": "Pop. (demogr.)",
         "Hybrid (id+content+graph)": "Hybrid"}


def short(name):
    if name.startswith("Final pipeline"):
        return "Pipeline + MMR" if "MMR (" in name else "Pipeline"
    if name.startswith("Pipeline, same-period"):
        return "Same-period ranker + MMR*" if "MMR (" in name else "Same-period ranker*"
    return SHORT.get(name, name)


def bars(ax, labels, values, errs=None, colors=None, fmt="{:.3f}"):
    x = np.arange(len(labels))
    ax.bar(x, values, yerr=errs, color=colors, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([short(l) for l in labels], rotation=35, ha="right")
    for xi, v in zip(x, values):
        ax.text(xi, v, fmt.format(v), ha="center", va="bottom", fontsize=7)


def main_comparison(T, rdir):
    m = pd.read_csv(T / "main_results.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    w = 0.26
    x = np.arange(len(m))
    for ax, metric in zip(axes, ("recall", "ndcg")):
        for j, k in enumerate((5, 10, 20)):
            ax.bar(x + (j - 1) * w, m[f"{metric}@{k}"], w, yerr=m[f"{metric}@{k}_std"].fillna(0), capsize=2,
                   label=f"@{k}", color=["#9ecae1", "#4292c6", "#08519c"][j])
        ax.set_xticks(x)
        ax.set_xticklabels([short(l) for l in m["model"]], rotation=35, ha="right")
        ax.set_title(f"{metric.upper()}@K on test users (mean over seeds, error bar = std)")
        ax.legend(loc="upper left", ncol=3)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    save(fig, "main_recall_ndcg_at_k.png", rdir)


def by_group(T, rdir):
    m = pd.read_csv(T / "main_results.csv")
    fig, ax = plt.subplots(figsize=(12, 4))
    w = 0.27
    x = np.arange(len(m))
    for j, (g, c) in enumerate((("warm", "#2ca02c"), ("sparse", "#ff7f0e"), ("cold", "#1f77b4"))):
        ax.bar(x + (j - 1) * w, m[f"{g}/recall@20"], w, yerr=m[f"{g}/recall@20_std"].fillna(0), capsize=2, color=c,
               label=f"{g} users")
    ax.set_xticks(x)
    ax.set_xticklabels([short(l) for l in m["model"]], rotation=35, ha="right")
    ax.set_ylabel("recall@20")
    ax.set_title("Recall@20 by user group (cold = no training history)")
    ax.legend()
    save(fig, "cold_start_user_groups.png", rdir)


def training_curves(cfg, rdir):
    ck = project_path(cfg["paths"]["checkpoint_dir"]) / "main"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for key, label in (("mf", "MF-BPR"), ("two_tower", "Two-Tower"), ("node2vec", "Node2Vec"),
                       ("graphsage", "GraphSAGE"), ("gat", "GAT"), ("lightgcn", "LightGCN"),
                       ("hybrid", "Hybrid (id+content+graph)")):
        path = ck / f"{key}_s42" / "run.json"
        if not path.exists():
            continue
        h = pd.DataFrame(load_json(path)["fit_history"])
        loss = h["loss"] / h["loss"].iloc[0]
        axes[0].plot(h["epoch"], loss, label=short(label), color=color(label))
        axes[1].plot(h["epoch"], h["val/recall@20"], label=short(label), color=color(label))
    axes[0].set_title("Training loss (divided by epoch-1 loss, losses differ per model)")
    axes[0].set_xlabel("epoch")
    axes[1].set_title("Validation recall@20 during training (seed 42)")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    save(fig, "training_curves.png", rdir)


def sage_vs_gat(T, rdir):
    """Depth comparison: 1 layer = the selected config (main suite, seed 42), 2 and 3 layers = the
    gnn_ablation runs with patience 50 (deeper models start improving later)."""
    ga = pd.read_csv(T / "gnn_ablation_runs.csv")
    mr = pd.read_csv(T / "main_runs.csv").set_index("run")
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    for key, label in (("graphsage", "GraphSAGE"), ("gat", "GAT"), ("lightgcn", "LightGCN")):
        rows = [(1, mr.loc[f"{key}_s42", "val/recall@20"], mr.loc[f"{key}_s42", "test/recall@20"],
                 mr.loc[f"{key}_s42", "train_time_s"])]
        for n in (2, 3):
            r = ga[(ga["base"] == key) & (ga["variant"] == f"{n} layers (patience 50)")]
            if len(r):
                rows.append((n, r["val/recall@20"].iloc[0], r["test/recall@20"].iloc[0], r["train_time_s"].iloc[0]))
        layers, val, test, secs = zip(*rows)
        axes[0].plot(layers, val, marker="o", label=label, color=color(label))
        axes[1].plot(layers, test, marker="o", label=label, color=color(label))
        axes[2].plot(layers, secs, marker="o", label=label, color=color(label))
    for ax, t in zip(axes, ("validation recall@20", "test recall@20", "training time until early stop (s)")):
        ax.set_xlabel("message-passing layers")
        ax.set_xticks([1, 2, 3])
        ax.set_title(t)
    axes[0].legend()
    fig.suptitle("Depth: 1 layer = tuned config, 2-3 layers trained with patience 50 (seed 42)", y=1.02, fontsize=10)
    save(fig, "gnn_depth_comparison.png", rdir)


def sparsity(T, rdir):
    s = pd.read_csv(T / "sparsity_results.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for model, g in s.groupby("model", sort=False):
        g = g.sort_values("train_fraction")
        base = g[g["train_fraction"] == 1.0]["recall@20"].iloc[0]
        axes[0].errorbar(g["train_fraction"] * 100, g["recall@20"], yerr=g["recall@20_std"].fillna(0), marker="o",
                         capsize=2, label=short(model), color=color(model))
        axes[1].plot(g["train_fraction"] * 100, g["recall@20"] / base, marker="o", label=short(model), color=color(model))
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks([10, 20, 50, 100])
        ax.set_xticklabels(["10%", "20%", "50%", "100%"])
        ax.set_xlabel("share of training positives kept")
    axes[0].set_title("Test recall@20 vs. training data (mean ± std, 3 seeds)")
    axes[1].set_title("Relative to the same model at 100%")
    axes[1].axhline(1, color="grey", lw=0.8, ls=":")
    axes[0].legend(fontsize=8)
    save(fig, "sparsity_curve.png", rdir)


def item_cold(T, rdir):
    ic = pd.read_csv(T / "item_cold_results.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    bars(axes[0], ic["model"], ic["coldtgt/recall@20_among_new"], ic["coldtgt/recall@20_among_new_std"].fillna(0),
         [color(m, "#17becf") for m in ic["model"]])
    axes[0].set_title("Ranking only the held-out (new) items: recall@20")
    bars(axes[1], ic["model"], ic["coldtgt/recall@100_full_catalog"], ic["coldtgt/recall@100_full_catalog_std"].fillna(0),
         [color(m, "#17becf") for m in ic["model"]])
    axes[1].set_title("New items competing with the full catalog: recall@100\n(share of new-item test positives found)")
    save(fig, "cold_start_items.png", rdir)


def onboarding(T, rdir):
    ob = pd.read_csv(T / "onboarding_results.csv")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for model, g in ob.groupby("model", sort=False):
        g = g.sort_values("k_revealed")
        ax.plot(g["k_revealed"], g["recall@20"], marker="o", label=f"{short(model)} ({g['method'].iloc[-1].split(':')[0]})",
                color=color(model))
    ax.set_xlabel("ratings revealed by the new user")
    ax.set_ylabel("recall@20 on the later ratings")
    ax.set_title(f"New users: what do the first k ratings buy? ({int(ob['n_users'].iloc[0])} cold test users)")
    ax.set_xticks(sorted(ob["k_revealed"].unique()))
    ax.legend(fontsize=7, loc="best")
    save(fig, "cold_start_onboarding.png", rdir)


def diversity(T, rdir):
    dv = pd.read_csv(T / "diversity_results.csv")
    m = pd.read_csv(T / "main_results.csv")
    sel = load_json(T / "diversity_selected.json")
    test = dv[dv["role"] == "test"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for sim, c in (("genre", "#d62728"), ("mf_embedding", "#1f77b4")):
        g = pd.concat([test[test["lambda"] == 1.0], test[test["similarity"] == sim]]).sort_values("lambda", ascending=False)
        axes[0].plot(g["diversity@10"], g["ndcg@10"], marker="o", color=c, label=f"MMR, {sim} similarity")
        for _, r in g.iterrows():
            axes[0].annotate(f"{r['lambda']:g}", (r["diversity@10"], r["ndcg@10"]), fontsize=7, xytext=(3, 3),
                             textcoords="offset points")
        axes[1].plot(g["coverage@10"], g["ndcg@10"], marker="o", color=c, label=f"MMR, {sim} similarity")
    pick = test[(test["lambda"] == sel["lambda"]) & (test["similarity"] == (sel["similarity"] if sel["lambda"] < 1 else "none"))]
    axes[0].scatter(pick["diversity@10"], pick["ndcg@10"], s=150, facecolors="none", edgecolors="k", label="selected on validation")
    axes[0].set_xlabel("intra-list genre diversity@10")
    axes[0].set_ylabel("NDCG@10")
    axes[0].set_title("Relevance vs. diversity (labels = lambda)")
    axes[0].legend(fontsize=8)
    axes[1].scatter(m["coverage@10"], m["ndcg@10"], c=[color(x) for x in m["model"]], marker="s")
    for _, r in m.iterrows():
        axes[1].annotate(short(r["model"]), (r["coverage@10"], r["ndcg@10"]), fontsize=7, xytext=(3, -8),
                         textcoords="offset points")
    axes[1].set_xlabel("catalog coverage@10")
    axes[1].set_title("Coverage vs. NDCG@10: single models (squares) and MMR paths")
    save(fig, "diversity_tradeoff.png", rdir)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(m["coverage@10"], m["diversity@10"], c=[color(x) for x in m["model"]], s=40)
    for _, r in m.iterrows():
        ax.annotate(short(r["model"]), (r["coverage@10"], r["diversity@10"]), fontsize=8, xytext=(4, 2),
                    textcoords="offset points")
    ax.set_xlabel("catalog coverage@10")
    ax.set_ylabel("intra-list genre diversity@10")
    ax.set_title("Coverage vs. diversity of the single models (test users)")
    save(fig, "coverage_vs_diversity.png", rdir)


def exposure(T, rdir, cfg):
    dec = pd.read_csv(T / "exposure_by_popularity_decile.csv")
    exp = pd.read_csv(T / "exposure_results.csv")
    keep = ["Popularity", "ItemKNN", "MF-BPR", "Two-Tower", "Node2Vec", "GraphSAGE", "GAT", "LightGCN", "Hybrid (id+content+graph)"]
    keep += [l for l in exp["list"] if l.startswith("Ranker")] + ["Actual test positives"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    piv = dec.pivot(index="popularity_decile", columns="list", values="exposure_share")
    for name in keep:
        if name in piv.columns:
            ls = "--" if name == "Actual test positives" else "-"
            axes[0].plot(piv.index, piv[name], marker=".", ls=ls, label=short(name), color=color(name, None))
    axes[0].set_yscale("log")
    axes[0].set_xlabel("item popularity decile (1 = most popular 10% of items)")
    axes[0].set_ylabel("share of top-10 slots (log)")
    axes[0].set_title("Where do recommendation slots go?")
    axes[0].legend(fontsize=7, ncol=2)
    e = exp.set_index("list").loc[[k for k in keep if k in set(exp["list"])]]
    bars(axes[1], e.index.tolist(), e["gini"].to_numpy(), colors=[color(n, "#17becf") for n in e.index])
    axes[1].set_title("Gini coefficient of item exposure (1 = all slots to one item)")
    save(fig, "exposure_distribution.png", rdir)


def retrieval(T, rdir):
    r = pd.read_csv(T / "retrieval_results.csv")
    lat = pd.read_csv(T / "retrieval_latency.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ns = [20, 50, 100, 200, 500]
    for model, g in r[r["role"] == "test"].groupby("model", sort=False):
        axes[0].plot(ns, [g[f"recall@{n}"].iloc[0] for n in ns], marker="o", label=short(model), color=color(model))
    axes[0].set_xscale("log")
    axes[0].set_xticks(ns)
    axes[0].set_xticklabels(ns)
    axes[0].set_xlabel("candidates retrieved (N)")
    axes[0].set_ylabel("candidate recall@N")
    axes[0].set_title("Retrieval: share of test positives among the N candidates")
    axes[0].legend(fontsize=8)
    for model, g in lat.groupby("model", sort=False):
        axes[1].plot(g["n"], g["latency_ms_mean"], marker="o", label=short(model), color=color(model))
    axes[1].set_xscale("log")
    axes[1].set_xticks(ns)
    axes[1].set_xticklabels(ns)
    axes[1].set_xlabel("candidates retrieved (N)")
    axes[1].set_ylabel("ms per query (mean, 1 user, CPU)")
    axes[1].set_title("FAISS exact search latency incl. seen-item filtering")
    save(fig, "retrieval_recall_latency.png", rdir)


def ranking(T, rdir):
    rk = pd.read_csv(T / "ranking_results.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, g in rk.groupby("ranker", sort=False):
        axes[0].plot(g["n_candidates"], g["recall@20"], marker="o", label=name)
        axes[1].plot(g["n_candidates"], g["ndcg@10"], marker="o", label=name)
    for ax, t in zip(axes, ("test recall@20", "test NDCG@10")):
        ax.set_xlabel("candidates passed to the ranker (N)")
        ax.set_title(t)
        ax.set_xticks(sorted(rk["n_candidates"].unique()))
    axes[0].legend()
    save(fig, "ranking_comparison.png", rdir)
    fi = pd.read_csv(T / "ranking_feature_importance.csv").head(15)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(fi["feature"], fi["gain"] / fi["gain"].sum(), color="#4292c6")
    ax.set_xlabel("share of total gain (top 15 features)")
    ax.set_title("XGBoost ranker feature importance")
    save(fig, "ranking_feature_importance.png", rdir)


def errors(T, rdir):
    act = pd.read_csv(T / "error_by_activity.csv")
    pop = pd.read_csv(T / "error_by_item_popularity.csv")
    show = ["Popularity", "ItemKNN", "MF-BPR", "Two-Tower", "GraphSAGE", "Hybrid (id+content+graph)", "Final pipeline"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    order = ["0", "1-5", "6-20", "21-50", "51-100", "101-200", ">200"]
    for name in show:
        g = act[act["model"] == name].set_index("train_positives").reindex(order)
        axes[0].plot(order, g["recall@20"], marker="o", label=short(name), color=color(name))
    n = act[act["model"] == show[0]].set_index("train_positives").reindex(order)["users"]
    axes[0].set_xticks(range(len(order)))
    axes[0].set_xticklabels([f"{o}\n(n={int(v) if not np.isnan(v) else 0})" for o, v in zip(order, n)], fontsize=8)
    axes[0].set_xlabel("training positives of the test user")
    axes[0].set_title("Recall@20 by user activity")
    axes[0].legend(fontsize=7)
    buckets = ["head (top 20%)", "mid", "tail (bottom 40%)", "cold (0 train positives)"]
    w = 0.12
    for j, name in enumerate(show):
        g = pop[pop["model"] == name].set_index("item_bucket").reindex(buckets)
        axes[1].bar(np.arange(4) + (j - 3) * w, g["hit_rate@20"], w, label=short(name), color=color(name))
    axes[1].set_xticks(range(4))
    axes[1].set_xticklabels([b.replace(" (", "\n(") for b in buckets], fontsize=8)
    axes[1].set_yscale("symlog", linthresh=1e-3)
    axes[1].set_ylabel("share of test positives found in top-20")
    axes[1].set_title("Which test positives are found, by item popularity")
    save(fig, "error_by_activity_and_popularity.png", rdir)


def final(T, rdir):
    f = pd.read_csv(T / "final_results.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    cols = [color(m, "#000000") if not m.startswith("Pipeline, same") else "#ffffff" for m in f["Model"]]
    for ax, col, title in ((axes[0], "recall@20", "Test recall@20, single models and the final pipeline"),
                           (axes[1], "cold_users_recall@20", "Test recall@20 for cold users (no training history)")):
        bars(ax, f["Model"], f[col], colors=cols)
        for patch, m in zip(ax.patches, f["Model"]):
            if m.startswith("Pipeline, same"):
                patch.set_hatch("//")
                patch.set_edgecolor("#555555")
        ax.set_title(title)
    axes[0].text(0.01, 0.97, "* ranker trained on validation users from the test period (optimistic, see report)",
                 transform=axes[0].transAxes, fontsize=7, va="top")
    save(fig, "final_comparison.png", rdir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", nargs="*", default=[])
    cfg = load_config(overrides=ap.parse_args().set)
    rdir = cfg["paths"]["results_dir"]
    T = project_path(rdir) / "tables"
    jobs = [("main", lambda: main_comparison(T, rdir)), ("groups", lambda: by_group(T, rdir)),
            ("curves", lambda: training_curves(cfg, rdir)), ("sage_vs_gat", lambda: sage_vs_gat(T, rdir)),
            ("sparsity", lambda: sparsity(T, rdir)), ("item_cold", lambda: item_cold(T, rdir)),
            ("onboarding", lambda: onboarding(T, rdir)), ("diversity", lambda: diversity(T, rdir)),
            ("exposure", lambda: exposure(T, rdir, cfg)), ("retrieval", lambda: retrieval(T, rdir)),
            ("ranking", lambda: ranking(T, rdir)), ("errors", lambda: errors(T, rdir)), ("final", lambda: final(T, rdir))]
    for name, fn in jobs:
        try:
            fn()
            print("ok  ", name)
        except FileNotFoundError as exc:
            print("skip", name, "(missing input:", exc.filename, ")")


if __name__ == "__main__":
    main()
