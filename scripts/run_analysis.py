"""Analysis on top of the trained models (main suite seed 42 + final pipeline):

  link prediction AUC/AP           -> link_prediction_results.csv
  personalization                  -> personalization_results.csv, recommendation_examples.md
  paired significance tests        -> significance_tests.csv
  error analysis                   -> error_*.csv, error_examples.md
  onboarding vector shift (k=0 vs k=10 revealed ratings) -> onboarding_vector_shift.csv
  embedding check (t-SNE + genre purity of nearest neighbours) -> embedding_genre_purity.csv

    python scripts/run_analysis.py
"""
import argparse

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.manifold import TSNE

from recsysx.analysis.errors import (ACTIVITY_BINS, ACTIVITY_LABELS, graph_vs_mf, item_popularity_bucket,
                                     per_user_recall, target_hit_rate_by_bucket, user_profile_stats)
from recsysx.config import load_config, project_path
from recsysx.data import RecData
from recsysx.data.movielens import GENRES
from recsysx.evaluation.link_prediction import auc_ap, link_prediction_scores, sample_eval_edges
from recsysx.evaluation.metrics import beyond_accuracy
from recsysx.experiments.artifacts import load_main_model, load_topk
from recsysx.experiments.suites import MAIN_ORDER, MODEL_SPECS
from recsysx.pipeline import RecommendationPipeline
from recsysx.plotting import plt, save
from recsysx.retrieval import FaissRetriever
from recsysx.utils import get_device, save_json

SEED = 42


def jaccard_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.empty(len(a))
    for j, (x, y) in enumerate(zip(a, b)):
        inter = len(np.intersect1d(x, y))
        out[j] = inter / (len(x) + len(y) - inter)
    return out


def fmt_items(items, data, hits=None) -> str:
    titles, genres = data.items["title"].to_numpy(), data.items["genres"].to_numpy()
    lines = []
    for r, i in enumerate(items, 1):
        mark = " **(hit)**" if hits is not None and i in hits else ""
        lines.append(f"{r}. {titles[i]} — {genres[i]}{mark}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. paths.results_dir=...")
    cfg = load_config(overrides=ap.parse_args().set)
    data = RecData.load(cfg)
    device = get_device(cfg["device"])
    tables = project_path(cfg["paths"]["results_dir"]) / "tables"
    users = data.eval_users("test")
    labels = {k: MODEL_SPECS[k][1] for k in MAIN_ORDER}

    lists = {}
    for key in MAIN_ORDER:
        u, topk = load_topk(cfg, key, SEED, "test")
        assert (u == users).all()
        lists[labels[key]] = topk
    pipe = RecommendationPipeline.from_artifacts(cfg, data)
    lists["Final pipeline"], _, _ = pipe.recommend_batch(users, k=20, diversify=True)

    # ---------------------------------------------------------------- link prediction
    eu, ei, ey = sample_eval_edges(users, data.targets, data.history, data.n_items, seed=SEED)
    lp_rows = []
    for key in MAIN_ORDER:
        m = load_main_model(cfg, data, key, SEED, device)
        s = link_prediction_scores(m.score, eu, ei, device)
        res = auc_ap(s, ey)
        groups = data.group[eu]
        for g in ("warm", "sparse", "cold"):
            sel = groups == g
            res[f"{g}/auc"] = auc_ap(s[sel], ey[sel])["auc"]
        lp_rows.append({"model": labels[key], **res, "n_pos_edges": int(ey.sum()), "n_neg_edges": int((1 - ey).sum())})
        del m
        torch.cuda.empty_cache()
    pd.DataFrame(lp_rows).to_csv(tables / "link_prediction_results.csv", index=False)
    print(pd.DataFrame(lp_rows).to_string(index=False))

    # ---------------------------------------------------------------- personalization
    pop_list = lists["Popularity"][:, :10]
    pers_rows = []
    for name, topk in lists.items():
        t10 = topk[:, :10]
        ba = beyond_accuracy(t10, data.n_items, data.genre_matrix(), data.item_popularity(),
                             int(data.known_users().sum()), data.items["is_head"].to_numpy())
        jac = jaccard_rows(t10, pop_list)
        distinct = len({tuple(sorted(r)) for r in t10.tolist()})
        pers_rows.append({"model": name, **ba, "mean_jaccard_with_popularity_list": float(jac.mean()),
                          "share_users_list_differs_from_popularity_in_5plus": float((jac < 5 / 15).mean()),
                          "distinct_lists": distinct, "users": len(t10)})
    pers = pd.DataFrame(pers_rows)
    pers.to_csv(tables / "personalization_results.csv", index=False)
    print(pers[["model", "personalization@10", "coverage@10", "mean_jaccard_with_popularity_list", "distinct_lists"]].to_string(index=False))

    # example users (picked by rule, not by hand): the warm user with the most concentrated
    # taste, a sparse user, a cold user - all with at least 10 test positives
    stats = user_profile_stats(data)
    n_tgt = np.diff(data.targets.indptr)[users]
    cand = pd.DataFrame({"user": users, "group": data.group[users], "n_tgt": n_tgt,
                         "top_share": stats["top_genre_share"].to_numpy()[users]})
    cand = cand[cand["n_tgt"] >= 10]
    picks = [("warm user with a narrow taste", cand[cand["group"] == "warm"].sort_values("top_share").iloc[-1]["user"]),
             ("sparse user", cand[cand["group"] == "sparse"].sort_values("user").iloc[0]["user"]),
             ("cold user (no training history)", cand[cand["group"] == "cold"].sort_values("user").iloc[0]["user"])]
    show = ["Popularity", "MF-BPR", "GraphSAGE", "Hybrid (id+content+graph)", "Final pipeline"]
    md = ["# Recommendation examples (test users)\n",
          "Generated by `scripts/run_analysis.py`. Users are picked by a fixed rule, hits are marked.\n"]
    tr = data.train_ratings
    for desc, u in picks:
        u = int(u)
        row = data.users.iloc[u]
        j = int(np.where(users == u)[0][0])
        tgt = set(data.targets[u].indices.tolist())
        liked = tr[(tr["user_idx"] == u) & (tr["rating"] >= 4)].sort_values("timestamp")["item_idx"].to_numpy()
        md.append(f"## {desc}: user_idx {u} ({row['gender']}, age group {row['age']}, {len(liked)} liked movies before the cutoff, {len(tgt)} test positives)\n")
        if len(liked):
            md.append("Last liked movies before the cutoff:\n\n" + fmt_items(liked[-8:][::-1], data) + "\n")
        for name in show:
            md.append(f"**{name}** ({len(tgt & set(lists[name][j, :10].tolist()))}/10 hits)\n\n" + fmt_items(lists[name][j, :10], data, tgt) + "\n")
    (tables / "recommendation_examples.md").write_text("\n".join(md), encoding="utf-8")

    # ---------------------------------------------------------------- error analysis
    rec = {name: per_user_recall(topk, users, data) for name, topk in lists.items()}
    st = stats.iloc[users].reset_index(drop=True)
    act = pd.cut(st["train_pos"], ACTIVITY_BINS, labels=ACTIVITY_LABELS)
    by_act = pd.DataFrame({"activity": act, **rec}).groupby("activity", observed=True).agg(["mean", "size"])
    rows = []
    for a in by_act.index:
        for name in rec:
            rows.append({"train_positives": a, "model": name, "recall@20": by_act.loc[a, (name, "mean")],
                         "users": int(by_act.loc[a, (name, "size")])})
    pd.DataFrame(rows).to_csv(tables / "error_by_activity.csv", index=False)

    # taste concentration is only meaningful with some history: warm users (>20 training positives)
    warmish = st["train_pos"] > cfg["split"]["user_sparse_max"]
    taste = pd.qcut(st.loc[warmish, "top_genre_share"], 3, labels=["broad", "medium", "narrow"])
    t_rows = []
    for name in rec:
        g = pd.Series(rec[name][warmish.to_numpy()]).groupby(taste.to_numpy(), observed=True).mean()
        for b, v in g.items():
            t_rows.append({"taste": b, "model": name, "recall@20": v, "users": int((taste == b).sum())})
    pd.DataFrame(t_rows).to_csv(tables / "error_by_taste.csv", index=False)

    hb = target_hit_rate_by_bucket(lists, users, data)
    hb.to_csv(tables / "error_by_item_popularity.csv", index=False)

    # paired comparisons on the same test users: per-user recall@20 averaged over the 3 seeds of
    # each trained model (seed noise would otherwise dominate), bootstrap CI + Wilcoxon test
    rng = np.random.default_rng(SEED)
    rec_seeds = dict(rec)
    for key in MAIN_ORDER:
        vals = []
        for sd in (42, 43, 44):
            try:
                vals.append(per_user_recall(load_topk(cfg, key, sd, "test")[1], users, data))
            except FileNotFoundError:
                pass
        rec_seeds[labels[key]] = np.mean(vals, axis=0)
    pairs = [("LightGCN", "MF-BPR"), ("GraphSAGE", "MF-BPR"), ("GAT", "MF-BPR"), ("GAT", "GraphSAGE"),
             ("Node2Vec", "MF-BPR"), ("Hybrid (id+content+graph)", "LightGCN"), ("Two-Tower", "MF-BPR"),
             ("ItemKNN", "MF-BPR"), ("Final pipeline", "Hybrid (id+content+graph)"), ("Final pipeline", "LightGCN"),
             ("Final pipeline", "Two-Tower")]
    groups_u = data.group[users]
    sig_rows = []
    for a, b in pairs:
        if a not in rec_seeds or b not in rec_seeds:
            continue
        for g in ("all", "warm", "sparse", "cold"):
            sel = np.ones(len(users), bool) if g == "all" else groups_u == g
            d = rec_seeds[a][sel] - rec_seeds[b][sel]
            boots = rng.choice(d, (2000, len(d)), replace=True).mean(1)
            p = wilcoxon(d[d != 0]).pvalue if (d != 0).sum() > 10 else np.nan
            sig_rows.append({"model_a": a, "model_b": b, "users": g, "n": int(sel.sum()),
                             "mean_recall@20_a": rec_seeds[a][sel].mean(), "mean_recall@20_b": rec_seeds[b][sel].mean(),
                             "diff": d.mean(), "ci95_low": np.percentile(boots, 2.5), "ci95_high": np.percentile(boots, 97.5),
                             "wilcoxon_p": p, "a_better_users": int((d > 0).sum()), "b_better_users": int((d < 0).sum())})
    pd.DataFrame(sig_rows).to_csv(tables / "significance_tests.csv", index=False)
    print(pd.DataFrame(sig_rows).query("users == 'all'").round(4).to_string(index=False))

    table, corr = graph_vs_mf(rec["GraphSAGE"], rec["MF-BPR"], users, data)
    table.to_csv(tables / "error_graph_neighbourhood.csv", index=False)
    save_json(corr, tables / "error_graph_neighbourhood_corr.json")
    print(table.to_string(index=False), corr)

    # retrieval vs ranking failures of the final pipeline
    retr = FaissRetriever(pipe.retriever.item_vectors(), "flat")
    cand_items, _ = retr.search(pipe.U[users], pipe.n_candidates, data.history, users)
    bucket = item_popularity_bucket(data)
    fail_rows = []
    for j, u in enumerate(users):
        tgt = data.targets[u].indices
        in_cand = np.isin(tgt, cand_items[j])
        in_top = np.isin(tgt, lists["Final pipeline"][j, :20])
        for i, c, t in zip(tgt, in_cand, in_top):
            fail_rows.append({"user_group": data.group[u], "item_bucket": bucket[i],
                              "outcome": "hit (top-20)" if t else ("ranked below 20" if c else "not retrieved")})
    fails = pd.DataFrame(fail_rows)
    br = fails.groupby(["item_bucket", "outcome"]).size().unstack(fill_value=0)
    br = br.div(br.sum(axis=1), axis=0).assign(targets=fails.groupby("item_bucket").size())
    br.to_csv(tables / "error_pipeline_breakdown.csv")
    bg = fails.groupby(["user_group", "outcome"]).size().unstack(fill_value=0)
    bg.div(bg.sum(axis=1), axis=0).assign(targets=fails.groupby("user_group").size()).to_csv(tables / "error_pipeline_breakdown_by_group.csv")
    print(br)

    # concrete failure cases, again chosen by rule
    md = ["# Error examples (test users)\n", "Generated by `scripts/run_analysis.py`. Each case is selected by a rule stated in its heading.\n"]
    rec_df = pd.DataFrame(rec).assign(user=users, group=data.group[users], n_tgt=n_tgt, tp=st["train_pos"].to_numpy(),
                                      top_share=st["top_genre_share"].to_numpy(), deg=st["mean_item_degree"].to_numpy())
    zero_all = rec_df[(rec_df[list(rec)].max(axis=1) == 0) & (rec_df["n_tgt"] >= 10)]
    cases = []
    if len(zero_all):
        cases.append(("no model gets a single hit in the top 20 (most test positives among such users)", zero_all.sort_values("n_tgt").iloc[-1]))
    loser = rec_df[(rec_df["group"] != "cold") & (rec_df["n_tgt"] >= 10)].assign(d=lambda d: d["GraphSAGE"] - d["MF-BPR"])
    cases.append(("largest GraphSAGE loss against MF-BPR", loser.sort_values("d").iloc[0]))
    narrow = rec_df[(rec_df["group"] == "warm") & (rec_df["n_tgt"] >= 10)].sort_values("top_share")
    cases.append(("narrowest taste among warm users (largest share of likes in one genre)", narrow.iloc[-1]))
    heavy = rec_df[rec_df["n_tgt"] >= 10].sort_values("tp")
    cases.append(("most active user (most training positives)", heavy.iloc[-1]))
    g = data.genre_matrix()
    for desc, r in cases:
        u = int(r["user"])
        j = int(np.where(users == u)[0][0])
        tgt = data.targets[u].indices
        liked = tr[(tr["user_idx"] == u) & (tr["rating"] >= 4)]["item_idx"].to_numpy()
        pop = data.item_popularity()
        md.append(f"## {desc}: user_idx {u} ({data.group[u]})\n")
        if len(liked):
            md.append(f"- liked before cutoff: {len(liked)} movies, their mean popularity is {pop[liked].mean():.0f} training positives")
        else:
            md.append("- no positive ratings before the cutoff")
        md.append(f"- test positives: {len(tgt)}, mean popularity {pop[tgt].mean():.0f}, "
                  f"{(bucket[tgt] == 'tail (bottom 40%)').mean():.0%} tail items, {(bucket[tgt] == 'cold (0 train positives)').mean():.0%} cold items")
        top_g = np.array(GENRES)[np.argsort(-g[tgt].sum(0))[:3]]
        md.append(f"- most common genres in the test positives: {', '.join(top_g)}")
        if len(liked):
            top_l = np.array(GENRES)[np.argsort(-g[liked].sum(0))[:3]]
            md.append(f"- most common genres in the training likes: {', '.join(top_l)}")
        md.append("- recall@20: " + ", ".join(f"{k} {r[k]:.2f}" for k in ["Popularity", "MF-BPR", "GraphSAGE", "Hybrid (id+content+graph)", "Final pipeline"]))
        in_cand = np.isin(tgt, cand_items[j]).mean()
        md.append(f"- final pipeline: {in_cand:.0%} of the test positives were in the {pipe.n_candidates} retrieved candidates\n")
        md.append("Final pipeline top 10:\n\n" + fmt_items(lists["Final pipeline"][j, :10], data, set(tgt.tolist())) + "\n")
        md.append("A few of the test positives:\n\n" + fmt_items(tgt[:5], data) + "\n")
    # most-missed popular-ish target items and long-tail targets missed by everyone
    t_all = data.targets[users].tocoo()
    found_any = np.zeros(len(t_all.col), dtype=bool)
    for topk in lists.values():
        sets = [set(row[:20].tolist()) for row in topk]
        found_any |= np.array([c in sets[r] for r, c in zip(t_all.row, t_all.col)])
    miss = pd.DataFrame({"item": t_all.col, "found": found_any})
    tail_missed = miss[(~miss["found"]) & (bucket[miss["item"]] == "tail (bottom 40%)")]["item"].value_counts().head(5)
    md.append("## long-tail test positives that no model ranks in any user's top 20 (most frequent)\n")
    md.append(fmt_items(tail_missed.index.to_numpy(), data) + "\n")
    md.append(f"Across all test users {(~miss['found']).mean():.0%} of the test positives are not in the top 20 of any model.\n")
    (tables / "error_examples.md").write_text("\n".join(md), encoding="utf-8")

    # ---------------------------------------------------------------- onboarding: how much do the revealed
    # ratings move a new user's vector in the inductive GNNs? (explains the flat LightGCN curve)
    ob_users, _, _, revealed = data.onboarding("test", 10, 10, 15)
    ob_pairs = np.array([(u, i) for u, r in zip(ob_users, revealed) for i in r])
    shift_rows = []
    for key in ("graphsage", "gat", "lightgcn", "hybrid"):
        m = load_main_model(cfg, data, key, SEED, device)
        m.refresh()
        U0 = m.user_vectors()[ob_users]
        m.refresh_with_edges(ob_pairs)
        U10 = m.user_vectors()[ob_users]
        cos = (U0 * U10).sum(1) / (np.linalg.norm(U0, axis=1) * np.linalg.norm(U10, axis=1))
        shift_rows.append({"model": labels[key], "users": len(ob_users),
                           "mean_cosine_k0_vs_k10": float(cos.mean()),
                           "mean_relative_change": float((np.linalg.norm(U10 - U0, axis=1) / np.linalg.norm(U0, axis=1)).mean())})
        del m
    pd.DataFrame(shift_rows).to_csv(tables / "onboarding_vector_shift.csv", index=False)
    print(pd.DataFrame(shift_rows).round(3).to_string(index=False))

    # ---------------------------------------------------------------- embeddings
    purity_rows = []
    pop = data.item_popularity()
    sel = np.where(pop >= 20)[0]
    gm = data.genre_matrix()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    focus = ["Horror", "Children's", "Documentary", "Film-Noir", "War", "Western"]
    gidx = {gname: GENRES.index(gname) for gname in focus}
    lab = np.full(len(sel), "other", dtype=object)
    for gname in focus[::-1]:
        lab[gm[sel, gidx[gname]] > 0] = gname
    for key in ("mf", "two_tower", "node2vec", "graphsage", "gat", "lightgcn", "hybrid"):
        m = load_main_model(cfg, data, key, SEED, device)
        V = m.item_vectors()[sel]
        Vn = V / np.linalg.norm(V, axis=1, keepdims=True).clip(1e-12)
        sim = Vn @ Vn.T
        np.fill_diagonal(sim, -np.inf)
        nn10 = np.argsort(-sim, axis=1)[:, :10]
        share = ((gm[sel][nn10] * gm[sel][:, None, :]).sum(-1) > 0).mean()
        rand = ((gm[sel][np.random.default_rng(0).integers(0, len(sel), (len(sel), 10))] * gm[sel][:, None, :]).sum(-1) > 0).mean()
        purity_rows.append({"model": labels[key], "nn10_share_sharing_a_genre": float(share), "random_baseline": float(rand),
                            "items": len(sel)})
        if key in ("mf", "graphsage", "hybrid"):
            ax = axes[["mf", "graphsage", "hybrid"].index(key)]
            xy = TSNE(n_components=2, random_state=0, init="pca", perplexity=30).fit_transform(Vn)
            ax.scatter(xy[lab == "other", 0], xy[lab == "other", 1], s=3, c="#dddddd")
            for gname, c in zip(focus, ["#d62728", "#ff7f0e", "#2ca02c", "#000000", "#9467bd", "#8c564b"]):
                mk = lab == gname
                ax.scatter(xy[mk, 0], xy[mk, 1], s=5, c=c, label=gname)
            ax.set_title(f"{labels[key]} item embeddings (t-SNE)")
            ax.set_xticks([])
            ax.set_yticks([])
        del m
    axes[0].legend(markerscale=3, loc="lower left", fontsize=8)
    save(fig, "embedding_tsne.png", cfg["paths"]["results_dir"])
    pd.DataFrame(purity_rows).to_csv(tables / "embedding_genre_purity.csv", index=False)
    print(pd.DataFrame(purity_rows).to_string(index=False))


if __name__ == "__main__":
    main()
