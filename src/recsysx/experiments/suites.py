"""Experiment suites. Each one writes a CSV into results/tables/.

tune        validation-only hyper-parameter search -> configs/selected.yaml
main        every model with the selected config, 3 seeds, val + test
gnn_ablation  a few GraphSAGE/GAT design choices evaluated on val + test
hybrid      id / content / graph ablation grid
sparsity    100/50/20/10% of the training positives
item_cold   simulated new items (10% of the catalog removed from training)
onboarding  new users with k revealed ratings, uses the main-suite checkpoints
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from ..config import CONFIG_DIR, deep_merge, load_config, project_path, read_yaml
from ..data.dataset import RecData, to_csr
from ..utils import get_device, get_logger
from .runner import run_model, upsert_csv

log = get_logger()
SELECTED_PATH = CONFIG_DIR / "selected.yaml"

# key -> (config file, label, fixed overrides)
MODEL_SPECS = {
    "popularity": ("popularity", "Popularity", {"variant": "global"}),
    "popularity_recent": ("popularity", "Popularity (recent)", {"variant": "recent"}),
    "popularity_demo": ("popularity", "Popularity (demographic)", {"variant": "demographic"}),
    "itemknn": ("itemknn", "ItemKNN", {}),
    "mf": ("mf", "MF-BPR", {}),
    "two_tower": ("two_tower", "Two-Tower", {}),
    "node2vec": ("node2vec", "Node2Vec", {}),
    "graphsage": ("graphsage", "GraphSAGE", {}),
    "gat": ("gat", "GAT", {}),
    "lightgcn": ("lightgcn", "LightGCN", {}),
    "hybrid": ("hybrid", "Hybrid (id+content+graph)", {}),
}
TRAINED = ["mf", "two_tower", "node2vec", "graphsage", "gat", "lightgcn", "hybrid"]
DETERMINISTIC = ["popularity", "popularity_recent", "popularity_demo", "itemknn"]
MAIN_ORDER = DETERMINISTIC + TRAINED
SEEDS = [42, 43, 44]

# Coordinate-descent style search: every stage tries a few changes on top of the best
# config so far. Kept small on purpose, the goal is a reasonable config per model, not
# squeezing out the last 0.1%.
TUNING_STAGES = {
    "popularity_recent": [[{"recent_days": d} for d in (7, 30, 90)]],
    "itemknn": [[{"top_k_neighbors": k, "shrink": s} for k in (50, 200, 1000) for s in (0.0, 10.0, 100.0)]],
    "mf": [
        [{"lr": lr, "l2_reg": l2} for lr in (1e-3, 5e-3) for l2 in (1e-5, 1e-4, 1e-3)],
        [{"num_negatives": 4}, {"user_id_dropout": 0.0}],
    ],
    "two_tower": [
        [{"temperature": t} for t in (0.05, 0.1, 0.2)],
        [{"log_q_correction": False}],
        [{"user_id_dropout": 0.0, "item_id_dropout": 0.0}, {"user_id_dropout": 0.3, "item_id_dropout": 0.3}],
    ],
    "node2vec": [
        [{"p": 1.0, "q": 1.0}, {"p": 0.25, "q": 1.0}, {"p": 4.0, "q": 1.0}],
        [{"walk_length": 40, "context_size": 10}, {"context_size": 5}],
        [{"score": "cosine"}, {"score": "hadamard_lr"}],
    ],
    "graphsage": [
        [{"num_layers": n} for n in (1, 2, 3)],
        [{"l2_reg": 1e-3}, {"dropout": 0.3}, {"lr": 5e-3}],
        [{"aggr": "max"}],
        [{"jk": "last"}, {"jk": "cat"}],
        [{"embedding_dim": 128}],
    ],
    "gat": [
        [{"num_layers": n} for n in (1, 2, 3)],
        [{"l2_reg": 1e-3}, {"dropout": 0.3}, {"lr": 5e-3}],
        [{"heads": 1}, {"heads": 8}],
        [{"jk": "last"}, {"jk": "cat"}],
        [{"embedding_dim": 128}],
    ],
    "lightgcn": [
        [{"num_layers": n} for n in (1, 2, 3)],
        [{"lr": 5e-3}, {"l2_reg": 1e-3}, {"l2_reg": 1e-5}],
        [{"embedding_dim": 128}],
    ],
    "hybrid": [
        None,  # stage 1 is built from the tuned graphsage / gat configs, see _hybrid_stage1
        [{"user_id_dropout": 0.3, "item_id_dropout": 0.3}, {"item_id_dropout": 0.5}],
    ],
}
TUNING_ORDER = ["popularity_recent", "itemknn", "mf", "two_tower", "node2vec", "graphsage", "gat", "lightgcn", "hybrid"]
GNN_ARCH_KEYS = ["conv", "num_layers", "aggr", "heads", "jk", "embedding_dim", "l2_reg", "lr", "dropout"]


# ---------------------------------------------------------------------- configs

def table_path(cfg: dict, name: str) -> Path:
    return project_path(cfg["paths"]["results_dir"]) / "tables" / name


def selected_path(cfg: dict | None = None) -> Path:
    rel = (cfg or {}).get("paths", {}).get("selected_config")
    return project_path(rel) if rel else SELECTED_PATH


def load_selected(cfg: dict | None = None) -> dict:
    path = selected_path(cfg)
    return read_yaml(path) if path.exists() else {}


def base_model_cfg(key: str) -> dict:
    cfg_name, label, fixed = MODEL_SPECS[key]
    mcfg = deep_merge(load_config(cfg_name)["model_cfg"], fixed)
    mcfg["label"] = label
    return mcfg


def selected_model_cfg(key: str, cfg: dict | None = None) -> dict:
    """Model config with the tuned overrides from configs/selected.yaml (if it exists)."""
    if key not in MODEL_SPECS:  # a plain config name / path, e.g. from scripts/train.py
        return load_config(key)["model_cfg"]
    return deep_merge(base_model_cfg(key), load_selected(cfg).get(key, {}))


def _run_name(key: str, overrides: dict) -> str:
    if not overrides:
        return f"{key}__default"
    parts = [f"{k}={v}" for k, v in sorted(overrides.items())]
    return f"{key}__" + ",".join(parts)


# ---------------------------------------------------------------------- tune

def _hybrid_stage1(selected: dict) -> list[dict]:
    out = []
    for arch in ("graphsage", "gat", "lightgcn"):
        tuned = deep_merge(base_model_cfg(arch), selected.get(arch, {}))
        out.append({k: tuned[k] for k in GNN_ARCH_KEYS if k in tuned})
    return out


def tune(cfg: dict, data: RecData, models: list[str] | None = None, seed: int = 42) -> pd.DataFrame:
    selected = load_selected(cfg)
    records = []
    for key in models or TUNING_ORDER:
        best_over: dict = {}
        best_score = -np.inf
        scores: dict[str, float] = {}
        seen_cfgs: dict[str, str] = {}
        for stage_idx, stage in enumerate(TUNING_STAGES[key]):
            if stage is None:
                stage = _hybrid_stage1(selected)
            if stage_idx == 0:
                candidates = list(stage)
                if key not in ("itemknn", "popularity_recent", "hybrid"):
                    candidates = [{}] + candidates  # the default config is always a candidate
            else:
                candidates = [{**best_over, **c} for c in stage]
            for over in candidates:
                mcfg = deep_merge(base_model_cfg(key), over)
                # identical configs under different names (e.g. default vs num_layers=2) run once
                name = seen_cfgs.setdefault(json.dumps(mcfg, sort_keys=True), _run_name(key, over))
                if name not in scores:
                    rec, _, _ = run_model(cfg, mcfg, data, name, group="tuning", seed=seed, roles=("val",),
                                          save=False, extra={"stage": stage_idx + 1, "overrides": json.dumps(over),
                                                             "tuned_model": key})
                    records.append(rec)
                    upsert_csv([rec], table_path(cfg, "tuning_results.csv"))
                    scores[name] = rec["val/recall@20"]
                if scores[name] > best_score + 1e-6:
                    best_score, best_over = scores[name], over
            log.info("[tune] %s stage %d best %s val recall@20=%.4f", key, stage_idx + 1, best_over, best_score)
        selected[key] = best_over
        selected_path(cfg).write_text(
            "# Written by `python scripts/run_experiments.py tune`. Overrides on top of configs/models/*.yaml,\n"
            "# chosen by validation recall@20 only (see results/tables/tuning_results.csv).\n"
            + yaml.safe_dump(selected, sort_keys=True))
    return pd.DataFrame(records)


# ---------------------------------------------------------------------- main comparison

def main_comparison(cfg: dict, data: RecData, models: list[str] | None = None, seeds=SEEDS) -> pd.DataFrame:
    records = []
    for key in models or MAIN_ORDER:
        mcfg = selected_model_cfg(key, cfg)
        for seed in (seeds if key in TRAINED else seeds[:1]):
            rec, _, _ = run_model(cfg, mcfg, data, f"{key}_s{seed}", group="main", seed=seed)
            rec["key"] = key
            records.append(rec)
            upsert_csv([rec], table_path(cfg, "main_runs.csv"))
    return pd.DataFrame(records)


# ---------------------------------------------------------------------- GNN design ablation

# deeper GNNs sit on a long plateau before they improve, the depth variants get patience 50 so
# early stopping is not the reason they lose
GNN_ABLATIONS = {
    "graphsage": [
        ("target edges kept in message passing", {"remove_target_edges": False}),
        ("unknown-user row trained with edges (no cold simulation)", {"cold_simulation": False}),
        ("no unknown-user embedding (id dropout 0)", {"user_id_dropout": 0.0}),
        ("2 layers (patience 50)", {"num_layers": 2, "patience": 50}),
        ("3 layers (patience 50)", {"num_layers": 3, "patience": 50}),
        ("max aggregation", {"aggr": "max"}),
        ("jk=last", {"jk": "last"}),
    ],
    "gat": [
        ("target edges kept in message passing", {"remove_target_edges": False}),
        ("unknown-user row trained with edges (no cold simulation)", {"cold_simulation": False}),
        ("2 layers (patience 50)", {"num_layers": 2, "patience": 50}),
        ("3 layers (patience 50)", {"num_layers": 3, "patience": 50}),
        ("1 head", {"heads": 1}),
    ],
    "lightgcn": [
        ("target edges kept in message passing", {"remove_target_edges": False}),
        ("unknown-user row trained with edges (no cold simulation)", {"cold_simulation": False}),
        ("no unknown-user embedding (id dropout 0)", {"user_id_dropout": 0.0}),
        ("2 layers (patience 50)", {"num_layers": 2, "patience": 50}),
        ("3 layers (patience 50)", {"num_layers": 3, "patience": 50}),
    ],
}


def gnn_ablation(cfg: dict, data: RecData, seed: int = 42) -> pd.DataFrame:
    records = []
    for key, variants in GNN_ABLATIONS.items():
        base = selected_model_cfg(key, cfg)
        for desc, over in variants:
            if all(base.get(k) == v for k, v in over.items()):
                continue  # identical to the selected config, the main run covers it
            name = f"{key}_abl__" + ",".join(f"{k}={v}" for k, v in over.items())
            rec, _, _ = run_model(cfg, deep_merge(base, over), data, name, group="gnn_ablation", seed=seed,
                                  extra={"base": key, "variant": desc}, save=False)
            records.append(rec)
            upsert_csv([rec], table_path(cfg, "gnn_ablation_runs.csv"))
    return pd.DataFrame(records)


# ---------------------------------------------------------------------- hybrid ablation

def hybrid_variants(cfg: dict | None = None) -> list[tuple[str, dict]]:
    h = selected_model_cfg("hybrid", cfg)
    layers = h["num_layers"]
    return [
        ("collaborative only (id, no graph)", {"use_id": True, "use_content": False, "num_layers": 0, "item_id_dropout": 0.0}),
        ("content only (no id, no graph)", {"use_id": False, "use_content": True, "num_layers": 0}),
        ("id + content (no graph)", {"use_id": True, "use_content": True, "num_layers": 0}),
        ("graph only (id + graph)", {"use_id": True, "use_content": False, "num_layers": layers, "item_id_dropout": 0.0}),
        ("content + graph (no id)", {"use_id": False, "use_content": True, "num_layers": layers}),
        ("hybrid (id + content + graph)", {}),
    ]


def hybrid_ablation(cfg: dict, data: RecData, seeds=SEEDS) -> pd.DataFrame:
    records = []
    base = selected_model_cfg("hybrid", cfg)
    for desc, over in hybrid_variants(cfg):
        for seed in seeds:
            name = "hyb__" + desc.split(" (")[0].replace(" ", "_").replace("+", "") + f"_s{seed}"
            rec, _, _ = run_model(cfg, deep_merge(base, over), data, name, group="hybrid", seed=seed,
                                  extra={"variant": desc}, label=desc)
            records.append(rec)
            upsert_csv([rec], table_path(cfg, "hybrid_runs.csv"))
    return pd.DataFrame(records)


# ---------------------------------------------------------------------- sparsity

SPARSITY_MODELS = ["popularity", "itemknn", "mf", "two_tower", "node2vec", "graphsage", "gat", "lightgcn", "hybrid"]


def sparsity(cfg: dict, data: RecData, fractions=(0.5, 0.2, 0.1), seeds=SEEDS, models=None) -> pd.DataFrame:
    records = []
    for frac in fractions:
        for seed in seeds:
            sub = data.with_train_fraction(frac, seed=seed)
            for key in models or SPARSITY_MODELS:
                rec, _, _ = run_model(cfg, selected_model_cfg(key, cfg), sub, f"{key}_frac{frac:g}_s{seed}",
                                      group="sparsity", seed=seed, extra={"key": key, "train_fraction": frac},
                                      save=False)
                records.append(rec)
                upsert_csv([rec], table_path(cfg, "sparsity_runs.csv"))
    return pd.DataFrame(records)


# ---------------------------------------------------------------------- item cold-start

ITEM_COLD_MODELS = ["popularity", "itemknn", "mf", "two_tower", "node2vec", "graphsage", "gat", "lightgcn", "hybrid"]


def item_cold(cfg: dict, data: RecData, fraction: float = 0.1, seeds=(42,), models=None) -> pd.DataFrame:
    from ..evaluation import Evaluator

    device = get_device(cfg.get("device", "auto"))
    records = []
    extra_variants = [("hybrid_content_graph", "hybrid", {"use_id": False}),
                      ("hybrid_content_only", "hybrid", {"use_id": False, "num_layers": 0})]
    for seed in seeds:
        cold_data = data.with_cold_items(fraction, seed=seed)
        cold = np.zeros(data.n_items, dtype=bool)
        cold[cold_data.notes["cold_items"]] = True
        jobs = [(k, k, {}) for k in (models or ITEM_COLD_MODELS)] + extra_variants
        for run_key, key, over in jobs:
            mcfg = deep_merge(selected_model_cfg(key, cfg), over)
            if over:
                mcfg["label"] = {"hybrid_content_graph": "Content + graph (no id)",
                                 "hybrid_content_only": "Content only"}[run_key]
            rec, model, topks = run_model(cfg, mcfg, cold_data, f"{run_key}_colditems_s{seed}", group="item_cold",
                                          seed=seed, extra={"key": run_key}, save=False)
            ev = Evaluator(cold_data, "test", ks=cfg["eval"]["ks"], device=device)
            # (a) cold items competing with the full catalog
            full = ev.topk(model.score, k=100)
            for kk in (20, 100):
                hits, n_t = ev.hits(full[:, :kk], item_subset=cold)
                valid = n_t > 0
                rec[f"coldtgt/recall@{kk}_full_catalog"] = float((hits.sum(1)[valid] / n_t[valid]).mean())
            rec["coldtgt/n_users"] = int(valid.sum())
            share = cold[full[:, :20]].mean()
            rec["cold_item_share_in_top20"] = float(share)
            # (b) ranking only among the new items ("which new movies to show this user")
            only = ev.topk(model.score, k=20, item_mask=cold)
            m = ev.metrics_from_topk(only, item_subset=cold, by_group=False)["all"]
            for name in ("recall@10", "recall@20", "ndcg@10", "ndcg@20"):
                rec[f"coldtgt/{name}_among_new"] = m[name]
            records.append(rec)
            upsert_csv([rec], table_path(cfg, "item_cold_runs.csv"))
            torch.cuda.empty_cache()
    return pd.DataFrame(records)


# ---------------------------------------------------------------------- onboarding (new users)

def fold_in(item_vec: torch.Tensor, init: torch.Tensor, revealed: list[np.ndarray], steps: int = 100,
            lr: float = 0.05, l2: float = 1e-3, n_neg: int = 20, seed: int = 0) -> torch.Tensor:
    """Fit a free user vector per new user against frozen item vectors with BPR.
    This is the usual way to get an embedding for a new user in MF without retraining."""
    g = torch.Generator(device=item_vec.device)
    g.manual_seed(seed)
    U = init.clone().requires_grad_(True)
    rows = torch.as_tensor(np.concatenate([np.full(len(r), j) for j, r in enumerate(revealed)]),
                           device=item_vec.device, dtype=torch.long)
    cols = torch.as_tensor(np.concatenate(revealed), device=item_vec.device, dtype=torch.long)
    if len(rows) == 0:
        return init
    opt = torch.optim.Adam([U], lr=lr)
    for _ in range(steps):
        neg = torch.randint(0, item_vec.shape[0], (len(rows), n_neg), device=item_vec.device, generator=g)
        u = U[rows]
        s_pos = (u * item_vec[cols]).sum(1, keepdim=True)
        s_neg = torch.einsum("bd,bnd->bn", u, item_vec[neg])
        loss = torch.nn.functional.softplus(s_neg - s_pos).mean() + l2 * U.pow(2).sum() / len(U)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return U.detach()


def onboarding(cfg: dict, data: RecData, ks_reveal=(0, 1, 3, 5, 10), seed: int = 42, min_eval_pos: int = 15,
               models=None) -> pd.DataFrame:
    from ..evaluation import Evaluator
    from ..models import build_recommender

    device = get_device(cfg.get("device", "auto"))
    records = []
    keys = models or ["popularity", "popularity_demo", "itemknn", "mf", "two_tower", "node2vec", "graphsage",
                      "gat", "lightgcn", "hybrid"]
    for key in keys:
        mcfg = selected_model_cfg(key, cfg)
        ck = project_path(cfg["paths"]["checkpoint_dir"]) / "main" / f"{key}_s{seed}"
        rec_model = build_recommender(mcfg, device)
        if key.startswith("popularity"):
            rec_model.fit(data)
        else:
            rec_model.load(ck, data)
        for k in ks_reveal:
            users, history, targets, revealed = data.onboarding("test", k, max(ks_reveal), min_eval_pos)
            ev = Evaluator(data, "test", ks=cfg["eval"]["ks"], device=device, users=users, history=history,
                           targets=targets)
            method = "as-is"
            if key == "itemknn":
                # only the k revealed movies count as the user's profile (``history`` is the fixed
                # seen-item filter, using it here would hand ItemKNN all 10 onboarding ratings)
                rows = np.concatenate([np.full(len(r), u) for u, r in zip(users, revealed)]) if k else np.array([], int)
                cols = np.concatenate(revealed) if k else np.array([], int)
                profile = to_csr(rows.astype(np.int64), cols.astype(np.int64), (data.n_users, data.n_items))
                score_fn = lambda u, profile=profile: rec_model.score_with_history(profile[u.cpu().numpy()])
                method = "item-item scores from revealed items"
            elif mcfg["model"] == "gnn":
                pairs = np.array([(u, i) for u, r in zip(users, revealed) for i in r], dtype=np.int64).reshape(-1, 2)
                rec_model.refresh()  # reset to training-graph embeddings
                if len(pairs):
                    rec_model.refresh_with_edges(pairs)
                score_fn = rec_model.score
                method = "inductive: revealed edges added to the graph"
            elif rec_model.is_embedding_model:
                U = rec_model.user_vec.clone()
                if k > 0:
                    idx = torch.as_tensor(users, device=device)
                    U[idx] = fold_in(rec_model.item_vec, U[idx], revealed, seed=seed)
                item_vec = rec_model.item_vec
                score_fn = lambda u, U=U: U[u] @ item_vec.T
                method = "fold-in: user vector fitted on revealed items" if k > 0 else "cold-user vector"
            else:
                score_fn = rec_model.score
            res, _ = ev.evaluate(score_fn, beyond=False)
            row = {"model": mcfg.get("label", key), "key": key, "k_revealed": k, "method": method,
                   "n_users": res["all"]["n_users"], "seed": seed}
            row.update({m: res["all"][m] for m in ("recall@10", "recall@20", "ndcg@10", "ndcg@20", "mrr")})
            records.append(row)
            log.info("[onboarding] %s k=%d recall@20=%.4f (%d users)", key, k, row["recall@20"], row["n_users"])
        if mcfg["model"] == "gnn":
            rec_model.refresh()
    df = pd.DataFrame(records)
    upsert_csv(df.assign(run=df["key"] + "_k" + df["k_revealed"].astype(str)), table_path(cfg, "onboarding_runs.csv"))
    return df


SUITES = {
    "tune": tune,
    "main": main_comparison,
    "gnn_ablation": gnn_ablation,
    "hybrid": hybrid_ablation,
    "sparsity": sparsity,
    "item_cold": item_cold,
    "onboarding": onboarding,
}

__all__ = ["SUITES", "MODEL_SPECS", "selected_model_cfg", "load_selected"]
