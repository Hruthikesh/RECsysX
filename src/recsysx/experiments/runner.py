"""Train one model, evaluate it on validation and test users, save everything.

All experiment suites go through ``run_model`` so every number in results/ comes from
the same evaluation code, the same users and the same seen-item filter.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..config import project_path
from ..data.dataset import RecData
from ..evaluation import Evaluator, flatten_result
from ..models import build_recommender
from ..tracking import Tracker
from ..utils import Timer, get_device, get_logger, save_json, set_seed

log = get_logger()


def make_evaluator(cfg: dict, data: RecData, role: str, device, **kwargs) -> Evaluator:
    return Evaluator(data, role, ks=cfg["eval"]["ks"], device=device, batch_size=cfg["eval"]["batch_size"], **kwargs)


def run_model(cfg: dict, model_cfg: dict, data: RecData, run_name: str, group: str, seed: int = 42,
              label: str | None = None, save: bool = True, roles=("val", "test"), extra: dict | None = None):
    """Returns (record, recommender, {role: topk})."""
    device = get_device(cfg.get("device", "auto"))
    set_seed(seed)
    model_cfg = copy.deepcopy(model_cfg)
    model_cfg["seed"] = seed
    if cfg.get("max_epochs") and "epochs" in model_cfg:  # quick mode for smoke tests
        model_cfg["epochs"] = min(int(model_cfg["epochs"]), int(cfg["max_epochs"]))
    label = label or model_cfg.get("label") or model_cfg["model"]
    tracker = Tracker(cfg, run_name, group, config={"model_cfg": model_cfg, "seed": seed, "data": data.tag,
                                                     "label": label, **(extra or {})})
    val_eval = make_evaluator(cfg, data, "val", device)
    rec = build_recommender(model_cfg, device)
    log.info("[%s] training %s (%s, seed %d, data=%s)", group, run_name, label, seed, data.tag)
    with Timer() as t_fit:
        rec.fit(data, val_eval, tracker)
    info = getattr(rec, "fit_info", {}) or {}

    record = {"run": run_name, "group": group, "model": label, "seed": seed, "data": data.tag,
              "fit_time_s": t_fit.elapsed, "train_time_s": info.get("train_time_s", t_fit.elapsed),
              "best_epoch": info.get("best_epoch"), "epochs_run": info.get("epochs_run"),
              "n_params": rec.num_parameters(), "peak_gpu_mem_mb": info.get("peak_gpu_mem_mb")}
    record.update(extra or {})
    topks, users = {}, {}
    for role in roles:
        ev = val_eval if role == "val" else make_evaluator(cfg, data, role, device)
        users[role] = ev.users
        res, topk = ev.evaluate(rec.score)
        topks[role] = topk
        record.update(flatten_result(res, prefix=f"{role}/"))
    tracker.set_summary({k: v for k, v in record.items() if isinstance(v, (int, float, str)) or v is None})
    tracker.finish()

    if save:
        out = project_path(cfg["paths"]["checkpoint_dir"]) / group / run_name
        out.mkdir(parents=True, exist_ok=True)
        rec.save(out)
        np.savez_compressed(out / "topk.npz", **topks, **{f"{r}_users": u for r, u in users.items()})
        save_json({"record": record, "model_cfg": model_cfg, "fit_history": info.get("history", [])},
                  out / "run.json")
    torch.cuda.empty_cache()
    return record, rec, topks


def upsert_csv(records: list[dict] | pd.DataFrame, path: str | Path, key: str = "run") -> pd.DataFrame:
    """Write records to a CSV, replacing rows that have the same key (re-runs overwrite)."""
    path = project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = records if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
    if path.exists():
        old = pd.read_csv(path)
        if key in old.columns and key in new.columns:
            old = old[~old[key].isin(new[key])]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)
    return new
