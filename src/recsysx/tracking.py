"""Experiment logging: always to a local JSON file, optionally to Weights & Biases.

The local file in results/logs/ is the source of truth for every table in the repo, so
nothing depends on having a W&B account. W&B is switched on with tracking.wandb=true
(or --wandb on the scripts). WANDB_MODE=offline works without logging in.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from .config import project_path
from .utils import save_json


class Tracker:
    def __init__(self, cfg: dict, run_name: str, group: str | None = None, config: dict | None = None):
        self.run_name = run_name
        self.group = group
        self.config = config or {}
        self.history: list[dict] = []
        self.summary: dict = {}
        self.started = time.time()
        self.log_dir = project_path(cfg["paths"]["results_dir"]) / "logs"
        self._wandb = None
        tcfg = cfg.get("tracking", {})
        if tcfg.get("wandb"):
            try:
                import wandb

                kwargs = dict(project=tcfg.get("project", "recsysx"), entity=tcfg.get("entity"), name=run_name,
                              group=group, config=self.config, reinit=True,
                              dir=str(project_path(".")))
                if tcfg.get("mode"):
                    kwargs["mode"] = tcfg["mode"]
                self._wandb = wandb.init(**kwargs)
            except Exception as exc:  # tracking must never kill an experiment
                print(f"[tracking] wandb disabled: {exc}")
                self._wandb = None

    def log(self, metrics: dict, step: int | None = None) -> None:
        row = dict(metrics)
        if step is not None:
            row["step"] = step
        self.history.append(row)
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)

    def set_summary(self, metrics: dict) -> None:
        self.summary.update(metrics)
        if self._wandb is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float, str)) or v is None:
                    self._wandb.summary[k] = v

    def finish(self) -> Path:
        out = self.log_dir / (self.group or "misc") / f"{self.run_name}.json"
        save_json({"run_name": self.run_name, "group": self.group, "config": self.config,
                   "summary": self.summary, "history": self.history,
                   "wall_time_s": time.time() - self.started,
                   "wandb_run": getattr(self._wandb, "id", None),
                   "wandb_mode": os.environ.get("WANDB_MODE")}, out)
        if self._wandb is not None:
            self._wandb.finish()
        return out
