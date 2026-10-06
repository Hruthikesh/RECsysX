"""Training loop shared by MF, two-tower and the GNN models.

Every module implements
    batch_loss(edge_ids, users, items, sampler) -> scalar loss
    compute_embeddings() -> (user_vectors, item_vectors)
One epoch is one pass over the training positives in random order. After each epoch
the model is evaluated on the validation users (full ranking) and the best epoch by
validation recall@20 is kept (early stopping with patience).
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..data.dataset import RecData
from ..data.sampling import NegativeSampler
from ..utils import Timer, count_parameters, get_logger
from .base import EmbeddingRecommender

log = get_logger()


def train_embedding_module(module: nn.Module, data: RecData, mcfg: dict, evaluator, device: torch.device,
                           tracker=None, seed: int = 0, name: str = "model") -> dict:
    opt = torch.optim.Adam(module.parameters(), lr=mcfg["lr"], weight_decay=mcfg.get("weight_decay", 0.0))
    pairs = torch.as_tensor(data.train_pos, device=device, dtype=torch.long)
    sampler = NegativeSampler(data.train_pos, data.n_items, device)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    primary = mcfg.get("primary_metric", "recall@20")
    epochs, patience = int(mcfg["epochs"]), int(mcfg.get("patience", 10))
    eval_every = int(mcfg.get("eval_every", 1))
    bs = int(mcfg["batch_size"])

    best_score, best_state, best_epoch, bad = -np.inf, None, 0, 0
    history, train_time = [], 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, epochs + 1):
        module.train()
        total, n_seen = 0.0, 0
        with Timer() as t:
            perm = torch.randperm(len(pairs), device=device, generator=gen)
            for batch in perm.split(bs):
                u, i = pairs[batch, 0], pairs[batch, 1]
                loss = module.batch_loss(batch, u, i, sampler)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total += float(loss.detach()) * len(batch)
                n_seen += len(batch)
        train_time += t.elapsed
        row = {"epoch": epoch, "loss": total / n_seen, "epoch_time": t.elapsed}
        if evaluator is not None and (epoch % eval_every == 0 or epoch == epochs):
            module.eval()
            with torch.no_grad():
                U, I = module.compute_embeddings()
            topk = evaluator.topk(lambda users: U[users] @ I.T)
            res = evaluator.metrics_from_topk(topk, by_group=True)
            m = res["all"]
            row.update({f"val/{k}": m[k] for k in ("recall@20", "ndcg@20", "recall@10", "ndcg@10") if k in m})
            # per-group curves: a model can get better for warm users while it breaks for cold ones
            row.update({f"val/{g}/{primary}": res[g][primary] for g in ("warm", "sparse", "cold")
                        if primary in res[g]})
            if primary not in m:  # evaluator built with other cutoffs (tests)
                primary = f"recall@{max(evaluator.ks)}"
            score = m[primary]
            if score > best_score + 1e-6:
                best_score, best_epoch, bad = score, epoch, 0
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in module.state_dict().items()})
            else:
                bad += 1
        history.append(row)
        if tracker is not None:
            tracker.log({k: v for k, v in row.items() if k != "epoch"}, step=epoch)
        if epoch == 1 or epoch % 5 == 0 or bad == 0:
            log.info("%s epoch %d loss %.4f %s (%.1fs)", name, epoch, row["loss"],
                     f"val {primary} {row.get('val/' + primary, float('nan')):.4f}" if evaluator is not None else "",
                     t.elapsed)
        if evaluator is not None and bad >= patience:
            log.info("%s early stop at epoch %d, best epoch %d (%s=%.4f)", name, epoch, best_epoch, primary, best_score)
            break
    if best_state is not None:
        module.load_state_dict(best_state)
    peak_mem = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
    return {"history": history, "best_epoch": best_epoch, "best_val": float(best_score),
            "epochs_run": len(history), "train_time_s": train_time, "peak_gpu_mem_mb": peak_mem,
            "n_params": count_parameters(module)}


class TorchEmbeddingRecommender(EmbeddingRecommender):
    """Wraps an nn.Module that follows the batch_loss / compute_embeddings convention."""

    def build_module(self, data: RecData) -> nn.Module:
        raise NotImplementedError

    def fit(self, data: RecData, evaluator=None, tracker=None):
        self.module = self.build_module(data).to(self.device)
        self.fit_info = train_embedding_module(self.module, data, self.cfg, evaluator, self.device, tracker,
                                               seed=int(self.cfg.get("seed", 0)), name=self.name)
        self.train_log = self.fit_info["history"]
        self.refresh()
        return self

    def refresh(self, **kwargs) -> None:
        self.module.eval()
        with torch.no_grad():
            U, I = self.module.compute_embeddings(**kwargs)
        self.set_embeddings(U, I)

    def num_parameters(self) -> int:
        return count_parameters(self.module)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.module.state_dict(), "cfg": self.cfg}, path / "model.pt")
        self.save_embeddings(path / "embeddings.npz")

    def load(self, path: Path, data: RecData):
        ckpt = torch.load(Path(path) / "model.pt", map_location=self.device, weights_only=False)
        self.cfg = ckpt["cfg"]
        self.module = self.build_module(data).to(self.device)
        self.module.load_state_dict(ckpt["state_dict"])
        self.refresh()
        return self
