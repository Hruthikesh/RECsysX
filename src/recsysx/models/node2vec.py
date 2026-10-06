"""Node2Vec embeddings of the user-item graph used directly for recommendation.

The point of this baseline: are graph-derived embeddings useful on their own, before any
GNN is trained with a recommendation loss? Node2Vec only optimises a skip-gram objective
over random walks, it never sees "user liked item" as a training target.

Note on p/q: the graph is bipartite, so a walk that went t -> v can only move to a node x
on the same side as t. The distance d(t, x) is therefore 0 (going back) or 2, never 1.
Only the return parameter p relative to q matters here, the BFS/DFS interpretation
from the paper does not really apply.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from torch_geometric.nn import Node2Vec

from ..data.dataset import RecData
from ..graph.build import edge_index
from ..utils import Timer, count_parameters, get_logger
from .base import EmbeddingRecommender

log = get_logger()


class Node2VecRecommender(EmbeddingRecommender):
    name = "node2vec"

    def _build(self, data: RecData) -> Node2Vec:
        c = self.cfg
        return Node2Vec(
            edge_index(data.train_pos, data.n_users), embedding_dim=int(c["embedding_dim"]),
            walk_length=int(c.get("walk_length", 20)), context_size=int(c.get("context_size", 10)),
            walks_per_node=int(c.get("walks_per_node", 10)), p=float(c.get("p", 1.0)), q=float(c.get("q", 1.0)),
            num_negative_samples=int(c.get("num_negative_samples", 1)),
            num_nodes=data.n_users + data.n_items, sparse=True,
        ).to(self.device)

    def _split(self, data: RecData, emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        U, I = emb[: data.n_users].clone(), emb[data.n_users:].clone()
        known = torch.as_tensor(data.known_users(), device=emb.device)
        # isolated user nodes only ever saw negative samples, use the average user instead
        U[~known] = U[known].mean(0)
        if self.cfg.get("score", "dot") == "cosine":
            U, I = F.normalize(U, dim=1), F.normalize(I, dim=1)
        return U, I

    def fit(self, data: RecData, evaluator=None, tracker=None):
        c = self.cfg
        torch.manual_seed(int(c.get("seed", 0)))
        self.module = self._build(data)
        loader = self.module.loader(batch_size=int(c.get("batch_size", 128)), shuffle=True, num_workers=0)
        opt = torch.optim.SparseAdam(list(self.module.parameters()), lr=float(c.get("lr", 0.01)))
        best, best_state, best_epoch, bad, train_time = -np.inf, None, 0, 0, 0.0
        history = []
        for epoch in range(1, int(c["epochs"]) + 1):
            self.module.train()
            total, n = 0.0, 0
            with Timer() as t:
                for pos_rw, neg_rw in loader:
                    opt.zero_grad()
                    loss = self.module.loss(pos_rw.to(self.device), neg_rw.to(self.device))
                    loss.backward()
                    opt.step()
                    total += float(loss)
                    n += 1
            train_time += t.elapsed
            row = {"epoch": epoch, "loss": total / n, "epoch_time": t.elapsed}
            if evaluator is not None:
                with torch.no_grad():
                    U, I = self._split(data, self.module.embedding.weight.detach())
                topk = evaluator.topk(lambda users: U[users] @ I.T)
                m = evaluator.metrics_from_topk(topk, by_group=False)["all"]
                row.update({f"val/{k}": m[k] for k in ("recall@20", "ndcg@20", "recall@10", "ndcg@10") if k in m})
                primary = "recall@20" if "recall@20" in m else f"recall@{max(evaluator.ks)}"
                if m[primary] > best + 1e-6:
                    best, best_epoch, bad = m[primary], epoch, 0
                    best_state = self.module.embedding.weight.detach().clone()
                else:
                    bad += 1
            history.append(row)
            if tracker is not None:
                tracker.log({k: v for k, v in row.items() if k != "epoch"}, step=epoch)
            log.info("node2vec epoch %d loss %.4f val recall@20 %.4f (%.1fs)", epoch, row["loss"],
                     row.get("val/recall@20", float("nan")), t.elapsed)
            if evaluator is not None and bad >= int(c.get("patience", 3)):
                break
        if best_state is not None:
            self.module.embedding.weight.data.copy_(best_state)
        self.train_log = history
        self.fit_info = {"history": history, "best_epoch": best_epoch, "best_val": float(best),
                         "epochs_run": len(history), "train_time_s": train_time,
                         "n_params": count_parameters(self.module), "peak_gpu_mem_mb": None}
        self.raw_embeddings = self.module.embedding.weight.detach()
        U, I = self._split(data, self.raw_embeddings)
        self.set_embeddings(U, I)
        if c.get("score", "dot") == "hadamard_lr":
            self._fit_hadamard_lr(data, U, I)
        return self

    def _fit_hadamard_lr(self, data: RecData, U: torch.Tensor, I: torch.Tensor, n_samples: int = 200_000):
        """Classic node2vec link prediction: logistic regression on u * i edge features.
        The resulting score is w . (u * i) + b, i.e. a weighted dot product, so it can still
        be computed for all items at once."""
        rng = np.random.default_rng(int(self.cfg.get("seed", 0)))
        idx = rng.choice(len(data.train_pos), min(n_samples, len(data.train_pos)), replace=False)
        pos = data.train_pos[idx]
        neg = np.stack([pos[:, 0], rng.integers(0, data.n_items, len(pos))], axis=1)
        pairs = np.concatenate([pos, neg])
        y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        Un, In = U.cpu().numpy(), I.cpu().numpy()
        X = Un[pairs[:, 0]] * In[pairs[:, 1]]
        clf = LogisticRegression(max_iter=1000).fit(X, y)
        w = torch.as_tensor(clf.coef_[0], dtype=torch.float32, device=self.device)
        self.set_embeddings(U * w, I)

    def num_parameters(self) -> int:
        return count_parameters(self.module)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save({"embedding": self.raw_embeddings.cpu(), "cfg": self.cfg}, path / "model.pt")
        self.save_embeddings(path / "embeddings.npz")

    def load(self, path: Path, data: RecData):
        ckpt = torch.load(Path(path) / "model.pt", map_location=self.device, weights_only=False)
        self.cfg = ckpt["cfg"]
        self.load_embeddings(Path(path) / "embeddings.npz")
        self.raw_embeddings = ckpt["embedding"].to(self.device)
        return self
