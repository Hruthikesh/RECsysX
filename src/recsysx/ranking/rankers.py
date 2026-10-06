"""Second-stage rankers with a common fit / predict interface.

X: feature matrix of (user, candidate) rows, y: 0/1 label, groups: user id of each row.
Rows of one user have to be contiguous (that is how the feature builder produces them).

logreg  pointwise logistic regression on standardised features, the simple baseline
xgb     XGBoost LambdaMART (rank:ndcg), optimises the order inside each user's list
neural  small MLP trained with a listwise softmax loss per user
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch import nn

from ..evaluation.metrics import ranking_metrics_per_user


def group_sizes(groups: np.ndarray) -> np.ndarray:
    change = np.flatnonzero(np.diff(groups)) + 1
    bounds = np.concatenate([[0], change, [len(groups)]])
    return np.diff(bounds)


def ndcg_from_scores(scores: np.ndarray, y: np.ndarray, groups: np.ndarray, k: int = 10) -> float:
    """Mean NDCG@k of re-ordering each user's candidates by score (only users with a positive)."""
    out, start = [], 0
    for size in group_sizes(groups):
        s, l = scores[start:start + size], y[start:start + size]
        start += size
        if l.sum() == 0:
            continue
        order = np.argsort(-s, kind="stable")[:k]
        hits = l[order][None, :].astype(bool)
        if hits.shape[1] < k:
            hits = np.pad(hits, ((0, 0), (0, k - hits.shape[1])))
        out.append(ranking_metrics_per_user(hits, np.array([l.sum()]), [k])[f"ndcg@{k}"][0])
    return float(np.mean(out)) if out else 0.0


class LogRegRanker:
    name = "logreg"

    def __init__(self, C: float = 1.0, **_):
        self.scaler = StandardScaler()
        self.model = LogisticRegression(C=C, max_iter=2000)

    def fit(self, X, y, groups, X_val=None, y_val=None, g_val=None):
        self.model.fit(self.scaler.fit_transform(X), y)
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.decision_function(self.scaler.transform(X))

    def info(self) -> dict:
        return {"best_iteration": None}


class XGBRanker:
    name = "xgb"

    def __init__(self, n_estimators: int = 1000, learning_rate: float = 0.05, max_depth: int = 6,
                 subsample: float = 0.8, colsample_bytree: float = 0.8, early_stopping_rounds: int = 50,
                 device: str = "cuda", seed: int = 0, objective: str = "rank:ndcg", **_):
        import xgboost as xgb

        self.params = dict(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
                           subsample=subsample, colsample_bytree=colsample_bytree, objective=objective,
                           eval_metric="ndcg@10", tree_method="hist", device=device, random_state=seed,
                           early_stopping_rounds=early_stopping_rounds)
        self.model = xgb.XGBRanker(**self.params)

    def fit(self, X, y, groups, X_val=None, y_val=None, g_val=None):
        kw = {}
        if X_val is not None:
            kw = dict(eval_set=[(X_val, y_val)], eval_group=[group_sizes(g_val)], verbose=False)
        self.model.fit(X, y, group=group_sizes(groups), **kw)
        # predictions are timed on CPU later, the trees are moved off the GPU here
        self.model.set_params(device="cpu")
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(np.asarray(X, dtype=np.float32))

    def feature_importance(self, names: list[str]) -> dict[str, float]:
        gain = self.model.get_booster().get_score(importance_type="gain")
        return {n: float(gain.get(f"f{j}", gain.get(n, 0.0))) for j, n in enumerate(names)}

    def info(self) -> dict:
        return {"best_iteration": int(getattr(self.model, "best_iteration", -1))}


class _MLP(nn.Module):
    def __init__(self, d_in: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden // 2, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class NeuralRanker:
    """Listwise loss: softmax over a user's candidates, cross-entropy against the
    normalised label distribution (ListNet with several positives)."""
    name = "neural"

    def __init__(self, hidden: int = 128, dropout: float = 0.1, lr: float = 1e-3, weight_decay: float = 1e-5,
                 epochs: int = 100, users_per_batch: int = 64, patience: int = 10, seed: int = 0,
                 device: str = "cuda", **_):
        self.hp = dict(hidden=hidden, dropout=dropout, lr=lr, weight_decay=weight_decay, epochs=epochs,
                       users_per_batch=users_per_batch, patience=patience, seed=seed)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.scaler = StandardScaler()
        self.history: list[dict] = []

    def _lists(self, X, y, groups):
        sizes = group_sizes(groups)
        n_max = sizes.max()
        Xp = np.zeros((len(sizes), n_max, X.shape[1]), dtype=np.float32)
        Yp = np.zeros((len(sizes), n_max), dtype=np.float32)
        M = np.zeros((len(sizes), n_max), dtype=bool)
        start = 0
        for j, s in enumerate(sizes):
            Xp[j, :s], Yp[j, :s], M[j, :s] = X[start:start + s], y[start:start + s], True
            start += s
        return Xp, Yp, M

    def fit(self, X, y, groups, X_val=None, y_val=None, g_val=None):
        torch.manual_seed(self.hp["seed"])
        Xs = self.scaler.fit_transform(X).astype(np.float32)
        Xp, Yp, M = self._lists(Xs, np.asarray(y, dtype=np.float32), groups)
        has_pos = Yp.sum(1) > 0               # lists without any positive carry no listwise signal
        Xp, Yp, M = (torch.as_tensor(a[has_pos], device=self.device) for a in (Xp, Yp, M))
        self.model = _MLP(X.shape[1], self.hp["hidden"], self.hp["dropout"]).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.hp["lr"], weight_decay=self.hp["weight_decay"])
        best, best_state, bad = -1.0, None, 0
        for epoch in range(self.hp["epochs"]):
            self.model.train()
            perm = torch.randperm(len(Xp), device=self.device)
            total = 0.0
            for b in perm.split(self.hp["users_per_batch"]):
                logits = self.model(Xp[b]).masked_fill(~M[b], float("-inf"))
                target = Yp[b] / Yp[b].sum(1, keepdim=True)
                loss = -(target * F.log_softmax(logits, dim=1).masked_fill(~M[b], 0.0)).sum(1).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss) * len(b)
            row = {"epoch": epoch + 1, "loss": total / len(Xp)}
            if X_val is not None:
                row["val_ndcg@10"] = ndcg_from_scores(self.predict(X_val), y_val, g_val)
                if row["val_ndcg@10"] > best + 1e-5:
                    best, bad = row["val_ndcg@10"], 0
                    best_state = copy.deepcopy(self.model.state_dict())
                else:
                    bad += 1
            self.history.append(row)
            if X_val is not None and bad >= self.hp["patience"]:
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model = self.model.cpu()
        self.device = torch.device("cpu")
        return self

    def predict(self, X) -> np.ndarray:
        self.model.eval()
        x = torch.as_tensor(self.scaler.transform(X).astype(np.float32), device=self.device)
        with torch.no_grad():
            return self.model.to(self.device)(x).cpu().numpy()

    def info(self) -> dict:
        best = max((h.get("val_ndcg@10", -1) for h in self.history), default=None)
        return {"epochs_run": len(self.history), "best_val_ndcg@10": best}


RANKERS = {"logreg": LogRegRanker, "xgb": XGBRanker, "neural": NeuralRanker}
