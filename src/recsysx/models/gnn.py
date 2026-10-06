"""GraphSAGE / GAT recommenders on the user-item graph, plus the hybrid content variants.

Users and items are nodes of one homogeneous graph (items are offset by n_users), every
training positive is an undirected edge. Node input features can be

  * a learned id embedding (use_id)          -> "graph only"
  * a projection of the content features     -> "content + graph"
  * both, summed                              -> hybrid
and num_layers=0 turns message passing off, which gives the collaborative-only and
content-only reference points of the hybrid ablation with exactly the same training code.

Training is link prediction with a BPR loss on (user, positive item, sampled negative).
"""
from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GATConv, LGConv, SAGEConv

from ..data.dataset import RecData
from ..graph.build import bipartite_edges
from .layers import bpr_loss, l2_batch
from .trainer import TorchEmbeddingRecommender


class GNNModule(nn.Module):
    def __init__(self, n_users: int, n_items: int, train_edges: torch.Tensor,
                 known_users: torch.Tensor, known_items: torch.Tensor,
                 user_content: torch.Tensor | None = None, item_content: torch.Tensor | None = None,
                 conv: str = "sage", num_layers: int = 2, dim: int = 64, aggr: str = "mean", heads: int = 4,
                 dropout: float = 0.1, attn_dropout: float = 0.0, use_id: bool = True, use_content: bool = False,
                 user_id_dropout: float = 0.1, item_id_dropout: float = 0.0, remove_target_edges: bool = True,
                 jk: str = "mean", l2_reg: float = 1e-4, num_negatives: int = 1, reg_target: str = "input",
                 conv_weight_decay: float = 0.0, cold_simulation: bool = True):
        super().__init__()
        if not use_id and not use_content:
            raise ValueError("need id embeddings, content features or both")
        self.n_users, self.n_items = n_users, n_items
        self.N = n_users + n_items
        self.use_id, self.use_content = use_id, use_content
        self.p_user, self.p_item = user_id_dropout, item_id_dropout
        self.remove_target_edges = remove_target_edges
        self.jk, self.l2_reg, self.num_negatives = jk, l2_reg, num_negatives
        self.reg_target, self.conv_weight_decay = reg_target, conv_weight_decay
        self.cold_simulation = cold_simulation
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("edges", train_edges)            # (E, 2): user, item + n_users
        self.register_buffer("known_users", known_users)
        self.register_buffer("known_items", known_items)

        if use_id:
            # rows: users, items, unknown user (N), unknown item (N + 1)
            self.emb = nn.Embedding(self.N + 2, dim)
            nn.init.normal_(self.emb.weight, std=0.1)
        if use_content:
            self.register_buffer("user_content", user_content)
            self.register_buffer("item_content", item_content)
            self.user_proj = nn.Sequential(nn.Linear(user_content.shape[1], dim), nn.ReLU(), nn.Linear(dim, dim))
            self.item_proj = nn.Sequential(nn.Linear(item_content.shape[1], dim), nn.ReLU(), nn.Linear(dim, dim))

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            if conv == "sage":
                self.convs.append(SAGEConv(dim, dim, aggr=aggr))
            elif conv == "gat":
                if dim % heads:
                    raise ValueError("dim must be divisible by heads")
                self.convs.append(GATConv(dim, dim // heads, heads=heads, concat=True, dropout=attn_dropout,
                                          add_self_loops=True))
            elif conv == "lgc":
                # LightGCN propagation: symmetric-normalised sum, no weights, no nonlinearity.
                # Used as a reference point for "graph propagation without learned transforms".
                self.convs.append(LGConv())
            else:
                raise ValueError(f"unknown conv {conv}")
        if jk == "cat" and num_layers > 0:
            self.jk_lin = nn.Linear(dim * (num_layers + 1), dim)

    # ------------------------------------------------------------------ inputs
    def node_ids(self, drop_users: torch.Tensor | None = None, drop_items: torch.Tensor | None = None) -> torch.Tensor:
        """Node ids into the embedding table. Unknown nodes (no training positives) and the nodes
        whose id is dropped in this training step use the unknown-user / unknown-item rows."""
        dev = self.edges.device
        ids = torch.arange(self.N, device=dev)
        u_unk = torch.zeros(self.n_users, dtype=torch.bool, device=dev)
        i_unk = torch.zeros(self.n_items, dtype=torch.bool, device=dev)
        if self.p_user > 0:
            u_unk |= ~self.known_users
            if drop_users is not None:
                u_unk |= drop_users
        if self.p_item > 0:
            i_unk |= ~self.known_items
            if drop_items is not None:
                i_unk |= drop_items
        ids[: self.n_users][u_unk] = self.N
        ids[self.n_users:][i_unk] = self.N + 1
        return ids

    def input_features(self, drop_users: torch.Tensor | None = None, drop_items: torch.Tensor | None = None) -> torch.Tensor:
        x = 0
        if self.use_id:
            x = self.emb(self.node_ids(drop_users, drop_items))
        if self.use_content:
            x = x + torch.cat([self.user_proj(self.user_content), self.item_proj(self.item_content)], dim=0)
        return x

    def message_edges(self, drop: torch.Tensor | None = None, extra: torch.Tensor | None = None,
                      drop_users: torch.Tensor | None = None, drop_items: torch.Tensor | None = None) -> torch.Tensor:
        e = self.edges
        if drop is not None or drop_users is not None or drop_items is not None:
            keep = torch.ones(len(e), dtype=torch.bool, device=e.device)
            if drop is not None:
                keep[drop] = False
            if drop_users is not None:
                keep &= ~drop_users[e[:, 0]]
            if drop_items is not None:
                keep &= ~drop_items[e[:, 1] - self.n_users]
            e = e[keep]
        if extra is not None and len(extra):
            e = torch.cat([e, extra.to(e.device)], dim=0)
        return torch.cat([e.T, e.flip(1).T], dim=1)

    def propagate(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if not self.convs:
            return x
        hs, h = [x], x
        linear = isinstance(self.convs[0], LGConv)
        for layer, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if layer < len(self.convs) - 1 and not linear:
                h = self.dropout(torch.relu(h))
            hs.append(h)
        if self.jk == "last":
            return h
        if self.jk == "mean":
            return torch.stack(hs).mean(0)
        return self.jk_lin(torch.cat(hs, dim=1))

    # ------------------------------------------------------------------ training / inference
    def batch_loss(self, edge_ids, users, items, sampler):
        # The supervised edges of this batch are taken out of the message-passing graph.
        # Otherwise the model can score (u, i) high just because i is literally one of u's
        # neighbours, which never happens for the future interactions we evaluate on.
        dev = self.edges.device
        du = torch.rand(self.n_users, device=dev) < self.p_user if self.p_user > 0 else None
        di = torch.rand(self.n_items, device=dev) < self.p_item if self.p_item > 0 else None
        # A cold user at inference has the unknown id *and* no edges. If the dropped users kept
        # their edges, the unknown row would only ever be trained together with a neighbourhood,
        # and the weighted GNNs produced garbage for real cold users (cold recall fell to ~0 while
        # warm recall improved, see results/logs/gnn_training_diagnostics.txt). So the dropped
        # users/items also lose their edges for this step.
        cold_u = du if self.cold_simulation else None
        cold_i = di if self.cold_simulation else None
        edge_index = self.message_edges(edge_ids if self.remove_target_edges else None,
                                        drop_users=cold_u, drop_items=cold_i)
        x = self.input_features(du, di)
        h = self.propagate(x, edge_index)
        neg = sampler.sample(users, self.num_negatives)
        iu, ip, ineg = users, items + self.n_users, neg + self.n_users
        loss = bpr_loss(h[iu], h[ip], h[ineg])
        # "input": L2 on the layer-0 embeddings (LightGCN style). "output": L2 on the embeddings
        # that are actually scored, which also limits how much the layer weights can inflate them.
        r = x if self.reg_target == "input" else h
        loss = loss + self.l2_reg * l2_batch(r[iu], r[ip], r[ineg].reshape(-1, r.shape[1]))
        if self.conv_weight_decay > 0:
            loss = loss + self.conv_weight_decay * sum(w.pow(2).sum() for w in self.convs.parameters()) / 2
        return loss

    def compute_embeddings(self, extra_edges: torch.Tensor | None = None):
        """extra_edges: (k, 2) user, item+n_users edges added at inference only
        (used for the new-user onboarding experiment, no retraining)."""
        x = self.input_features()
        h = self.propagate(x, self.message_edges(extra=extra_edges))
        return h[: self.n_users], h[self.n_users:]


class GNNRecommender(TorchEmbeddingRecommender):
    name = "gnn"

    def build_module(self, data: RecData) -> nn.Module:
        c = self.cfg
        use_content = bool(c.get("use_content", False))
        uc = torch.as_tensor(data.user_content, dtype=torch.float32) if use_content else None
        ic = torch.as_tensor(data.item_content, dtype=torch.float32) if use_content else None
        return GNNModule(
            data.n_users, data.n_items, torch.as_tensor(bipartite_edges(data.train_pos, data.n_users)),
            torch.as_tensor(data.known_users()), torch.as_tensor(data.known_items()), uc, ic,
            conv=c.get("conv", "sage"), num_layers=int(c.get("num_layers", 2)), dim=int(c["embedding_dim"]),
            aggr=c.get("aggr", "mean"), heads=int(c.get("heads", 4)), dropout=float(c.get("dropout", 0.1)),
            attn_dropout=float(c.get("attn_dropout", 0.0)), use_id=bool(c.get("use_id", True)),
            use_content=use_content, user_id_dropout=float(c.get("user_id_dropout", 0.1)),
            item_id_dropout=float(c.get("item_id_dropout", 0.0)),
            remove_target_edges=bool(c.get("remove_target_edges", True)), jk=c.get("jk", "mean"),
            l2_reg=float(c.get("l2_reg", 1e-4)), num_negatives=int(c.get("num_negatives", 1)),
            reg_target=c.get("reg_target", "input"), conv_weight_decay=float(c.get("conv_weight_decay", 0.0)),
            cold_simulation=bool(c.get("cold_simulation", True)),
        )

    def refresh_with_edges(self, extra_pairs) -> None:
        """Recompute embeddings with extra (user, item) edges, e.g. revealed ratings of new users."""
        extra = torch.as_tensor(bipartite_edges(extra_pairs, self.module.n_users), device=self.device)
        self.refresh(extra_edges=extra)
