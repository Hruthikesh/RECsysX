"""FAISS candidate generation.

Item vectors go into an index, user vectors are the queries, score = inner product
(which is cosine for the two-tower model since its vectors are normalised).

Filtering already-seen items: FAISS cannot exclude a different id set per query in one
batched search without IDSelector tricks, so we over-fetch k = n + (largest history in
the batch) and drop the seen items afterwards. With 3.7k items this is cheap. For a
catalog with millions of items you would cap the history used for filtering.
"""
from __future__ import annotations

import time

import faiss
import numpy as np
import scipy.sparse as sp


def build_index(item_vectors: np.ndarray, kind: str = "flat", nlist: int = 64, nprobe: int = 8,
                hnsw_m: int = 32, ef_search: int = 128, seed: int = 0) -> faiss.Index:
    x = np.ascontiguousarray(item_vectors, dtype=np.float32)
    d = x.shape[1]
    if kind == "flat":
        index = faiss.IndexFlatIP(d)
    elif kind == "ivf":
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        index.cp.seed = seed
        index.train(x)
        index.nprobe = nprobe
    elif kind == "hnsw":
        index = faiss.IndexHNSWFlat(d, hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efSearch = ef_search
    else:
        raise ValueError(f"unknown index type {kind}")
    index.add(x)
    return index


class FaissRetriever:
    def __init__(self, item_vectors: np.ndarray, kind: str = "flat", **params):
        self.kind = kind
        self.params = params
        t = time.perf_counter()
        self.index = build_index(item_vectors, kind, **params)
        self.build_time_s = time.perf_counter() - t
        self.n_items = item_vectors.shape[0]

    def _set_search_k(self, k: int) -> None:
        if self.kind == "hnsw":
            # HNSW returns at most efSearch results, it has to be >= k
            self.index.hnsw.efSearch = max(self.params.get("ef_search", 128), k)

    def search(self, user_vectors: np.ndarray, n: int, history: sp.csr_matrix | None = None,
               user_ids: np.ndarray | None = None, batch_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
        """Top-n unseen items per query. Returns (item_ids, scores), -1 / -inf pad if fewer found."""
        q = np.ascontiguousarray(user_vectors, dtype=np.float32)
        out_ids = np.full((len(q), n), -1, dtype=np.int64)
        out_sc = np.full((len(q), n), -np.inf, dtype=np.float32)
        hist_len = np.diff(history.indptr)[user_ids] if history is not None else np.zeros(len(q), dtype=int)
        for s in range(0, len(q), batch_size):
            e = min(s + batch_size, len(q))
            k = int(min(self.n_items, n + hist_len[s:e].max()))
            self._set_search_k(k)
            scores, ids = self.index.search(q[s:e], k)
            for r in range(e - s):
                row_ids, row_sc = ids[r], scores[r]
                keep = row_ids >= 0
                if history is not None:
                    u = user_ids[s + r]
                    seen = history.indices[history.indptr[u]:history.indptr[u + 1]]
                    keep &= ~np.isin(row_ids, seen)
                row_ids, row_sc = row_ids[keep][:n], row_sc[keep][:n]
                out_ids[s + r, :len(row_ids)] = row_ids
                out_sc[s + r, :len(row_ids)] = row_sc
        return out_ids, out_sc

    def memory_bytes(self) -> int:
        return int(faiss.serialize_index(self.index).nbytes)


def candidate_recall(cand: np.ndarray, target_lists: list[np.ndarray], ns: list[int]) -> dict[str, float]:
    """Share of each user's targets that made it into the first n candidates (mean over users)."""
    out = {}
    for n in ns:
        r = [np.isin(t, c[:n]).mean() for c, t in zip(cand, target_lists) if len(t)]
        out[f"recall@{n}"] = float(np.mean(r))
    return out


def ann_overlap(approx: np.ndarray, exact: np.ndarray) -> float:
    """How much of the exact top-n the approximate index returns (ANN recall)."""
    hits = [len(np.intersect1d(a[a >= 0], b)) / max(len(b), 1) for a, b in zip(approx, exact)]
    return float(np.mean(hits))
