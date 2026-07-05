"""Temporal neighborhood heuristics for link prediction — aimed at the NOVEL-edge headroom.

For a query (s, r, ?, t) we score candidates by neighborhood overlap with s in the trailing
K-week subgraph (edges with t-K <= time < t; strictly PIT-clean):
  - CN : common neighbors      score(s,o) = |N(s) ∩ N(o)|                    = (A · A)[s,o]
  - AA : Adamic-Adar           score(s,o) = Σ_{z∈N(s)∩N(o)} 1/log deg(z)    = (A · WA)[s,o]
Both are computed for all candidates at once via sparse matrix products on the recent adjacency
A. Unlike recurrence these fire on NOVEL edges (s and o never linked before but sharing recent
neighbors). Reported in the same novel-vs-recurring frame, with the recurrence->AA hybrid.

Run: `.venv/Scripts/python -m dynamics.temporal_heur`
"""
from pathlib import Path
import sys
from collections import deque, defaultdict
import numpy as np
import pandas as pd
import scipy.sparse as sp

from dynamics.loader import load_core
from dynamics.linkpred import group_rank, Meter

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
OUT = Path("data/dynamics/linkpred")
K = 12  # trailing window (weeks)
METHODS = ["recurrence", "CN", "AA", "rec->CN", "rec->AA"]
SUBSETS = ["all", "recurring", "novel"]


def build_recent(window, n):
    """Binary undirected recent-adjacency A and Adamic-Adar weighted WA = diag(1/log deg)·A."""
    ss = np.concatenate([w[1] for w in window]); oo = np.concatenate([w[2] for w in window])
    src = np.concatenate([ss, oo]); dst = np.concatenate([oo, ss])           # undirected
    A = sp.coo_matrix((np.ones(len(src), np.float32), (src, dst)), shape=(n, n)).tocsr()
    A.data[:] = 1.0                                                          # binarise (dedup)
    deg = np.asarray(A.sum(1)).ravel()
    WA = sp.diags(1.0 / np.log(np.maximum(deg, 2.0))) @ A
    return A, WA


def ranks_from_csr(R, true_ids, n, query_ids=None):
    """Raw average rank of each true id within its row of the sparse score matrix R.

    query_ids: the query entity per row, EXCLUDED from the candidate pool (audit fix) -- the
    diagonal of A@A is deg(q) >= CN(q, o) for every o (likewise for AA), so leaving the query in
    guarantees one distractor that ties-or-beats every real candidate. Genuine self-loop queries
    (true id == query id) keep the full pool."""
    out = np.empty(len(true_ids))
    indptr, indices, data = R.indptr, R.indices, R.data
    for i, o in enumerate(true_ids):
        a, b = indptr[i], indptr[i + 1]
        cols, vals = indices[a:b], data[a:b]
        n_eff = n
        if query_ids is not None and query_ids[i] != o:
            q = np.where(cols == query_ids[i])[0]
            if q.size:
                keep = np.ones(len(cols), dtype=bool); keep[q[0]] = False
                cols, vals = cols[keep], vals[keep]
            n_eff = n - 1
        hit = np.where(cols == o)[0]
        st = vals[hit[0]] if hit.size else 0.0
        if st > 0:
            higher = int((vals > st).sum()); equal = int((vals == st).sum()) - 1
            out[i] = 1 + higher + equal / 2
        else:
            nnz = vals.size
            out[i] = 1 + nnz + (n_eff - nnz - 1) / 2
    return out


def main():
    """Chronological sweep: for each test week, build A/WA from the trailing K-week window
    (edges with time < t only), rank CN/AA/recurrence and the hybrids, then update the counts
    and window AFTER the week is scored (strict PIT). Writes temporal_heur_results.csv."""
    kg = load_core(drop_noise=True)
    N = kg.n_entities
    test_lo = kg.splits["test"][0]
    e = kg.edges.sort_values("time")
    by_week = {t: g[["subj", "rel", "obj"]].to_numpy() for t, g in e.groupby("time")}

    sro = defaultdict(lambda: defaultdict(int)); ors = defaultdict(lambda: defaultdict(int))
    window = deque()
    meters = {(m, s): Meter() for m in METHODS for s in SUBSETS}

    for t in range(kg.n_times):
        wk = by_week.get(t)
        if wk is not None and t >= test_lo:
            S_t, O_t = wk[:, 0], wk[:, 2]
            if window:
                A, WA = build_recent(window, N)
                cn_o = ranks_from_csr((A[S_t] @ A).tocsr(), O_t, N, query_ids=S_t)
                aa_o = ranks_from_csr((A[S_t] @ WA).tocsr(), O_t, N, query_ids=S_t)
                cn_s = ranks_from_csr((A[O_t] @ A).tocsr(), S_t, N, query_ids=O_t)
                aa_s = ranks_from_csr((A[O_t] @ WA).tocsr(), S_t, N, query_ids=O_t)
            else:
                cn_o = aa_o = cn_s = aa_s = np.full(len(wk), 1 + (N - 1) / 2)
            for i, (s, r, o) in enumerate(wk):
                go = sro.get((s, r), {}); gs = ors.get((o, r), {})
                seen = go.get(o, 0) > 0
                sub = "recurring" if seen else "novel"
                rec = (group_rank(go, o, N), group_rank(gs, s, N))
                pairs = {
                    "recurrence": rec,
                    "CN": (cn_o[i], cn_s[i]),
                    "AA": (aa_o[i], aa_s[i]),
                    "rec->CN": rec if seen else (cn_o[i], cn_s[i]),
                    "rec->AA": rec if seen else (aa_o[i], aa_s[i]),
                }
                for m, (ro, rs) in pairs.items():
                    for g in ("all", sub):
                        meters[(m, g)].add(ro); meters[(m, g)].add(rs)
        if wk is not None:
            for s, r, o in wk:
                sro[(s, r)][o] += 1; ors[(o, r)][s] += 1
            window.append((t, wk[:, 0].copy(), wk[:, 2].copy()))
            while window and window[0][0] <= t - K:
                window.popleft()

    rows = {}
    for m in METHODS:
        row = {f"MRR/{g}": meters[(m, g)].row()["MRR"] for g in SUBSETS}
        row["H10/all"] = meters[(m, "all")].row()["H10"]
        rows[m] = row
    tbl = pd.DataFrame(rows).T
    tbl.to_csv(OUT / "temporal_heur_results.csv")
    with pd.option_context("display.float_format", lambda v: f"{v:.4f}"):
        print(f"\ntrailing window K={K} wk\n\n" + tbl.to_string())
    print(f"\nsaved {OUT/'temporal_heur_results.csv'}")


if __name__ == "__main__":
    main()
