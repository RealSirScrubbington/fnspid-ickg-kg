"""Temporal link prediction — PIT-clean eval harness + non-neural baselines.

Forecasting / extrapolation protocol: for each test quad (s, r, o, t), rank candidate entities
using ONLY history with time < t (strict point-in-time; no same-t or future facts). Raw
(unfiltered) ranking. Metrics: MRR and Hits@1/3/10, averaged over object-prediction
(s, r, ?) and subject-prediction (?, r, o) across all test quads.

Baselines:
  - recurrence : score(o') = #times (s, r, o') seen in history < t   (recency-agnostic count)
  - popularity : score(o') = #times o' appeared (as that role) in history < t
  - backoff    : recurrence if (s, r, o) seen before, else popularity  (handles novel edges)

Run: `.venv/Scripts/python -m dynamics.linkpred`
"""
from pathlib import Path
import sys
from collections import defaultdict
import numpy as np
import pandas as pd

from dynamics.loader import load_core

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
OUT = Path("data/dynamics/linkpred")
OUT.mkdir(parents=True, exist_ok=True)


def group_rank(group: dict, true: int, n: int) -> float:
    """Average/realistic rank of `true` among n candidates, where `group` maps the
    nonzero-scoring candidates to their counts (all unlisted candidates score 0)."""
    c = group.get(true, 0)
    if c > 0:
        higher = sum(1 for v in group.values() if v > c)
        equal = sum(1 for v in group.values() if v == c) - 1     # exclude self
        return 1 + higher + equal / 2
    nnz = len(group)                                             # all listed score > 0 > c
    nzero = n - nnz                                              # zero-score mass incl. true
    return 1 + nnz + (nzero - 1) / 2


def pop_rank(sorted_asc: np.ndarray, c: float, n: int) -> float:
    """Average rank of a candidate with popularity `c` given the sorted popularity vector."""
    right = np.searchsorted(sorted_asc, c, side="right")
    left = np.searchsorted(sorted_asc, c, side="left")
    higher = n - right
    equal = right - left - 1
    return 1 + higher + equal / 2


class Meter:
    def __init__(self):
        self.rr = []; self.h1 = 0; self.h3 = 0; self.h10 = 0; self.n = 0
    def add(self, rank):
        self.rr.append(1.0 / rank); self.n += 1
        self.h1 += rank <= 1; self.h3 += rank <= 3; self.h10 += rank <= 10
    def row(self):
        return dict(MRR=np.mean(self.rr), H1=self.h1 / self.n, H3=self.h3 / self.n, H10=self.h10 / self.n, n=self.n)


def main():
    kg = load_core(drop_noise=True)
    N = kg.n_entities
    e = kg.edges.sort_values("time")
    test_lo = kg.splits["test"][0]
    by_week = {t: g[["subj", "rel", "obj"]].to_numpy() for t, g in e.groupby("time")}

    sro = defaultdict(lambda: defaultdict(int))   # (s,r) -> o -> count
    ors = defaultdict(lambda: defaultdict(int))   # (o,r) -> s -> count
    objpop = np.zeros(N); subjpop = np.zeros(N)
    meters = {b: Meter() for b in ("recurrence", "popularity", "backoff")}
    novel_obj = 0; total_obj = 0

    for t in range(kg.n_times):
        wk = by_week.get(t)
        if wk is not None and t >= test_lo:
            sa_o = np.sort(objpop); sa_s = np.sort(subjpop)
            for s, r, o in wk:
                go = sro.get((s, r), {}); gs = ors.get((o, r), {})
                # object prediction (s, r, ?)
                r_rec = group_rank(go, o, N); r_pop = pop_rank(sa_o, objpop[o], N)
                meters["recurrence"].add(r_rec); meters["popularity"].add(r_pop)
                meters["backoff"].add(r_rec if go.get(o, 0) > 0 else r_pop)
                total_obj += 1; novel_obj += go.get(o, 0) == 0
                # subject prediction (?, r, o)
                s_rec = group_rank(gs, s, N); s_pop = pop_rank(sa_s, subjpop[s], N)
                meters["recurrence"].add(s_rec); meters["popularity"].add(s_pop)
                meters["backoff"].add(s_rec if gs.get(s, 0) > 0 else s_pop)
        if wk is not None:
            for s, r, o in wk:
                sro[(s, r)][o] += 1; ors[(o, r)][s] += 1
                objpop[o] += 1; subjpop[s] += 1

    res = pd.DataFrame({b: m.row() for b, m in meters.items()}).T
    res = res[["MRR", "H1", "H3", "H10", "n"]]
    res.to_csv(OUT / "baseline_results.csv")
    print(f"test queries: {total_obj:,} obj + {total_obj:,} subj = {2*total_obj:,} ranked")
    print(f"novel object queries (never seen with (s,r)): {novel_obj/total_obj:.1%}\n")
    with pd.option_context("display.float_format", lambda v: f"{v:.4f}"):
        print(res.to_string())
    print(f"\nsaved {OUT/'baseline_results.csv'}")


if __name__ == "__main__":
    main()
