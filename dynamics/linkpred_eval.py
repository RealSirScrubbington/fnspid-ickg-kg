"""Unified temporal link-prediction eval: every method scored on the SAME per-test-quad set,
split by whether the (s,r,o) is RECURRING (seen in history < t) or NOVEL, plus two hybrids.

This is the analysis that justifies each baseline: recurrence should dominate on recurring
edges, ComplEx should lead among single methods on novel edges, and `recurrence -> ComplEx`
should be the strongest overall.

Run: `.venv/Scripts/python -m dynamics.linkpred_eval`
"""
from pathlib import Path
import sys
from collections import defaultdict
import numpy as np
import pandas as pd

from dynamics.loader import load_core
from dynamics.linkpred import group_rank, pop_rank, Meter
from dynamics.complex_kge import train, test_triples, test_ranks

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
OUT = Path("data/dynamics/linkpred")
OUT.mkdir(parents=True, exist_ok=True)
METHODS = ["recurrence", "popularity", "ComplEx", "backoff(pop)", "backoff(ComplEx)"]
SUBSETS = ["all", "recurring", "novel"]


def main():
    kg = load_core(drop_noise=True)
    N = kg.n_entities
    test_lo = kg.splits["test"][0]
    e = kg.edges.sort_values("time")

    model = train(kg)
    te = test_triples(kg)
    orank, srank = test_ranks(model, te)
    cx = {(int(s), int(r), int(o), int(t)): (float(a), float(b))
          for s, r, o, t, a, b in zip(te["subj"], te["rel"], te["obj"], te["time"], orank, srank)}

    by_week = {t: g[["subj", "rel", "obj"]].to_numpy() for t, g in e.groupby("time")}
    sro = defaultdict(lambda: defaultdict(int)); ors = defaultdict(lambda: defaultdict(int))
    objpop = np.zeros(N); subjpop = np.zeros(N)
    meters = {(m, s): Meter() for m in METHODS for s in SUBSETS}
    n_rec = n_nov = 0

    for t in range(kg.n_times):
        wk = by_week.get(t)
        if wk is not None and t >= test_lo:
            sa_o = np.sort(objpop); sa_s = np.sort(subjpop)
            for s, r, o in wk:
                go = sro.get((s, r), {}); gs = ors.get((o, r), {})
                seen = go.get(o, 0) > 0
                sub = "recurring" if seen else "novel"
                n_rec += seen; n_nov += not seen
                rec = (group_rank(go, o, N), group_rank(gs, s, N))
                pop = (pop_rank(sa_o, objpop[o], N), pop_rank(sa_s, subjpop[s], N))
                cxo, cxs = cx[(int(s), int(r), int(o), int(t))]
                pairs = {
                    "recurrence": rec,
                    "popularity": pop,
                    "ComplEx": (cxo, cxs),
                    "backoff(pop)": rec if seen else pop,
                    "backoff(ComplEx)": rec if seen else (cxo, cxs),
                }
                for m, (ro, rs) in pairs.items():
                    for grp in ("all", sub):
                        meters[(m, grp)].add(ro); meters[(m, grp)].add(rs)
        if wk is not None:
            for s, r, o in wk:
                sro[(s, r)][o] += 1; ors[(o, r)][s] += 1
                objpop[o] += 1; subjpop[s] += 1

    rows = {}
    for m in METHODS:
        row = {f"MRR/{g}": meters[(m, g)].row()["MRR"] for g in SUBSETS}
        row["H10/all"] = meters[(m, "all")].row()["H10"]
        rows[m] = row
    tbl = pd.DataFrame(rows).T
    tbl.to_csv(OUT / "unified_results.csv")
    tot = n_rec + n_nov
    print(f"\ntest queries (per direction): {tot:,}  | recurring {n_rec/tot:.1%}  novel {n_nov/tot:.1%}")
    with pd.option_context("display.float_format", lambda v: f"{v:.4f}"):
        print("\n" + tbl.to_string())
    print(f"\nsaved {OUT/'unified_results.csv'}")


if __name__ == "__main__":
    main()
