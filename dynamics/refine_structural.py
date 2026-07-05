"""Refinement 2: re-run velocity + burst on STRUCTURAL-only relations (excluding the 5 causal
'impact' relations the blinded test flagged as entity-dependent). If the event recovery (COVID,
Pfizer, GameStop, ...) survives at the right dates without the impact relations, the temporal
signal is text-anchored, not driven by potentially-hindsight causal attributions.

Run: `.venv/Scripts/python -m dynamics.refine_structural`
"""
import sys
import numpy as np
import pandas as pd

from dynamics.loader import load_core, IMPACT_RELATIONS
from dynamics.velocity import formation_events, entity_week_matrix, centered_sum, WIN
from dynamics.burst import kleinberg_bursts

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
KEY = ["Coronavirus", "Pfizer", "GameStop", "Inflation", "Boeing", "Bitcoin"]


def analyze(drop):
    kg = load_core(drop_noise=True, drop_relations=drop, verbose=True)
    ev = formation_events(kg)
    idx = pd.RangeIndex(kg.n_times)
    f = ev.groupby("time").size().reindex(idx, fill_value=0).astype(float)
    vel = f.rolling(WIN, min_periods=1, center=True).mean()
    gpeak = kg.date(int(vel.idxmax()))

    M = entity_week_matrix(ev, kg.n_entities, kg.n_times)
    roll = centered_sum(M, WIN)
    peak, onset = roll.max(1), M.argmax(1)
    res = {}
    for kname in KEY:
        ids = [i for i, n in kg.id2name.items() if str(n) == kname] or \
              [i for i, n in kg.id2name.items() if kname.lower() in str(n).lower()]
        if ids:
            best = max(ids, key=lambda i: M[i].sum())
            res[kname] = (int(peak[best]), kg.date(int(onset[best])), int(M[best].sum()))
        else:
            res[kname] = None

    gb = sorted(kleinberg_bursts(f.to_numpy()), key=lambda x: -x[2])[:3]
    bursts = [f"{kg.date(a)}->{kg.date(b)}" for a, b, _ in gb]
    return len(kg.edges), gpeak, res, bursts


print("\n#### ALL relations ####")
e_all, gp_all, r_all, b_all = analyze(None)
print("\n#### STRUCTURAL-only (impact relations removed) ####")
e_str, gp_str, r_str, b_str = analyze(IMPACT_RELATIONS)

print("\n================ COMPARISON ================")
print(f"edge-instances:       ALL {e_all:,}   STRUCTURAL {e_str:,} ({100*e_str/e_all:.0f}%)")
print(f"global velocity peak: ALL {gp_all}   STRUCTURAL {gp_str}")
print(f"top global bursts ALL: {b_all}")
print(f"top global bursts STR: {b_str}")
print("\nkey-entity onset (peak 8wk velocity, onset date, total formations):")
print(f"  {'entity':14} {'ALL onset':>12} {'pk':>4}   {'STRUCTURAL onset':>16} {'pk':>4}")
for k in KEY:
    a, srow = r_all[k], r_str[k]
    a_s = f"{a[1]} {a[0]:>4}" if a else "—"
    s_s = f"{srow[1]} {srow[0]:>4}" if srow else "—"
    print(f"  {k:14} {a_s:>17}   {s_s:>21}")
