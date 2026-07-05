"""Refinement: quantify how much entity fragmentation inflates the 72% novelty (and thus the
'forecasting-hard' claim). Apply a LIGHT surface-form canonicalisation (lowercase, strip corporate
suffixes/punctuation), merge entities sharing a key, then re-measure novelty + recurrence link-pred.
This is a crude merge (a lower bound on what full resolution would achieve), so the novelty drop it
shows is a *lower bound*. Run: `.venv/Scripts/python -m dynamics.resolve_check`
"""
import re
import sys
from collections import defaultdict
import pandas as pd

from dynamics.loader import load_core
from dynamics.linkpred import group_rank, Meter

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
SUFFIX = re.compile(r"\b(inc|corp|corporation|co|ltd|limited|llc|plc|group|holdings?|the|company)\b")


def keynorm(s):
    """Light canonical key: lowercase, strip punctuation and corporate suffixes."""
    s = re.sub(r"[.,'\"&]", "", str(s).lower().strip())
    key = re.sub(r"\s+", " ", SUFFIX.sub(" ", s)).strip()
    return key or s   # suffix-only names ("Co", "The Group") must NOT merge into one pseudo-entity


def eval_recurrence(edges, n_ent, n_times, test_lo):
    """Recurrence-baseline MRR on `edges` under the standard PIT sweep (counts from weeks < t,
    updated AFTER each week is scored). Returns (test novelty rate, MRR by all/recurring/novel)."""
    by_week = {t: g[["subj", "rel", "obj"]].to_numpy() for t, g in edges.groupby("time")}
    sro = defaultdict(lambda: defaultdict(int)); ors = defaultdict(lambda: defaultdict(int))
    meters = {g: Meter() for g in ("all", "recurring", "novel")}
    n_rec = n_nov = 0
    for t in range(n_times):
        wk = by_week.get(t)
        if wk is not None and t >= test_lo:
            for s, r, o in wk:
                seen = sro.get((s, r), {}).get(o, 0) > 0
                sub = "recurring" if seen else "novel"
                n_rec += seen; n_nov += not seen
                ro = group_rank(sro.get((s, r), {}), o, n_ent)
                rs = group_rank(ors.get((o, r), {}), s, n_ent)
                for g in ("all", sub):
                    meters[g].add(ro); meters[g].add(rs)
        if wk is not None:
            for s, r, o in wk:
                sro[(s, r)][o] += 1; ors[(o, r)][s] += 1
    nov = n_nov / max(1, n_rec + n_nov)
    return nov, {g: meters[g].row()["MRR"] for g in ("all", "recurring", "novel")}


def main():
    """Compare test novelty and recurrence MRR before/after the crude surface-form merge."""
    kg = load_core(drop_noise=True)
    test_lo = kg.splits["test"][0]
    nov0, mrr0 = eval_recurrence(kg.edges, kg.n_entities, kg.n_times, test_lo)

    keys = {i: keynorm(kg.id2name[i]) for i in range(kg.n_entities)}
    ukey = {k: j for j, k in enumerate(sorted(set(keys.values())))}
    old2new = {i: ukey[keys[i]] for i in range(kg.n_entities)}
    n_res = len(ukey)
    e = kg.edges.copy()
    e["subj"] = e["subj"].map(old2new); e["obj"] = e["obj"].map(old2new)
    e = e[e["subj"] != e["obj"]].drop_duplicates(["subj", "rel", "obj", "time"]).reset_index(drop=True)
    nov1, mrr1 = eval_recurrence(e, n_res, kg.n_times, test_lo)

    print(f"\n=== light entity resolution (crude, lower bound) ===")
    print(f"entities:        {kg.n_entities:,} -> {n_res:,}  ({100*n_res/kg.n_entities:.0f}%)")
    print(f"edge-instances:  {len(kg.edges):,} -> {len(e):,}")
    print(f"TEST novelty:    {nov0:.1%} -> {nov1:.1%}   (drop {100*(nov0-nov1):.1f} pts)")
    print(f"\nrecurrence MRR     unresolved   resolved")
    for g in ("all", "recurring", "novel"):
        print(f"  {g:10}       {mrr0[g]:.4f}      {mrr1[g]:.4f}")


if __name__ == "__main__":
    main()
