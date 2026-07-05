"""Systematic event-recovery scoring (replaces cherry-picked anecdotes).

A curated list of major 2017-2023 financial events -- compiled from general financial history,
*independently* of which entities our pipeline flagged -- is matched against the detection output:
  - RECALL: does the entity's Kleinberg burst interval overlap the event week (+/- W)?
  - TIMING: signed error between the entity's peak-formation week and the event week.
  - GLOBAL: do gold-event weeks have higher global formation velocity than random weeks (permutation)?

Run: `.venv/Scripts/python -m dynamics.event_validate`
"""
import sys
import numpy as np
import pandas as pd

from dynamics.loader import load_core
from dynamics.velocity import formation_events, entity_week_matrix
from dynamics.burst import kleinberg_bursts, cusum_bursts, bocd_bursts

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
rng = np.random.default_rng(0)
W = 3  # match window (weeks)

GOLD = [  # (event_date, KG entity name, label) -- listed independent of our results
    ("2018-10-29", "Aurora Cannabis", "cannabis boom"),
    ("2019-03-13", "Boeing", "737 MAX grounding"),
    ("2019-11-12", "Disney", "Disney+ launch"),
    ("2020-03-16", "Coronavirus", "COVID crash"),
    ("2020-04-20", "Crude Oil", "negative oil price"),
    ("2020-11-09", "Pfizer", "vaccine readout"),
    ("2021-01-27", "GameStop", "short squeeze"),
    ("2021-05-08", "Dogecoin", "Doge mania"),
    ("2021-11-01", "Meta Platforms", "Meta rebrand"),
    ("2022-02-24", "Russia", "Ukraine invasion"),
    ("2022-06-15", "Inflation", "Fed 75bp / CPI peak"),
    ("2022-07-27", "West Virginia", "Manchin / IRA deal"),
    ("2022-11-11", "FTX", "FTX collapse"),
    ("2023-03-10", "Silicon Valley Bank", "SVB collapse"),
    ("2023-05-25", "Nvidia", "AI / datacenter boom"),
    ("2023-09-15", "UAW", "auto-workers strike"),
    ("2023-10-07", "Gold", "Israel-Hamas safe haven"),
    ("2023-11-01", "Magnificent Seven", "AI mega-cap concept"),
]


def main():
    kg = load_core(drop_noise=True)
    ev = formation_events(kg)
    M = entity_week_matrix(ev, kg.n_entities, kg.n_times)
    gvel = (ev.groupby("time").size().reindex(pd.RangeIndex(kg.n_times), fill_value=0)
            .rolling(8, center=True, min_periods=1).mean().to_numpy())

    def event_week(dstr):
        d = pd.Timestamp(dstr)
        return int(min(range(kg.n_times), key=lambda t: abs((kg.times[t] - d).days)))

    def find_entity(name):
        ids = [i for i, n in kg.id2name.items() if str(n) == name] or \
              [i for i, n in kg.id2name.items() if name.lower() in str(n).lower()]
        return max(ids, key=lambda i: M[i].sum()) if ids else None

    DET = [("Kleinberg", kleinberg_bursts), ("CUSUM", cusum_bursts), ("BOCD", bocd_bursts)]
    rows = []; in_kg = 0; hits = {n: 0 for n, _ in DET}; errs = []
    cover = {n: [] for n, _ in DET}   # per-event dilated flagged-week share = chance of a random hit
    for dstr, ename, label in GOLD:
        ew = event_week(dstr)
        eid = find_entity(ename)
        if eid is None:
            rows.append((dstr, ename[:20], label[:20], "—", ["not in KG"] * len(DET))); continue
        in_kg += 1
        marks = []
        for dname, fn in DET:
            bursts = fn(M[eid])
            overlap = any(a - W <= ew <= b + W for a, b, _ in bursts)
            hits[dname] += overlap
            marks.append("HIT" if overlap else "miss")
            flagged = set()
            for a2, b2, _ in bursts:   # detector's flagged weeks, dilated by the +/-W hit window
                flagged.update(range(max(0, a2 - W), min(kg.n_times, b2 + W + 1)))
            cover[dname].append(len(flagged) / kg.n_times)
        lo, hi = max(0, ew - 8), min(kg.n_times, ew + 9)
        onset = lo + int(M[eid, lo:hi].argmax()); err = onset - ew   # local peak within +/-8wk of event
        if marks[0] == "HIT":
            errs.append(abs(err))
        rows.append((dstr, ename[:20], label[:20], f"{err:+d}w", marks))

    print(f"=== systematic event recovery ({len(GOLD)} gold events, window +/-{W} wk, "
          f"{len(DET)} detectors) ===")
    print(f"{'event date':11} {'entity':21} {'label':21} {'timing':>7} {'Klein':>6} {'CUSUM':>6} {'BOCD':>6}")
    for r in rows:
        m = r[4] if len(r[4]) == len(DET) else [r[4][0]] * len(DET)
        print(f"{r[0]:11} {r[1]:21} {r[2]:21} {r[3]:>7} {m[0]:>6} {m[1]:>6} {m[2]:>6}")
    print(f"\nin-KG events: {in_kg}/{len(GOLD)}  ({len(GOLD)-in_kg} not covered by the US-nasdaq corpus)")
    # chance = mean dilated flagged-week share: the recall a detector would get on RANDOM event
    # weeks. Wide-interval detectors (BOCD regimes) buy recall with coverage -- recall must be
    # read against this, not in isolation (audit fix: raw recall comparison was not fair).
    for dname, _ in DET:
        ch = float(np.mean(cover[dname]))
        print(f"RECALL {dname:10} (burst overlaps event +/-{W}wk): {hits[dname]}/{in_kg} = "
              f"{100*hits[dname]/in_kg:.0f}%  | chance (dilated coverage) {100*ch:.0f}%  "
              f"| excess {100*(hits[dname]/in_kg - ch):+.0f}pp")
    hits = hits["Kleinberg"]
    errs = np.array(errs)
    print(f"TIMING |weeks| (hits, local peak): median {np.median(errs):.0f}, "
          f"within 1wk {100*np.mean(errs <= 1):.0f}%, within 3wk {100*np.mean(errs <= 3):.0f}%")

    ews = [event_week(d) for d, _, _ in GOLD]
    obs = np.mean([gvel[w] for w in ews])
    null = np.array([gvel[rng.integers(0, kg.n_times, len(ews))].mean() for _ in range(10000)])
    print(f"\nglobal velocity @ event weeks: {obs:.0f} vs random-week null mean {null.mean():.0f} "
          f"(p = {np.mean(null >= obs):.4f})")


if __name__ == "__main__":
    main()
