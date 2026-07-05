"""Lead-time evaluation of the walk-forward theme detector against a FROZEN gold list.

The gold list below is fixed BEFORE inspection of detector output beyond initial development, and
every event week is an externally documented public date (not derived from this corpus). For each
gold theme and each detector config (lenient/strict), we find the earliest lifeline whose members
match the theme's patterns and report LEAD = birth week - event week (negative = detected early).

Run: `.venv/Scripts/python -m dynamics.themes_eval`
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
OUT = Path("data/dynamics/themes")

# (name, documented public event week, match patterns [any, case-insensitive, substring on members])
GOLD = [
    ("COVID market crisis",   "2020-02-24", ["coronavirus", "covid"]),          # first crash week (S&P peak 02-19)
    ("Oil crash",             "2020-04-20", ["oil price", "exxon", "bp"]),      # negative WTI settle 04-20
    ("Vaccine race (OWS)",    "2020-05-15", ["warp speed", "novavax", "biontech", "inovio"]),  # OWS announced
    ("Vaccine efficacy",      "2020-11-09", ["pfizer", "moderna", "bnt162b2", "vaccine"]),     # Pfizer readout
    ("GameStop squeeze",      "2021-01-25", ["gamestop", "robinhood", "amc "]),  # peak squeeze week
    ("Russia-Ukraine war",    "2022-02-21", ["russia", "ukraine"]),              # invasion 02-24
    ("FTX / crypto collapse", "2022-11-07", ["ftx", "crypto", "bitcoin", "binance"]),  # FTX bankruptcy week
    ("AI / ChatGPT wave",     "2022-11-28", ["chatgpt", "openai", "artificial intelligence", " ai "]),  # ChatGPT launch 11-30
    ("SVB banking crisis",    "2023-03-06", ["silicon valley bank", "first republic", "fdic"]),  # SVB failed 03-10
    ("UAW strike",            "2023-09-11", ["uaw", "united auto workers"]),     # strike began 09-15
]


import re


def match_lifelines(lf, patterns):
    """Word-boundary matching (substring matching is unsafe: 'uaw' hits 'Huawei')."""
    m = lf["members"].str.lower()
    hit = pd.Series(False, index=lf.index)
    for p in patterns:
        hit |= m.str.contains(r"\b" + re.escape(p.strip()) + r"\b", regex=True)
    return lf[hit]


# a detection counts only if born within [event-13wk, event+8wk]: recurring TOPICS (Russia, crypto,
# AI) have lifelines years earlier that are different stories -- matching the earliest-ever lifeline
# would fabricate absurd 'leads'. Pre-window lifelines are reported as context, not detections.
PRE, POST = 13, 8


def main():
    modes = sys.argv[1:] or ["lenient", "strict"]   # evaluate specific --tag outputs if given
    for mode in modes:
        lf = pd.read_csv(OUT / f"theme_lifelines_{mode}.csv", parse_dates=["birth", "peak_wk"])
        print(f"\n=== {mode.upper()} config: lead time vs frozen gold list "
              f"(birth window event-{PRE}..+{POST}wk; negative lead = detected EARLY) ===")
        print(f"{'gold theme':24} {'event wk':>10}  {'birth':>10}  {'lead(wk)':>8}  {'status':>9}  members")
        leads, hits_n = [], 0
        for name, ew, pats in GOLD:
            ew = pd.Timestamp(ew)
            topical = match_lifelines(lf, pats)
            hits = topical[(topical["birth"] >= ew - pd.Timedelta(weeks=PRE)) &
                           (topical["birth"] <= ew + pd.Timedelta(weeks=POST))]
            if len(hits) == 0:
                pre = topical[topical["birth"] < ew - pd.Timedelta(weeks=PRE)]
                note = f"(pre-existing topical lifeline born {pre['birth'].max().date()})" if len(pre) else ""
                print(f"{name:24} {ew.date()!s:>10}  {'--':>10}  {'MISS':>8}  {note}")
                continue
            best = hits.sort_values("birth").iloc[0]
            lead = int((best["birth"] - ew).days / 7)
            leads.append(lead); hits_n += 1
            print(f"{name:24} {ew.date()!s:>10}  {best['birth'].date()!s:>10}  {lead:>+8}  "
                  f"{best['status']:>9}  {best['members'][:58]}")
        if leads:
            s = pd.Series(leads)
            print(f"\ndetected {hits_n}/{len(GOLD)}  |  median lead {s.median():+.0f}wk  "
                  f"|  early-or-on-time {(s <= 0).mean():.0%}")
        placebo(lf, hits_n)


def placebo(lf, observed, n_sims=100_000, seed=42):
    """PLACEBO CONTROL: how often would RANDOM event weeks be 'detected' under the same criterion?
    For each gold theme, the chance rate is the fraction of calendar weeks whose [-PRE, +POST]
    window contains a birth of a pattern-matching lifeline. The null distribution of total
    detections is Poisson-binomial over the 10 per-theme chance rates (Monte Carlo)."""
    rng = np.random.default_rng(seed)
    lo, hi = lf["birth"].min(), lf["birth"].max()
    grid = pd.date_range(lo, hi, freq="W-MON").values
    chance = []
    print(f"\n--- placebo control ({n_sims:,} sims over {len(grid)} candidate weeks) ---")
    for name, _, pats in GOLD:
        births = match_lifelines(lf, pats)["birth"].values
        if len(births) == 0:
            chance.append(0.0)
            print(f"  {name:24} chance {0:>5.1%}")
            continue
        d = (births[None, :] >= grid[:, None] - np.timedelta64(PRE, "W")) & \
            (births[None, :] <= grid[:, None] + np.timedelta64(POST, "W"))
        c = float(d.any(axis=1).mean())
        chance.append(c)
        print(f"  {name:24} chance {c:>5.1%}   ({len(births)} topical lifelines)")
    ch = np.array(chance)
    sims = (rng.random((n_sims, len(ch))) < ch).sum(axis=1)
    pval = float((sims >= observed).mean())
    print(f"expected detections by chance: {ch.sum():.1f}/{len(GOLD)}  |  observed: {observed}  "
          f"|  P(random >= observed) = {pval:.4g}")


if __name__ == "__main__":
    main()
