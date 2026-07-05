"""Burst detection on the core KG formation series -- THREE independent detectors.

(1) Kleinberg, "Bursty and Hierarchical Structure in Streams" (2002). Discrete 2-state version:
each week is in a baseline state (rate p0 = mean formations/wk) or a burst state (rate s*p0).
A Viterbi pass finds the min-cost state path, trading emission likelihood against a transition
cost gamma*ln(T) to enter a burst.

(2) CUSUM (Page, "Continuous Inspection Schemes", Biometrika 1954). A one-sided upper cumulative-
sum control chart on the standardised series: C_t = max(0, C_{t-1} + (x_t-mu)/sigma - k); a burst
is a run where C_t exceeds the decision interval h (a sustained upward shift in formation rate).

(3) BOCD (Adams & MacKay, "Bayesian Online Changepoint Detection", 2007), Gamma-Poisson conjugate:
an exact online run-length posterior over "weeks since the last changepoint"; a changepoint is
declared where the MAP run length resets. Online, hence walk-forward by construction. O(T^2) per
series, so it runs on the global and gold-event series; the cheap detectors cover the catalogue.

All return burst INTERVALS (start, end, intensity = excess formations over baseline), so they are
directly comparable. Running several is a robustness check: where they agree, detected events are
not an artifact of one algorithm. Applied to the global / per-relation / per-entity series.

Run: `.venv/Scripts/python -m dynamics.burst`
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dynamics.loader import load_core
from dynamics.velocity import formation_events, entity_week_matrix

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = Path("data/dynamics/burst")
OUT.mkdir(parents=True, exist_ok=True)
S = 2.0        # Kleinberg burst-state rate multiplier
GAMMA = 1.0    # Kleinberg transition cost weight
MIN_TOTAL = 30 # min formations for an entity to be eligible
K_SLACK = 0.5  # CUSUM reference value / allowance (sigma units; detects ~1-sigma shifts)
H_THRESH = 5.0 # CUSUM decision interval / alarm threshold (sigma units; ~textbook low-false-alarm)


def kleinberg_bursts(counts, s=S, gamma=GAMMA):
    """Return [(start, end, intensity)] burst intervals via a 2-state Kleinberg/Viterbi pass."""
    c = np.asarray(counts, dtype=float)
    T = len(c); N = c.sum()
    if T == 0 or N <= 0:
        return []
    p0 = N / T
    rates = (p0, p0 * s)
    emit = lambda q, t: rates[q] - c[t] * np.log(rates[q])     # Poisson NLL (drop constant log c!)
    trans = gamma * np.log(max(T, 2))

    cost = np.full((2, T), np.inf); back = np.zeros((2, T), int)
    cost[0, 0] = emit(0, 0); cost[1, 0] = emit(1, 0) + trans
    for t in range(1, T):
        for q in (0, 1):
            stay = cost[q, t - 1]
            jump = cost[1 - q, t - 1] + (trans if q == 1 else 0)
            prev = q if stay <= jump else 1 - q
            cost[q, t] = min(stay, jump) + emit(q, t); back[q, t] = prev
    q = int(np.argmin(cost[:, T - 1])); states = [q]
    for t in range(T - 1, 0, -1):
        q = back[q, t]; states.append(q)
    states = states[::-1]

    bursts, i = [], 0
    while i < T:
        if states[i] == 1:
            j = i
            while j + 1 < T and states[j + 1] == 1:
                j += 1
            bursts.append((i, j, float(c[i:j + 1].sum() - p0 * (j - i + 1))))
            i = j + 1
        else:
            i += 1
    return bursts


def cusum_bursts(counts, k=K_SLACK, h=H_THRESH):
    """One-sided upper-CUSUM burst intervals (Page 1954). A burst is a run of the standardised
    cumulative sum C_t = max(0, C_{t-1} + (x_t-mu)/sigma - k) that reaches the decision interval h
    (a confirmed upward shift), spanning from where C first rose above 0 to where it returns to 0.
    Returns [(start, end, intensity = excess formations over baseline)]."""
    x = np.asarray(counts, dtype=float)
    T = len(x)
    if T == 0:
        return []
    mu, sd = x.mean(), x.std() + 1e-9
    z = (x - mu) / sd
    C = 0.0; bursts = []; start = None; alarmed = False
    for t in range(T):
        C = max(0.0, C + z[t] - k)
        if C > 0 and start is None:
            start = t
        if start is not None and C >= h:
            alarmed = True
        if C == 0 and start is not None:
            if alarmed:
                bursts.append((start, t - 1, float((x[start:t] - mu).sum())))
            start, alarmed = None, False
    if start is not None and alarmed:
        bursts.append((start, T - 1, float((x[start:] - mu).sum())))
    return bursts


def bocd_bursts(counts, hazard=1 / 100.0, a0=1.0, b0=1.0):
    """Bayesian online changepoint detection (Adams & MacKay 2007), Gamma-Poisson conjugate.

    Maintains the exact run-length posterior online: each week, the posterior predictive of the
    count under every 'weeks since last changepoint' hypothesis is negative binomial under that
    run's accumulated Gamma posterior; probability mass moves to run length 0 via a constant
    hazard. A changepoint is declared where the MAP run length resets, back-dated to the inferred
    onset t - run_length. Segments between changepoints whose mean rate exceeds the series
    baseline are returned as (start, end, intensity) bursts. NOTE: the run-length posterior is
    computed online, but interval extraction (like Kleinberg's and CUSUM's baselines) uses the
    full-series mean -- all three are retrospective descriptive tools; the walk-forward system is
    themes.py. BOCD intervals are REGIME-scale (whole elevated segments), systematically wider
    than Kleinberg's discrete bursts -- overlap-based recall comparisons must therefore be read
    alongside each detector's flagged-week coverage (see event_validate)."""
    from scipy.stats import nbinom
    x = np.asarray(counts, dtype=float)
    T = len(x)
    if T == 0 or x.sum() <= 0:
        return []
    r = np.array([1.0])                       # run-length posterior
    a, b = np.array([a0]), np.array([b0])     # Gamma posterior per run-length hypothesis
    map_rl = np.zeros(T, dtype=int)
    for t in range(T):
        # log-space predictive (audit fix: pmf underflows to exact 0 for counts ~>1000, which
        # would permanently collapse the posterior); the common scale cancels in normalisation
        lp = nbinom.logpmf(x[t], a, b / (b + 1.0))
        pred = np.exp(lp - lp.max())
        cp = float((r * pred).sum()) * hazard
        r = np.concatenate([[cp], r * pred * (1.0 - hazard)])
        r /= max(r.sum(), 1e-300)
        a = np.concatenate([[a0], a + x[t]])
        b = np.concatenate([[b0], b + 1.0])
        map_rl[t] = int(r.argmax())
    # a MAP reset detected at week t implies the change happened at t - map_rl[t] (audit fix:
    # dating the boundary at the DETECTION week placed onsets up to ~15 weeks late); the set
    # de-duplicates plateau re-detections of the same onset
    cps = sorted({t - int(map_rl[t]) for t in range(1, T) if map_rl[t] < map_rl[t - 1] + 1})
    cps = [c for c in cps if 0 < c < T]
    bounds = [0] + cps + [T]
    mu = x.mean()
    bursts = []
    for s, e in zip(bounds[:-1], bounds[1:]):
        if e > s and x[s:e].mean() > mu:
            if bursts and s == bursts[-1][1] + 1:                      # merge adjacent elevated segments
                ps, _, pi = bursts[-1]
                bursts[-1] = (ps, e - 1, pi + float((x[s:e] - mu).sum()))
            else:
                bursts.append((s, e - 1, float((x[s:e] - mu).sum())))
    return bursts


def main():
    """Run all three detectors on the global formation series (with pairwise week-level
    agreement), Kleinberg on each relation and each eligible entity (burst_catalog.csv),
    and save the global-burst figure."""
    kg = load_core(drop_noise=True)
    ev = formation_events(kg)
    idx = pd.RangeIndex(kg.n_times, name="time")
    f = ev.groupby("time").size().reindex(idx, fill_value=0).to_numpy()

    # --- global bursts ---
    gb = kleinberg_bursts(f)
    print("=== GLOBAL formation bursts ===")
    for a, b, inten in sorted(gb, key=lambda x: -x[2]):
        print(f"  {kg.date(a)} -> {kg.date(b)}  ({b-a+1:>2} wk)  intensity {inten:>6.0f}")

    # --- CUSUM global bursts + agreement with Kleinberg (robustness check) ---
    cb = cusum_bursts(f)
    print(f"\n=== GLOBAL formation bursts: CUSUM (k={K_SLACK}, h={H_THRESH}) ===")
    for a, b, inten in sorted(cb, key=lambda x: -x[2]):
        print(f"  {kg.date(a)} -> {kg.date(b)}  ({b-a+1:>2} wk)  intensity {inten:>6.0f}")
    kset = {w for a, b, _ in gb for w in range(a, b + 1)}
    cset = {w for a, b, _ in cb for w in range(a, b + 1)}
    inter, union = len(kset & cset), len(kset | cset)
    print(f"\nKleinberg vs CUSUM week-level agreement: Jaccard {inter/max(1,union):.2f}  "
          f"(Kleinberg {len(kset)} wk, CUSUM {len(cset)} wk, overlap {inter} wk)")

    # --- BOCD global changepoint segments + three-way agreement ---
    bb = bocd_bursts(f)
    print("\n=== GLOBAL formation bursts: BOCD (Adams-MacKay, Gamma-Poisson, hazard 1/100) ===")
    for a, b, inten in sorted(bb, key=lambda x: -x[2]):
        print(f"  {kg.date(a)} -> {kg.date(b)}  ({b-a+1:>2} wk)  intensity {inten:>6.0f}")
    bset = {w for a, b, _ in bb for w in range(a, b + 1)}
    for name, other in (("CUSUM", cset), ("BOCD", bset)):
        nest = len(kset & other) / max(1, len(kset))
        print(f"Kleinberg burst-weeks nested in {name}: {nest:.0%}")
    print(f"CUSUM vs BOCD Jaccard: {len(cset & bset)/max(1, len(cset | bset)):.2f}")

    # --- per-relation bursts ---
    relv = ev.groupby(["time", "rel"]).size().unstack(fill_value=0).reindex(idx, fill_value=0)
    print("\n=== per-relation top burst (by intensity) ===")
    for r in relv.columns:
        rb = kleinberg_bursts(relv[r].to_numpy())
        if rb:
            a, b, inten = max(rb, key=lambda x: x[2])
            print(f"  {kg.id2rel[r]:20} {kg.date(a)} -> {kg.date(b)}  intensity {inten:>5.0f}")

    # --- per-entity burst catalog ---
    M = entity_week_matrix(ev, kg.n_entities, kg.n_times)
    totals = M.sum(axis=1)
    rows = []
    for e in np.where(totals >= MIN_TOTAL)[0]:
        for a, b, inten in kleinberg_bursts(M[e]):
            pk = a + int(M[e, a:b + 1].argmax())
            rows.append((kg.id2name[e], kg.id2type[e], kg.date(a), kg.date(b),
                         b - a + 1, round(inten, 1), kg.date(pk)))
    cat = pd.DataFrame(rows, columns=["entity", "type", "start", "end", "weeks", "intensity", "peak"])
    cat = cat.sort_values("intensity", ascending=False).reset_index(drop=True)
    cat.to_csv(OUT / "burst_catalog.csv", index=False)

    print(f"\n=== entity burst catalog: {len(cat):,} bursts over "
          f"{cat.entity.nunique():,} entities (>= {MIN_TOTAL} formations) ===")
    print("top 18 by intensity:")
    for _, r in cat.head(18).iterrows():
        print(f"  {r.intensity:>5.0f}  {r.entity[:30]:30} {r.type:13} {r.start} -> {r.end} ({r.weeks}wk)")

    # --- figure: global series with burst intervals shaded ---
    fig, ax = plt.subplots(figsize=(15, 4.5))
    x = idx.values
    ax.plot(x, f, color="#1a73e8", lw=0.9)
    for a, b, _ in gb:
        ax.axvspan(a, b, color="#d93025", alpha=0.16)
    for a, b, _ in cb:
        ax.axvspan(a, b, facecolor="none", edgecolor="#188038", lw=1.1, hatch="///")
    from matplotlib.patches import Patch
    ax.legend(handles=[plt.Line2D([], [], color="#1a73e8", label="formations/wk"),
                       Patch(facecolor="#d93025", alpha=0.3, label="Kleinberg burst"),
                       Patch(facecolor="none", edgecolor="#188038", hatch="///", label="CUSUM burst")],
              fontsize=8, loc="upper left")
    ax.set_title("Global formation bursts: Kleinberg (shaded) vs CUSUM (hatched)")
    ax.set_xlabel("week")
    fig.tight_layout(); fig.savefig(OUT / "global_bursts.png", dpi=110)
    print(f"\nsaved {OUT/'burst_catalog.csv'} and {OUT/'global_bursts.png'}")


if __name__ == "__main__":
    main()
