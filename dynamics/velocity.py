"""Edge-formation velocity on the core KG.

Velocity = rate of NEW relationship formation (a 'formation' is the first week a distinct
(subj, rel, obj) appears). Computed at three levels:
  - global   : formations/week (centered smoothing) + acceleration (change in the rate)
  - relation : formations/week per relation type, and H2/H1 growth (which kinds are rising)
  - entity   : per-entity CENTERED rolling new-edge rate -> two leaderboards:
                 * absolute  (peak velocity)        -> the biggest formers (market hubs, big events)
                 * relative  (peak z-score / burst)  -> specific entities spiking above own baseline

Windows are CENTERED so an entity's peak date aligns with the event, not the window's trailing
edge. Templated-media noise is filtered via `load_core(drop_noise=True)`.

Run: `.venv/Scripts/python -m dynamics.velocity`
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # entity names carry non-cp1252 chars

from dynamics.loader import load_core

OUT = Path("data/dynamics/velocity")
OUT.mkdir(parents=True, exist_ok=True)
WIN = 8          # rolling window (weeks)
MIN_TOTAL = 20   # min formations to be eligible for the RELATIVE (burst) leaderboard


def formation_events(kg) -> pd.DataFrame:
    """One row per distinct relationship, at the week it first appeared."""
    return kg.edges.groupby(["subj", "rel", "obj"], as_index=False)["time"].min()


def entity_week_matrix(ev, n_ent, n_ts) -> np.ndarray:
    """Dense [entity x week] count of formations involving each entity (both endpoints)."""
    ent = np.concatenate([ev["subj"].to_numpy(), ev["obj"].to_numpy()])
    wk = np.concatenate([ev["time"].to_numpy(), ev["time"].to_numpy()])
    M = np.zeros((n_ent, n_ts), dtype=np.float64)
    np.add.at(M, (ent, wk), 1.0)
    return M


def centered_sum(M, win) -> np.ndarray:
    """Centered rolling sum over [t-win//2, t+win-win//2) along the time axis (vectorised)."""
    half = win // 2
    n = M.shape[1]
    csz = np.concatenate([np.zeros((M.shape[0], 1)), np.cumsum(M, axis=1)], axis=1)  # csz[:,k]=sum first k
    lo = np.clip(np.arange(n) - half, 0, n)
    hi = np.clip(np.arange(n) + (win - half), 0, n)
    return csz[:, hi] - csz[:, lo]


def main():
    kg = load_core(drop_noise=True)
    ev = formation_events(kg)
    idx = pd.RangeIndex(kg.n_times, name="time")

    # --- global velocity + acceleration (centered) ---
    f = ev.groupby("time").size().reindex(idx, fill_value=0).astype(float)
    vel = f.rolling(WIN, min_periods=1, center=True).mean()
    accel = vel.diff()

    # --- per-relation velocity / growth ---
    relv = ev.groupby(["time", "rel"]).size().unstack(fill_value=0).reindex(idx, fill_value=0)
    relv.columns = [kg.id2rel[c] for c in relv.columns]
    h = kg.n_times // 2
    growth = pd.Series({c: relv[c][h:].sum() / max(1, relv[c][:h].sum()) for c in relv.columns}).sort_values(ascending=False)

    # --- per-entity velocity (centered, vectorised) ---
    M = entity_week_matrix(ev, kg.n_entities, kg.n_times)
    roll = centered_sum(M, WIN)
    total = M.sum(axis=1)
    peak = roll.max(axis=1)
    peak_wk = M.argmax(axis=1)        # onset = biggest single formation week (aligns date to the event)
    # baseline mu/sd EXCLUDING the window around each entity's peak (audit fix: including the burst
    # itself contaminates the baseline -- deflates z for genuinely bursty entities)
    roll_pk = roll.argmax(axis=1)     # audit fix: exclude around the ROLLING peak used in the z
    mask = np.ones_like(roll, dtype=bool)    # numerator (masking the single-week peak left the
    for i, pk in enumerate(roll_pk):         # rolling burst inside the baseline for 36% of entities)
        mask[i, max(0, int(pk) - WIN):int(pk) + WIN + 1] = False
    cnt = np.maximum(mask.sum(axis=1), 1)
    mu = (roll * mask).sum(axis=1) / cnt
    sd = np.sqrt(np.maximum(((roll - mu[:, None]) ** 2 * mask).sum(axis=1) / cnt, 0.0))
    # sd floor of 1 formation: entities whose ONLY activity is the peak itself (debut spikes) have a
    # zero out-of-peak baseline; without the floor their z diverges
    burst_z = (peak - mu) / np.maximum(sd, 1.0)      # how sharply the peak exceeds the entity's own baseline

    ent_ids = np.arange(kg.n_entities)
    lb = pd.DataFrame({
        "entity": [kg.id2name[i] for i in ent_ids],
        "type": [kg.id2type[i] for i in ent_ids],
        "total_formations": total.astype(int),
        "peak_velocity": peak.astype(int),
        "peak_date": [kg.date(int(w)) for w in peak_wk],
        "burst_z": burst_z.round(2),
    })
    lb.sort_values("peak_velocity", ascending=False).to_csv(OUT / "velocity_entities.csv", index=False)

    print(f"\n=== edge-formation velocity (centered {WIN}-wk, noise-filtered) ===")
    print(f"global formations/wk: mean {f.mean():.0f} | peak velocity {vel.max():.0f} "
          f"at {kg.date(int(vel.idxmax()))} | max accel {accel.max():.0f}/wk")
    print("\nrelation H2/H1 growth (rising kinds):")
    for c, g in growth.head(5).items():
        print(f"  {c:20} {int(relv[c].sum()):>6} total  x{g:.2f}")

    print("\nABSOLUTE — biggest formers (peak velocity):")
    for _, r in lb.sort_values("peak_velocity", ascending=False).head(12).iterrows():
        print(f"  {r.peak_velocity:>4}  {r.entity[:32]:32} {r.type:13} peak {r.peak_date}")

    rel_lb = lb[lb.total_formations >= MIN_TOTAL].sort_values("burst_z", ascending=False)
    print(f"\nRELATIVE — sharpest bursts (peak z-score, total>= {MIN_TOTAL}):")
    for _, r in rel_lb.head(12).iterrows():
        print(f"  z={r.burst_z:>5}  {r.entity[:30]:30} {r.type:13} peak {r.peak_date}  (pk {r.peak_velocity})")

    # --- figure: global vel/accel + top relative-burst trajectories ---
    x = idx.values
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    a = axes[0]
    a.plot(x, vel, color="#d93025", label=f"velocity ({WIN}-wk)")
    a.set_ylabel("formations / wk", color="#d93025"); a.tick_params(axis="y", labelcolor="#d93025")
    a2 = a.twinx(); a2.plot(x, accel, color="#1a73e8", lw=0.8, alpha=0.7)
    a2.axhline(0, color="#888", lw=0.5); a2.set_ylabel("Δ velocity", color="#1a73e8")
    a2.tick_params(axis="y", labelcolor="#1a73e8")
    a.set_title("Global edge-formation velocity & acceleration"); a.set_xlabel("week")

    a = axes[1]
    for ent in rel_lb.head(6).index:
        a.plot(x, roll[ent], lw=1.1, label=kg.id2name[ent][:20])
    a.set_title(f"Sharpest-burst entity trajectories (centered {WIN}-wk)"); a.set_xlabel("week")
    a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "velocity_overview.png", dpi=110)
    print(f"\nsaved {OUT/'velocity_entities.csv'} and {OUT/'velocity_overview.png'}")


if __name__ == "__main__":
    main()
