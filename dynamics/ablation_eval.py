"""Velocity-feature ablation: paired comparison of RE-GCN with vs without the three trailing
velocity features (log1p O, log1p E, clipped z), 2 arms x 3 seeds on the canonical dedup core.

Inputs: data/dynamics/linkpred/regcn_ranks_abl_{base,vel}_s{0,1,2}.npz written by
`dynamics.regcn --tag ...` (per-query test ranks in identical week order across arms, so the
per-query pairing is exact). Reports per-seed MRRs, then the paired per-query bootstrap gap
(vel - base) on seed-averaged reciprocal ranks, same resample for both arms (house method,
matching dynamics/bootstrap_ci.py: B=1000, percentile CI, rng seed 0).

Run: `.venv/Scripts/python -m dynamics.ablation_eval`
"""
import sys
from pathlib import Path
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
OUT = Path("data/dynamics/linkpred")
B = 1000
rng = np.random.default_rng(0)
SEEDS = (0, 1, 2)


def load_arm(arm):
    """Per-seed reciprocal-rank matrices [n_seeds, n_directions] plus the novel mask
    (both ranking directions of each test edge, novel flag shared by the two)."""
    rrs, novel = [], None
    for s in SEEDS:
        z = np.load(OUT / f"regcn_ranks_abl_{arm}_s{s}.npz")
        rr = np.concatenate([1.0 / z["ro"], 1.0 / z["rs"]])
        nv = np.concatenate([z["novel"], z["novel"]])
        if novel is None:
            novel = nv
        assert np.array_equal(novel, nv), f"novel mask mismatch {arm} s{s}"
        rrs.append(rr)
    return np.stack(rrs), novel


def main():
    base, novel = load_arm("base")
    vel, novel_v = load_arm("vel")
    assert np.array_equal(novel, novel_v), "arms disagree on novelty (different eval order?)"
    n = novel.size
    masks = {"all": np.ones(n, bool), "recurring": ~novel, "novel": novel}

    print(f"{n:,} test directions | novel {novel.mean():.1%} | seeds {list(SEEDS)} | B={B}\n")
    print(f"{'run':12} {'MRR all':>9} {'recurring':>10} {'novel':>8}")
    for arm, M in (("base", base), ("vel", vel)):
        for i, s in enumerate(SEEDS):
            print(f"{arm}_s{s:<9} {M[i].mean():9.4f} {M[i][~novel].mean():10.4f} {M[i][novel].mean():8.4f}")
        print(f"{arm + ' mean':12} {M.mean():9.4f} {M[:, ~novel].mean():10.4f} {M[:, novel].mean():8.4f}")
    spread = lambda M, m: M[:, m].mean(axis=1).max() - M[:, m].mean(axis=1).min()
    print(f"\nseed spread (max-min of per-seed MRR): base all {spread(base, masks['all']):.4f} "
          f"novel {spread(base, masks['novel']):.4f} | vel all {spread(vel, masks['all']):.4f} "
          f"novel {spread(vel, masks['novel']):.4f}")

    # paired gap on seed-averaged per-query RR: one resample indexes both arms
    a, b = vel.mean(axis=0), base.mean(axis=0)
    print("\n=== paired gap vel - base (95% bootstrap CI; CI excluding 0 = SIG) ===")
    for name, m in masks.items():
        pool = np.where(m)[0]
        bs = np.array([(a[s] - b[s]).mean() for s in (rng.choice(pool, pool.size) for _ in range(B))])
        d, lo, hi = (a[pool] - b[pool]).mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)
        sig = "SIG" if (lo > 0 or hi < 0) else "n.s."
        print(f"  {name:10} {d:+.4f} [{lo:+.4f},{hi:+.4f}]  {sig}")


if __name__ == "__main__":
    main()
