"""Bootstrap 95% confidence intervals on the link-prediction MRRs and the key pairwise gaps, so
the small differences we call '~=' or 'best' become defensible (paired bootstrap over test
directions). Methods: recurrence, ComplEx, ChronoBERT-2022/2024, and the recurrence backoffs.

Run: `.venv/Scripts/python -m dynamics.bootstrap_ci`
"""
import os
import sys
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from dynamics.loader import load_core
from dynamics.linkpred import group_rank
from dynamics import complex_kge

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
B = 1000
rng = np.random.default_rng(0)


def text_ranks(emb_path, kg, te, dim=256, epochs=40, lr=2e-3, batch=512):
    """Train projection + DistMult on frozen LM embeddings; return per-test-row obj/subj ranks."""
    Et = F.normalize(torch.tensor(np.load(emb_path).astype(np.float32), device=DEV), dim=1)
    proj = nn.Linear(Et.shape[1], dim, bias=False).to(DEV)
    Rel = nn.Parameter(torch.randn(kg.n_relations, dim, device=DEV) * 0.1)
    scale = nn.Parameter(torch.tensor(8.0, device=DEV))
    opt = torch.optim.Adam(list(proj.parameters()) + [Rel, scale], lr=lr)
    ce = nn.CrossEntropyLoss()
    H = lambda: F.normalize(proj(Et), dim=1)
    obj = lambda Hm, s, r: scale * ((Hm[s] * Rel[r]) @ Hm.t())
    sub = lambda Hm, r, o: scale * ((Hm[o] * Rel[r]) @ Hm.t())
    test_lo = kg.splits["test"][0]
    tr = kg.edges[kg.edges["time"] < test_lo].drop_duplicates(["subj", "rel", "obj"])
    s_tr = torch.tensor(tr["subj"].values, device=DEV)
    r_tr = torch.tensor(tr["rel"].values, device=DEV)
    o_tr = torch.tensor(tr["obj"].values, device=DEV)
    n = len(tr)
    for _ in range(epochs):
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]; s, r, o = s_tr[idx], r_tr[idx], o_tr[idx]
            Hm = H(); loss = ce(obj(Hm, s, r), o) + ce(sub(Hm, r, o), s)
            opt.zero_grad(); loss.backward(); opt.step()
    s_te = torch.tensor(te["subj"].values, device=DEV)
    r_te = torch.tensor(te["rel"].values, device=DEV)
    o_te = torch.tensor(te["obj"].values, device=DEV)
    with torch.no_grad():
        Hm = H(); oo, ss = [], []
        for i in range(0, len(te), 1024):
            sl = slice(i, i + 1024)
            oo.append(complex_kge.ranks(obj(Hm, s_te[sl], r_te[sl]), o_te[sl]).cpu())
            ss.append(complex_kge.ranks(sub(Hm, r_te[sl], o_te[sl]), s_te[sl]).cpu())
    return torch.cat(oo).numpy(), torch.cat(ss).numpy()


def main():
    torch.manual_seed(0)   # reproducibility (audit fix: text-scorer init/batch order were unseeded)
    kg = load_core(drop_noise=True)
    N = kg.n_entities; test_lo = kg.splits["test"][0]
    te = kg.edges[kg.edges["time"] >= test_lo]
    print("training ComplEx + ChronoBERT-2022 + ChronoBERT-2024 ...", flush=True)
    cm = complex_kge.train(kg, verbose=False); co, cs = complex_kge.test_ranks(cm, te)
    a2o, a2s = text_ranks(os.environ.get("EMB22", "data/dynamics/linkpred/emb_chrono.npy"), kg, te)
    a4o, a4s = text_ranks(os.environ.get("EMB24", "data/dynamics/linkpred/emb_chrono24.npy"), kg, te)
    key = list(zip(te["subj"], te["rel"], te["obj"], te["time"]))
    mk = lambda o, s: {(int(a), int(b), int(c), int(d)): (float(o[i]), float(s[i])) for i, (a, b, c, d) in enumerate(key)}
    cxd, d22, d24 = mk(co, cs), mk(a2o, a2s), mk(a4o, a4s)

    by_week = {t: g[["subj", "rel", "obj"]].to_numpy() for t, g in kg.edges.sort_values("time").groupby("time")}
    sro = defaultdict(lambda: defaultdict(int)); ors = defaultdict(lambda: defaultdict(int))
    rr = defaultdict(list); novel = []
    for t in range(kg.n_times):
        wk = by_week.get(t)
        if wk is not None and t >= test_lo:
            for s, r, o in wk:
                seen = sro.get((s, r), {}).get(o, 0) > 0
                ro = group_rank(sro.get((s, r), {}), o, N); rs = group_rank(ors.get((o, r), {}), s, N)
                co_, cs_ = cxd[(s, r, o, t)]; a2o_, a2s_ = d22[(s, r, o, t)]; a4o_, a4s_ = d24[(s, r, o, t)]
                for rec_r, cx_r, c22, c24 in ((ro, co_, a2o_, a4o_), (rs, cs_, a2s_, a4s_)):
                    rr["recurrence"].append(1 / rec_r); rr["ComplEx"].append(1 / cx_r)
                    rr["ChronoBERT-2022"].append(1 / c22); rr["ChronoBERT-2024"].append(1 / c24)
                    rr["backoff(rec->ComplEx)"].append(1 / (rec_r if seen else cx_r))
                    rr["backoff(rec->ChronoBERT)"].append(1 / (rec_r if seen else c22))
                    novel.append(not seen)
        if wk is not None:
            for s, r, o in wk:
                sro[(s, r)][o] += 1; ors[(o, r)][s] += 1

    rr = {m: np.array(v) for m, v in rr.items()}
    novel = np.array(novel); allmask = np.ones(len(novel), bool)

    def ci(vals, mask):
        pool = np.where(mask)[0]
        bs = np.array([vals[rng.choice(pool, pool.size)].mean() for _ in range(B)])
        return vals[pool].mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)

    def gap(a, b, mask):
        pool = np.where(mask)[0]
        bs = np.array([(a[s] - b[s]).mean() for s in (rng.choice(pool, pool.size) for _ in range(B))])
        return (a[pool] - b[pool]).mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)

    print(f"\n=== MRR with 95% bootstrap CI (B={B}), {len(novel):,} test directions ===")
    print(f"{'method':28} {'MRR all [95% CI]':28} {'MRR novel [95% CI]'}")
    for m in ("recurrence", "ComplEx", "ChronoBERT-2022", "ChronoBERT-2024",
              "backoff(rec->ComplEx)", "backoff(rec->ChronoBERT)"):
        ma, la, ha = ci(rr[m], allmask); mn, ln, hn = ci(rr[m], novel)
        print(f"{m:28} {ma:.4f} [{la:.4f},{ha:.4f}]   {mn:.4f} [{ln:.4f},{hn:.4f}]")

    print("\n=== key gaps (paired bootstrap; CI excluding 0 = significant) ===")
    tests = [
        ("ChronoBERT-2024 - 2022 (novel)  [lookahead]", "ChronoBERT-2024", "ChronoBERT-2022", novel),
        ("ChronoBERT-2022 - ComplEx (novel) [semantic]", "ChronoBERT-2022", "ComplEx", novel),
        ("backoff ChronoBERT - ComplEx (all) [new best]", "backoff(rec->ChronoBERT)", "backoff(rec->ComplEx)", allmask),
    ]
    for label, a, b, mask in tests:
        d, lo, hi = gap(rr[a], rr[b], mask)
        sig = "SIG" if (lo > 0 or hi < 0) else "n.s."
        print(f"  {label:46} {d:+.4f} [{lo:+.4f},{hi:+.4f}]  {sig}")


if __name__ == "__main__":
    main()
