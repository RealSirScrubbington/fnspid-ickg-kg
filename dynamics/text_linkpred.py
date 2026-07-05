"""Text-embedding link-prediction baseline (lookahead-bias probe).

Frozen LM entity embeddings + a learned projection and DistMult relation operator, trained on the
pre-test split and evaluated on test with the same raw forecasting rank + novel/recurring split.
Comparing ChronoBERT (PIT-clean, no test-period knowledge) vs Qwen-14B (cutoff postdates the test
period) quantifies lookahead bias: a positive MRR gap = the larger model's future knowledge helping.

  python -m dynamics.text_linkpred --emb data/dynamics/linkpred/emb_chrono.npy --name ChronoBERT-2022
"""
import argparse
import sys
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from dynamics.loader import load_core
from dynamics.complex_kge import ranks
from dynamics.linkpred import Meter, group_rank

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ap = argparse.ArgumentParser()
ap.add_argument("--emb", required=True)
ap.add_argument("--name", default="text-emb")
ap.add_argument("--dim", type=int, default=256)
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--batch", type=int, default=512)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
torch.manual_seed(a.seed)   # reproducibility (audit fix: scorer init/batch order were unseeded)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

kg = load_core(drop_noise=True)
N, Rn = kg.n_entities, kg.n_relations
test_lo = kg.splits["test"][0]
Et = torch.tensor(np.load(a.emb).astype(np.float32), device=DEV)
Et = F.normalize(Et, dim=1)
d_lm = Et.shape[1]
print(f"{a.name}: embeddings {tuple(Et.shape)}", flush=True)


class Scorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(d_lm, a.dim, bias=False)
        self.R = nn.Parameter(torch.randn(Rn, a.dim) * 0.1)
        self.scale = nn.Parameter(torch.tensor(8.0))

    def H(self):
        return F.normalize(self.proj(Et), dim=1)

    def obj(self, H, s, r):
        return self.scale * ((H[s] * self.R[r]) @ H.t())

    def sub(self, H, r, o):
        return self.scale * ((H[o] * self.R[r]) @ H.t())


m = Scorer().to(DEV)
opt = torch.optim.Adam(m.parameters(), lr=a.lr)
ce = nn.CrossEntropyLoss()
tr = kg.edges[kg.edges["time"] < test_lo].drop_duplicates(["subj", "rel", "obj"])
s_tr = torch.tensor(tr["subj"].values, device=DEV)
r_tr = torch.tensor(tr["rel"].values, device=DEV)
o_tr = torch.tensor(tr["obj"].values, device=DEV)
n = len(tr)
for ep in range(a.epochs):
    perm = torch.randperm(n, device=DEV); tot = 0.0
    for i in range(0, n, a.batch):
        idx = perm[i:i + a.batch]; s, r, o = s_tr[idx], r_tr[idx], o_tr[idx]
        H = m.H()
        loss = ce(m.obj(H, s, r), o) + ce(m.sub(H, r, o), s)
        opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
    if ep % 5 == 0 or ep == a.epochs - 1:
        print(f"  epoch {ep:3} loss {tot/(n//a.batch+1):.3f}", flush=True)

m.eval()
te = kg.edges[kg.edges["time"] >= test_lo]
s_te = torch.tensor(te["subj"].values, device=DEV)
r_te = torch.tensor(te["rel"].values, device=DEV)
o_te = torch.tensor(te["obj"].values, device=DEV)
with torch.no_grad():
    H = m.H(); orank = []; srank = []
    for i in range(0, len(te), 1024):
        sl = slice(i, i + 1024)
        orank.append(ranks(m.obj(H, s_te[sl], r_te[sl]), o_te[sl]).cpu())
        srank.append(ranks(m.sub(H, r_te[sl], o_te[sl]), s_te[sl]).cpu())
    orank = torch.cat(orank).numpy(); srank = torch.cat(srank).numpy()
cx = {(int(s), int(r), int(o), int(t)): (float(oo), float(ss))
      for s, r, o, t, oo, ss in zip(te["subj"], te["rel"], te["obj"], te["time"], orank, srank)}

e = kg.edges.sort_values("time")
by_week = {t: g[["subj", "rel", "obj"]].to_numpy() for t, g in e.groupby("time")}
sro = defaultdict(lambda: defaultdict(int)); ors = defaultdict(lambda: defaultdict(int))
BK = f"backoff(rec->{a.name})"
meters = {(mm, g): Meter() for mm in (a.name, "recurrence", BK) for g in ("all", "recurring", "novel")}
for t in range(kg.n_times):
    wk = by_week.get(t)
    if wk is not None and t >= test_lo:
        for s, r, o in wk:
            seen = sro.get((s, r), {}).get(o, 0) > 0
            sub = "recurring" if seen else "novel"
            txt = cx[(int(s), int(r), int(o), int(t))]
            rec = (group_rank(sro.get((s, r), {}), o, N), group_rank(ors.get((o, r), {}), s, N))
            for nm, (po, ps) in ((a.name, txt), ("recurrence", rec), (BK, rec if seen else txt)):
                for g in ("all", sub):
                    meters[(nm, g)].add(po); meters[(nm, g)].add(ps)
    if wk is not None:
        for s, r, o in wk:
            sro[(s, r)][o] += 1; ors[(o, r)][s] += 1

rows = {}
for mm in (a.name, "recurrence", BK):
    rows[mm] = {f"MRR/{g}": meters[(mm, g)].row()["MRR"] for g in ("all", "recurring", "novel")}
    rows[mm]["H10/all"] = meters[(mm, "all")].row()["H10"]
with pd.option_context("display.float_format", lambda v: f"{v:.4f}"):
    print("\n" + pd.DataFrame(rows).T.to_string())
print("reference: ComplEx MRR/all 0.082 novel 0.025 | backoff(rec->ComplEx) 0.152")
