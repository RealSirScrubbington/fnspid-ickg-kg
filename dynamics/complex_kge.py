"""ComplEx structural baseline for temporal link prediction.

A static KG embedding trained on all PRE-TEST edges (weeks 0-309, PIT-clean) and evaluated on
the test weeks with the same raw forecasting ranking as the non-neural baselines. Unlike
recurrence/popularity it scores edge *plausibility*, so it can rank NOVEL edges (72% of the
test set). 1-N (full-softmax) training on GPU. `train()`/`test_ranks()` are reused by
`linkpred_eval` for the unified novel-vs-recurring comparison.

Run: `.venv/Scripts/python -m dynamics.complex_kge`
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from dynamics.loader import load_core

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
OUT = Path("data/dynamics/linkpred")
DIM, EPOCHS, BATCH, LR, REG = 128, 25, 512, 1e-3, 1e-6
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class ComplEx(nn.Module):
    """ComplEx (Trouillon et al. 2016) with full-vocabulary (1-N) scoring heads."""
    def __init__(self, n_ent, n_rel, d):
        super().__init__()
        self.Ere = nn.Embedding(n_ent, d); self.Eim = nn.Embedding(n_ent, d)
        self.Rre = nn.Embedding(n_rel, d); self.Rim = nn.Embedding(n_rel, d)
        for emb in (self.Ere, self.Eim, self.Rre, self.Rim):
            nn.init.xavier_uniform_(emb.weight)

    def obj_scores(self, s, r):                       # score (s, r, ?) over all entities -> [B, N]
        a = self.Ere(s) * self.Rre(r) - self.Eim(s) * self.Rim(r)
        b = self.Ere(s) * self.Rim(r) + self.Eim(s) * self.Rre(r)
        return a @ self.Ere.weight.T + b @ self.Eim.weight.T

    def subj_scores(self, r, o):                      # score (?, r, o) over all entities -> [B, N]
        c = self.Rre(r) * self.Ere(o) + self.Rim(r) * self.Eim(o)
        e = self.Rre(r) * self.Eim(o) - self.Rim(r) * self.Ere(o)
        return c @ self.Ere.weight.T + e @ self.Eim.weight.T


def ranks(scores, true):
    """Raw average rank of the true entity in each row of `scores` ([B, N])."""
    ts = scores.gather(1, true.view(-1, 1))
    return (scores > ts).sum(1) + ((scores == ts).sum(1) - 1) / 2 + 1


def metrics(rk):
    """MRR and Hits@1/3/10 summary dict from a tensor of ranks."""
    rk = rk.float()
    return dict(MRR=(1 / rk).mean().item(), H1=(rk <= 1).float().mean().item(),
                H3=(rk <= 3).float().mean().item(), H10=(rk <= 10).float().mean().item(), n=rk.numel())


def train(kg, epochs=EPOCHS, verbose=True, seed=0):
    """Train ComplEx on all edges with time < test_lo; return the fitted model."""
    torch.manual_seed(seed)   # reproducibility (audit fix: init/batch order were unseeded)
    N, R = kg.n_entities, kg.n_relations
    test_lo = kg.splits["test"][0]
    tr = kg.edges[kg.edges["time"] < test_lo].drop_duplicates(["subj", "rel", "obj"])
    s_tr = torch.tensor(tr["subj"].values, device=DEV)
    r_tr = torch.tensor(tr["rel"].values, device=DEV)
    o_tr = torch.tensor(tr["obj"].values, device=DEV)
    if verbose:
        print(f"device {DEV} | train triples {len(tr):,}")
    model = ComplEx(N, R, DIM).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.CrossEntropyLoss()
    n = len(tr)
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEV); tot = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            s, r, o = s_tr[idx], r_tr[idx], o_tr[idx]
            loss = lossf(model.obj_scores(s, r), o) + lossf(model.subj_scores(r, o), s)
            loss = loss + REG * sum((p ** 2).sum() for p in model.parameters())
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            print(f"  epoch {ep:3} loss {tot/(n//BATCH+1):.3f}")
    model.eval()
    return model


def test_triples(kg) -> pd.DataFrame:
    """Per-test-quad rows (edges_weighted is already deduped per (s,r,o,t))."""
    return kg.edges[kg.edges["time"] >= kg.splits["test"][0]]


def test_ranks(model, te: pd.DataFrame):
    """Per-row ComplEx object- and subject-prediction ranks, aligned to `te`."""
    s = torch.tensor(te["subj"].values, device=DEV)
    r = torch.tensor(te["rel"].values, device=DEV)
    o = torch.tensor(te["obj"].values, device=DEV)
    obj, subj = [], []
    with torch.no_grad():
        for i in range(0, len(te), 1024):
            sl = slice(i, i + 1024)
            obj.append(ranks(model.obj_scores(s[sl], r[sl]), o[sl]).cpu())
            subj.append(ranks(model.subj_scores(r[sl], o[sl]), s[sl]).cpu())
    return torch.cat(obj).numpy(), torch.cat(subj).numpy()


def main():
    """Train on pre-test weeks, rank all test quads, and merge the ComplEx row into
    baseline_results.csv (replacing any previous ComplEx entry)."""
    kg = load_core(drop_noise=True)
    model = train(kg)
    te = test_triples(kg)
    orank, srank = test_ranks(model, te)
    rk = torch.tensor(np.concatenate([orank, srank]))
    res = pd.DataFrame({"ComplEx": metrics(rk)}).T[["MRR", "H1", "H3", "H10", "n"]]
    prev = pd.read_csv(OUT / "baseline_results.csv", index_col=0) if (OUT / "baseline_results.csv").exists() else None
    allres = pd.concat([prev[~prev.index.isin(["ComplEx"])], res]) if prev is not None else res
    allres.to_csv(OUT / "baseline_results.csv")
    with pd.option_context("display.float_format", lambda v: f"{v:.4f}"):
        print("\n" + allres.to_string())


if __name__ == "__main__":
    main()
