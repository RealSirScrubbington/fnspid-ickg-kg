"""RE-GCN-style temporal GNN for link-prediction forecasting on the core KG.

Entity embeddings EVOLVE through weekly snapshots: each week a CompGCN-style relational
message pass produces a structural update that a GRU folds into the running state H_t. At time
t the model predicts facts AT t from H_{t-1} (forecasting; strict PIT) with a DistMult decoder
over the time-evolved embeddings. Trained on weeks < test_lo with truncated BPTT, then H is
rolled forward through history and the test weeks are scored with the same raw rank + novel/
recurring split as the baselines. Ref: Li et al., "Temporal Knowledge Graph Reasoning Based on
Evolutional Representation Learning" (RE-GCN), 2021.

Run: `.venv/Scripts/python -m dynamics.regcn`
"""
from pathlib import Path
import sys
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from dynamics.loader import load_core
from dynamics.complex_kge import ranks
from dynamics.linkpred import Meter

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
OUT = Path("data/dynamics/linkpred")
DIM, EPOCHS, HIST, LR, DROP = 128, 30, 6, 2e-3, 0.2   # HIST = recent-window length re-evolved from E0
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class REGCN(nn.Module):
    def __init__(self, n_ent, n_rel, d, drop=DROP, feat_dim=0):
        super().__init__()
        self.N, self.R = n_ent, n_rel
        self.E0 = nn.Parameter(torch.empty(n_ent, d)); nn.init.xavier_normal_(self.E0)
        self.Rel = nn.Parameter(torch.empty(2 * n_rel, d)); nn.init.xavier_normal_(self.Rel)
        self.Wn = nn.Linear(d, d); self.Ws = nn.Linear(d, d)
        self.gru = nn.GRUCell(d, d); self.drop = nn.Dropout(drop)
        self.scale = nn.Parameter(torch.tensor(8.0))        # logit temperature (normalized embs)
        self.shuffle = False                                # placebo: randomize history order
        # velocity-feature ablation: per-entity trailing velocity statistics injected at every
        # snapshot step through a learned projection. feat_dim=0 leaves the baseline arm
        # bit-identical to the original model (no extra parameters).
        self.Wf = nn.Linear(feat_dim, d) if feat_dim else None
        self.feat = None                                    # [N, T, feat_dim] tensor, set by main()

    def init_state(self):
        """Evolution start state: normalized tanh of the static base embeddings E0."""
        return F.normalize(torch.tanh(self.E0), dim=1)

    def evolve(self, H, ed, w=None):
        """One weekly evolution step: degree-normalized relational messages (forward +
        inverse), a residual conv layer, then a GRU update; returns the new normalized H.
        With velocity features enabled, the snapshot week's trailing velocity statistics
        (strictly <= w, hence < prediction time) enter the conv pre-activation."""
        s, rel, o = ed
        msg_o = H[s] * self.Rel[rel]                         # forward messages -> object
        msg_s = H[o] * self.Rel[rel + self.R]               # inverse messages -> subject
        agg = torch.zeros_like(H)
        agg.index_add_(0, o, msg_o); agg.index_add_(0, s, msg_s)
        deg = torch.zeros(self.N, device=H.device)
        ones = torch.ones(len(s), device=H.device)
        deg.index_add_(0, o, ones); deg.index_add_(0, s, ones)
        agg = agg / deg.clamp(min=1).unsqueeze(1)
        pre = self.Wn(agg) + self.Ws(H)
        if self.Wf is not None and w is not None:
            pre = pre + self.Wf(self.feat[:, w])
        conv = self.drop(F.relu(pre))
        return F.normalize(self.gru(conv, H), dim=1)

    # DistMult decoder over the evolved embeddings; inverse-relation embeddings serve (?, r, o)
    def score_obj(self, H, s, rel):
        return self.scale * ((H[s] * self.Rel[rel]) @ H.t())
    def score_sub(self, H, o, rel):
        return self.scale * ((H[o] * self.Rel[rel + self.R]) @ H.t())

    def embed_at(self, week, t, hist):
        """Time-evolved embeddings for predicting facts AT t: re-evolve from E0 over the
        recent window [t-hist, t-1] so the base embeddings get gradient from every t."""
        H = self.init_state()
        ws = [w for w in range(max(0, t - hist), t) if week.get(w) is not None]
        if self.shuffle:
            random.shuffle(ws)                         # placebo: destroy chronological order
        for w in ws:
            H = self.evolve(H, week[w], w)
        return H


@torch.no_grad()
def valid_mrr(model, week, lo, hi, hist):
    """Forecasting MRR on the validation split (model selection criterion)."""
    model.eval(); rr = []
    for t in range(lo, hi):
        ed = week.get(t)
        if ed is None:
            continue
        s, rel, o = ed
        H = model.embed_at(week, t, hist)
        rr.append(1.0 / ranks(model.score_obj(H, s, rel), o).float())
        rr.append(1.0 / ranks(model.score_sub(H, o, rel), s).float())
    return torch.cat(rr).mean().item() if rr else 0.0


def main():
    """Train RE-GCN on weeks < test_lo (early-stopped on valid MRR, or fixed epochs with
    --through-valid), then walk the test weeks: facts at t are scored from embeddings evolved
    over weeks < t only, and the recurrence history updates AFTER each week is scored (PIT).
    Writes regcn_results.csv with the novel/recurring split."""
    import argparse
    from collections import defaultdict
    from dynamics.linkpred import group_rank
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60, help="max epochs (early stopping may cut short)")
    ap.add_argument("--dim", type=int, default=DIM)
    ap.add_argument("--hist", type=int, default=HIST)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--wd", type=float, default=1e-5, help="weight decay")
    ap.add_argument("--drop", type=float, default=0.3)
    ap.add_argument("--through-valid", action="store_true",
                    help="train on train+valid (apples-to-apples vs ComplEx), fixed epochs, no early stop")
    ap.add_argument("--shuffle-hist", action="store_true",
                    help="placebo: shuffle the order of history snapshots (kills chronological signal)")
    ap.add_argument("--velocity-features", action="store_true",
                    help="ablation arm: inject per-entity trailing velocity statistics "
                         "(log1p recent formations, log1p baseline, surprise z) at every snapshot")
    ap.add_argument("--tag", default="", help="suffix for the results CSV (ablation bookkeeping)")
    ap.add_argument("--dump-topk", type=int, default=0,
                    help="also dump top-K candidate ids per test query (worked-example error "
                         "analysis); 0 = off, behaviour unchanged")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed); torch.manual_seed(a.seed)

    kg = load_core(drop_noise=True)
    N, R = kg.n_entities, kg.n_relations
    test_lo = kg.splits["test"][0]
    week = {t: (torch.tensor(g["subj"].values, device=DEV),
                torch.tensor(g["rel"].values, device=DEV),
                torch.tensor(g["obj"].values, device=DEV))
            for t, g in kg.edges.groupby("time")}
    print(f"device {DEV} | {N:,} entities | dim {a.dim} hist {a.hist} epochs {a.epochs} lr {a.lr}")

    valid_lo = kg.splits["valid"][0]
    model = REGCN(N, R, a.dim, a.drop, feat_dim=3 if a.velocity_features else 0).to(DEV)
    model.shuffle = a.shuffle_hist
    if a.velocity_features:
        # Trailing velocity features, identical machinery to the theme detector: recent 4-week
        # formation count O_w, scaled 26-week baseline E_w, surprise z_w. Windows end AT the
        # snapshot week w; embed_at only visits weeks < prediction time, so features stay PIT.
        from dynamics.velocity import formation_events, entity_week_matrix
        r_win, b_win = 4, 26
        M = entity_week_matrix(formation_events(kg), N, kg.n_times)
        cs = np.concatenate([np.zeros((N, 1)), np.cumsum(M, axis=1)], axis=1)
        feat = np.zeros((N, kg.n_times, 3), dtype=np.float32)
        for w in range(kg.n_times):
            lo_r = max(0, w + 1 - r_win)
            lo_b = max(0, lo_r - b_win)
            O = cs[:, w + 1] - cs[:, lo_r]
            E = (cs[:, lo_r] - cs[:, lo_b]) * (r_win / b_win)
            z = (O - E) / np.sqrt(E + 1.0)
            feat[:, w, 0] = np.log1p(O)
            feat[:, w, 1] = np.log1p(E)
            feat[:, w, 2] = np.clip(z, -10.0, 10.0) / 10.0
        model.feat = torch.tensor(feat, device=DEV)
        print(f"[ablation] velocity features ON: {feat.shape}, "
              f"{model.feat.element_size() * model.feat.nelement() / 1e6:.0f} MB on {DEV}")
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.wd)
    ce = nn.CrossEntropyLoss()
    hi = test_lo if a.through_valid else valid_lo
    train_times = [t for t in range(1, hi) if t in week]
    print(f"train weeks 1-{hi-1} ({len(train_times)}) | through_valid={a.through_valid} "
          f"shuffle_hist={a.shuffle_hist}", flush=True)

    def train_step():
        """One epoch over the training weeks (truncated BPTT via embed_at); returns mean loss."""
        model.train(); tot = 0.0
        for t in train_times:
            H = model.embed_at(week, t, a.hist)
            s, rel, o = week[t]
            loss = ce(model.score_obj(H, s, rel), o) + ce(model.score_sub(H, o, rel), s)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item()
        return tot / len(train_times)

    if a.through_valid:                                  # apples-to-apples: fixed epochs, no holdout
        for ep in range(a.epochs):
            ls = train_step()
            if ep % 3 == 0 or ep == a.epochs - 1:
                print(f"  epoch {ep:3} loss {ls:.3f}", flush=True)
    else:                                                # validation-based early stopping
        best, best_ep, bad, best_state = -1.0, -1, 0, None
        for ep in range(a.epochs):
            ls = train_step()
            vm = valid_mrr(model, week, valid_lo, test_lo, a.hist)
            print(f"  epoch {ep:3} loss {ls:.3f} | valid MRR {vm:.4f}", flush=True)
            if vm > best + 1e-5:
                best, best_ep, bad = vm, ep, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= a.patience:
                    print(f"  early stop (no valid gain for {a.patience} epochs)"); break
        model.load_state_dict({k: v.to(DEV) for k, v in best_state.items()})
        print(f"selected epoch {best_ep} | best valid MRR {best:.4f}", flush=True)

    # --- eval: RE-GCN vs recurrence vs the recurrence->RE-GCN hybrid, split novel/recurring ---
    model.eval()
    METH = ["RE-GCN", "recurrence", "backoff(rec->RE-GCN)"]
    meters = {(m, g): Meter() for m in METH for g in ("all", "recurring", "novel")}
    sro = defaultdict(lambda: defaultdict(int)); ors = defaultdict(lambda: defaultdict(int)); seen = set()

    def add_week(ed):
        """Fold a week's edges into the recurrence counts and the seen-(s,r,o) set."""
        for s, r, o in zip(*(x.tolist() for x in ed)):
            sro[(s, r)][o] += 1; ors[(o, r)][s] += 1; seen.add((s, r, o))

    for t in range(test_lo):
        ed = week.get(t)
        if ed is not None:
            add_week(ed)
    dump_ro, dump_rs, dump_nov = [], [], []   # per-query ranks for paired arm-vs-arm bootstraps
    dump_tko, dump_tks = [], []               # optional top-K candidates (--dump-topk)
    with torch.no_grad():
        for t in range(test_lo, kg.n_times):
            ed = week.get(t)
            if ed is None:
                continue
            s, rel, o = ed
            H = model.embed_at(week, t, a.hist)
            sc_o = model.score_obj(H, s, rel)
            sc_s = model.score_sub(H, o, rel)
            ro = ranks(sc_o, o).cpu().numpy()
            rs = ranks(sc_s, s).cpu().numpy()
            dump_ro.append(ro); dump_rs.append(rs)
            if a.dump_topk:
                dump_tko.append(torch.topk(sc_o, a.dump_topk, dim=1).indices.cpu().numpy())
                dump_tks.append(torch.topk(sc_s, a.dump_topk, dim=1).indices.cpu().numpy())
            dump_nov.append(np.array([(si, ri, oi) not in seen
                                      for si, ri, oi in zip(s.tolist(), rel.tolist(), o.tolist())]))
            for i, (si, ri, oi) in enumerate(zip(s.tolist(), rel.tolist(), o.tolist())):
                sq = (si, ri, oi) in seen
                grp = "recurring" if sq else "novel"
                gnn = (ro[i], rs[i])
                rec = (group_rank(sro.get((si, ri), {}), oi, N), group_rank(ors.get((oi, ri), {}), si, N))
                for nm, (po, ps) in (("RE-GCN", gnn), ("recurrence", rec),
                                     ("backoff(rec->RE-GCN)", rec if sq else gnn)):
                    for g in ("all", grp):
                        meters[(nm, g)].add(po); meters[(nm, g)].add(ps)
            # history update AFTER scoring week t (PIT)
            add_week(ed)

    rows = {}
    for m in METH:
        rows[m] = {f"MRR/{g}": meters[(m, g)].row()["MRR"] for g in ("all", "recurring", "novel")}
        rows[m]["H10/all"] = meters[(m, "all")].row()["H10"]
    res = pd.DataFrame(rows).T
    res.to_csv(OUT / f"regcn_results{('_' + a.tag) if a.tag else ''}.csv")
    if a.tag:   # per-query ranks in test-week order (identical across arms -> paired tests)
        extra = {}
        if a.dump_topk:
            extra = {"topk_o": np.concatenate(dump_tko), "topk_s": np.concatenate(dump_tks)}
        np.savez_compressed(OUT / f"regcn_ranks_{a.tag}.npz",
                            ro=np.concatenate(dump_ro), rs=np.concatenate(dump_rs),
                            novel=np.concatenate(dump_nov), **extra)
    with pd.option_context("display.float_format", lambda v: f"{v:.4f}"):
        print("\n" + res.to_string())
    print("\nreference: ComplEx MRR/all 0.082, novel 0.025 | backoff(ComplEx) MRR/all 0.152, H10 0.242")


if __name__ == "__main__":
    main()
