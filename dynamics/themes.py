"""Walk-forward emerging-theme detection on the core KG — the thesis headline system.

At each week t, using ONLY history <= t (strict TRAILING windows — unlike velocity.py's centered
windows, which are descriptive; this system must be causal to claim real-time detection):

  1. Score every entity's formation acceleration with a Poisson surprise:
       obs = formations in the trailing RECENT weeks,
       exp = RECENT * (rate over the preceding BASE weeks),
       z   = (obs - exp) / sqrt(exp + 1).
     The +1 floor handles zero baselines, so BRAND-NEW entities (e.g. 'Coronavirus' in Jan 2020)
     score high — new-entity emergence is a feature, not an artifact.
  2. Keep entities with z >= Z_MIN and obs >= MIN_OBS (anomalously accelerating this week).
  3. Cluster them on the trailing COOC-week co-occurrence subgraph (Louvain, fixed seed):
     a THEME is a connected group of co-accelerating entities, not a lone spike.
  4. Score a theme by its members' total excess formations (obs - exp); rank; keep the top K.
  5. Track themes week-over-week by member overlap (Jaccard >= MATCH within the last GAP weeks)
     -> theme LIFELINES with a BIRTH week: the date the system would have flagged the theme live.

Two reported configurations (the thesis presents both — the ubiquity filter trades COVID lead time
for a cleaner list):
  LENIENT (default): template-entity exclusion + habituation freshness.
  STRICT (--strict): additionally excludes walk-forward-UBIQUITOUS entities from candidacy,
  dissolving star-shaped screener rituals at the cost of later births for hub-heavy themes.

Outputs: data/dynamics/themes/{theme_weeks,theme_lifelines}_{lenient|strict}.csv
Run: `.venv/Scripts/python -m dynamics.themes [--strict]`   (KG_CORE_PATH selects the core)
"""
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import networkx as nx

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dynamics.loader import load_core
from dynamics.velocity import formation_events, entity_week_matrix

OUT = Path("data/dynamics/themes")
OUT.mkdir(parents=True, exist_ok=True)

RECENT = 4      # trailing acceleration window (weeks) -- override with --recent
BASE = 26       # trailing baseline window preceding RECENT (weeks) -- override with --base
Z_MIN = 3.0     # Poisson-surprise threshold for an entity to be 'accelerating'
MIN_OBS = 6     # min formations in the RECENT window (support floor)
COOC = 8        # trailing co-occurrence window for clustering (weeks)
MIN_SIZE = 3    # min entities per theme
TOP_K = 8       # themes reported per week
MATCH = 0.3     # Jaccard overlap to continue an existing theme
GAP = 8         # a theme can skip up to GAP weeks and still be the same lifeline
SEED = 42


def surprise(cs, t, recent, base_w, alpha=0.0):
    """Surprise z and excess (obs - exp) for every entity at week t, trailing-only.

    alpha = 0 gives the Poisson approximation, var = exp (+1 floor).
    alpha > 0 gives the NEGATIVE-BINOMIAL variance, var = exp + alpha*exp^2 (+1 floor) --
    news-driven counts are overdispersed (coverage clusters), so the Poisson z overstates
    burstiness for high-volume entities; the NB variance corrects this. alpha is estimated
    on TRAIN weeks only (walk-forward safe)."""
    obs = cs[:, t + 1] - cs[:, t + 1 - recent]
    base = cs[:, t + 1 - recent] - cs[:, t + 1 - recent - base_w]
    exp = base * (recent / base_w)
    z = (obs - exp) / np.sqrt(exp + alpha * exp ** 2 + 1.0)
    return z, obs, obs - exp


def estimate_dispersion(M, train_hi):
    """Method-of-moments NB dispersion alpha from TRAIN weeks only: for entity weekly counts,
    Var = mu + alpha*mu^2, so alpha = median over active entities of (v_e - m_e)/m_e^2."""
    Mt = M[:, :train_hi]
    m = Mt.mean(axis=1)
    v = Mt.var(axis=1)
    act = m >= 0.25                       # entities with enough baseline activity to estimate
    a = (v[act] - m[act]) / np.maximum(m[act] ** 2, 1e-9)
    return float(np.median(a)), int(act.sum())


def template_ids(kg):
    """Entity ids measured as template-dominated (>=60% of mentions from slot-fingerprint template
    articles; see the title/body fingerprint audit) — excluded from theme candidacy. Data-driven,
    not hand-curated; the list file is produced by the template-flagging script.
    CARVE-OUT (disclosed in the thesis): this is a STATIC list built from full-period corpus
    statistics — the one non-trailing input to an otherwise walk-forward system. No gold-event
    anchor entity is on the list; a live deployment would rebuild it on a rolling window."""
    f = OUT / "template_entities.txt"
    if not f.exists():
        return set()
    names = {l.split("\t")[0] for l in f.read_text(encoding="utf-8").splitlines() if l.strip()}
    return {i for i, n in kg.id2name.items() if str(n) in names}


def main():
    """Walk the weeks forward: at each t select accelerating entities (trailing surprise on
    history <= t), cluster them on the trailing co-occurrence subgraph, match clusters to
    theme lifelines, and write the per-week and lifeline CSVs. All walk-forward state
    (habituation, ubiquity) is updated only AFTER week t is processed."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true",
                    help="also exclude walk-forward-ubiquitous entities from candidacy")
    ap.add_argument("--nb", action="store_true",
                    help="negative-binomial surprise (overdispersion-corrected) instead of Poisson")
    ap.add_argument("--recent", type=int, default=RECENT, help="trailing acceleration window (wk)")
    ap.add_argument("--base", type=int, default=BASE, help="trailing baseline window (wk)")
    ap.add_argument("--tag", default=None, help="output-file tag (default: lenient/strict)")
    a = ap.parse_args()
    mode = a.tag or ("strict" if a.strict else "lenient")

    kg = load_core(drop_noise=True)
    ev = formation_events(kg)
    M = entity_week_matrix(ev, kg.n_entities, kg.n_times)
    cs = np.concatenate([np.zeros((kg.n_entities, 1)), np.cumsum(M, axis=1)], axis=1)

    edges_by_week = {t: g[["subj", "obj", "weight"]].to_numpy()
                     for t, g in kg.edges.groupby("time")}

    themes = {}          # tid -> dict(members, last_seen, birth, peak, peak_wk, excess_by_ent, fresh)
    next_tid = 0
    week_rows = []
    last_pass = np.full(kg.n_entities, -(10 ** 9))   # last week each entity accelerated (walk-forward)
    n_episodes = np.zeros(kg.n_entities, dtype=int)  # count of DISTINCT past acceleration episodes

    tmpl = template_ids(kg)
    print(f"[templates] excluding {len(tmpl):,} template-dominated entities from theme candidacy")
    tmask = np.ones(kg.n_entities, dtype=bool)
    if tmpl:
        tmask[list(tmpl)] = False

    # walk-forward ubiquity: cumulative count of active weeks per entity. Entities active in more
    # than UBIQ_FRAC of history (after a burn-in) are BACKGROUND (Stock Value, S&P 500, Earnings
    # ESP, Insider Ownership, ...) -- they join every big story, so they cannot *define* emergence.
    # Excluding them from candidacy dissolves star-shaped ritual clusters (hub + rotating spokes
    # that never co-occur with each other) while clique-shaped real themes survive.
    act_cum = np.cumsum((M > 0).astype(np.int32), axis=1)   # act_cum[e, t] = active weeks in 0..t
    UBIQ_FRAC, UBIQ_BURNIN = 0.5, 52

    alpha = 0.0
    if a.nb:
        alpha, n_act = estimate_dispersion(M, int(kg.n_times * 0.70))
        alpha = max(alpha, 0.0)   # clamp: a negative MoM estimate (underdispersion) would flip the variance sign
        # NOTE: alpha is estimated ONCE on train weeks (0-254) and applied at every t. For events
        # before that horizon this embeds later dispersion -- conservative (COVID-era overdispersion
        # RAISES the detection bar) but not strictly point-in-time; disclosed in the thesis.
        print(f"[nb] dispersion alpha = {alpha:.3f} (method of moments, {n_act:,} active entities, "
              f"train weeks only)")

    for t in range(a.recent + a.base, kg.n_times):
        z, obs, excess = surprise(cs, t, a.recent, a.base, alpha)
        keep = (z >= Z_MIN) & (obs >= MIN_OBS) & tmask
        if a.strict:
            keep &= ~((t >= UBIQ_BURNIN) & (act_cum[:, t - 1] > UBIQ_FRAC * t))
        cand = np.where(keep)[0]

        def book():
            """Record week t's candidates in the habituation state (last_pass / n_episodes)."""
            # habituation bookkeeping runs for EVERY week (audit fix: earlier 'continue' paths
            # skipped it, corrupting last_pass/n_episodes on cluster-less weeks); a pass more than
            # RECENT weeks after the previous one starts a NEW episode
            n_episodes[cand[(t - last_pass[cand]) > a.recent]] += 1
            last_pass[cand] = t

        if len(cand) < MIN_SIZE:
            book(); continue
        cset = set(cand.tolist())

        # trailing co-occurrence subgraph among accelerating entities
        G = nx.Graph()
        G.add_nodes_from(cand.tolist())
        for w in range(max(0, t - COOC + 1), t + 1):
            for s, o, wt in edges_by_week.get(w, ()):
                s, o = int(s), int(o)
                if s in cset and o in cset and s != o:
                    G.add_edge(s, o, weight=G.get_edge_data(s, o, {"weight": 0})["weight"] + wt)

        comms = nx.community.louvain_communities(G, weight="weight", seed=SEED)
        clusters = [sorted(c) for c in comms if len(c) >= MIN_SIZE]
        if not clusters:
            book(); continue
        scored = sorted(((float(excess[c].sum()), c) for c in clusters), reverse=True)[:TOP_K]

        # match clusters to theme lifelines by member overlap: LIVE lifelines first (a continuation
        # beats a re-activation regardless of Jaccard -- audit fix: the old single-pass tuple test
        # was dict-order dependent), then dead ones (a RE-ACTIVATION keeps its old identity -- it is
        # not a new emergence). Each lifeline can be claimed by at most one cluster per week.
        claimed = set()
        for rank, (score, members) in enumerate(scored, 1):
            mset = set(members)
            best = None
            for live_pass in (True, False):
                bj = MATCH
                for tid, th in themes.items():
                    if tid in claimed or ((t - th["last_seen"]) <= GAP) != live_pass:
                        continue
                    j = len(mset & th["members"]) / len(mset | th["members"])
                    if j >= bj:
                        best, bj = tid, j
                if best is not None:
                    break
            if best is None:
                # freshness: EXCESS-WEIGHTED share of members that are NOT habitual accelerators.
                # An entity is fresh if it has < 2 distinct PRIOR acceleration episodes (the ongoing
                # episode does not count against it). Weighting by excess means a theme is 'emerging'
                # only if the entities DRIVING it are new phenomena -- ritual themes (earnings-season
                # screens, index complexes) whose anchors re-accelerate every quarter classify as
                # recurring even though their rotating company cast looks fresh.
                def prior_eps(e):
                    """Distinct acceleration episodes of e strictly BEFORE the ongoing one."""
                    return n_episodes[e] - (1 if (t - last_pass[e]) <= a.recent else 0)
                tot_x = float(sum(excess[e] for e in members))
                fresh = float(sum(excess[e] for e in members if prior_eps(e) < 2) / max(tot_x, 1e-9))
                best = next_tid; next_tid += 1
                themes[best] = {"members": mset, "last_seen": t, "birth": t, "fresh": fresh,
                                "peak": score, "peak_wk": t, "excess_by_ent": {}}
            claimed.add(best)
            th = themes[best]
            th["members"] = mset; th["last_seen"] = t
            if score > th["peak"]:
                th["peak"], th["peak_wk"] = score, t
            for e in members:
                th["excess_by_ent"][e] = th["excess_by_ent"].get(e, 0.0) + float(excess[e])
            top = sorted(mset, key=lambda e: -excess[e])[:5]
            week_rows.append((t, kg.date(t), best, rank, round(score, 1), len(mset),
                              " | ".join(str(kg.id2name[e])[:28] for e in top)))
        book()   # update AFTER the week is processed (freshness stays walk-forward)

    wk = pd.DataFrame(week_rows, columns=["week", "date", "theme", "rank", "score", "size", "top_members"])
    wk.to_csv(OUT / f"theme_weeks_{mode}.csv", index=False)

    rows = []
    for tid, th in themes.items():
        label = sorted(th["excess_by_ent"], key=lambda e: -th["excess_by_ent"][e])[:6]
        rows.append((tid, kg.date(th["birth"]), kg.date(th["last_seen"]), kg.date(th["peak_wk"]),
                     round(th["peak"], 1), int((wk["theme"] == tid).sum()), round(th["fresh"], 2),
                     "EMERGING" if th["fresh"] >= 0.5 else "recurring",
                     " | ".join(str(kg.id2name[e])[:28] for e in label)))
    lf = pd.DataFrame(rows, columns=["theme", "birth", "last", "peak_wk", "peak_score", "weeks",
                                     "fresh", "status", "members"])
    lf = lf.sort_values("peak_score", ascending=False).reset_index(drop=True)
    lf.to_csv(OUT / f"theme_lifelines_{mode}.csv", index=False)

    em = lf[lf["status"] == "EMERGING"]
    print(f"\n=== [{mode}] {len(lf):,} theme lifelines ({len(em):,} EMERGING / {len(lf)-len(em):,} recurring) "
          f"over {wk['week'].nunique()} active weeks (z>={Z_MIN}, obs>={MIN_OBS}, top-{TOP_K}/wk) ===")
    print("\ntop 25 EMERGING themes by peak score — BIRTH = the week the system flags the theme live:")
    for _, r in em.head(25).iterrows():
        print(f"  {r.peak_score:>7.0f}  born {r.birth}  peak {r.peak_wk}  ({r.weeks:>3}wk, fresh {r.fresh:.0%})  {r.members}")
    print(f"\nsaved {OUT/f'theme_weeks_{mode}.csv'} and {OUT/f'theme_lifelines_{mode}.csv'}")


if __name__ == "__main__":
    main()
