"""Lifeline-level precision audit, part 1: build an audit pack per EMERGING lifeline.

A pack is everything a judge (human or LLM) needs to decide whether a detected theme is a real
financial storyline or noise, WITHOUT seeing the detector's own scores: the member entities, the
birth month, and the actual headlines of the source articles that drove the theme around its
birth. Headlines come from the provenance chain: lifeline members -> dedup triple rows (article
id, date) -> article titles in the corpus parquet.

FROZEN RUBRIC (fixed before any judging; mirrors the frozen-gold-list discipline):
  REAL       members form a coherent, externally documentable financial storyline (an event,
             sector wave or macro development) and the sample headlines are genuine news coverage.
  TEMPLATE   members/headlines are dominated by screener or boilerplate constructs (metric names,
             ranking-service products, formulaic mail-merge titles).
  INCOHERENT no discernible common storyline; the co-occurrence looks incidental.

Outputs (data/dynamics/themes/):
  audit_packs.jsonl        one pack per EMERGING lifeline (lenient config)
  audit_sample_human.csv   seed-42 sample of 15 packs + controls, empty label column for hand-audit

Run: KG_CORE_PATH=data/kg_600k_dedup_core .venv/Scripts/python -m dynamics.theme_audit
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
OUT = Path("data/dynamics/themes")
TRIPL = ["data/tripl_600k_0_dedup.csv", "data/tripl_600k_1_dedup.csv"]
PARQUET = "data/subset_600k.parquet"
TOP_MEMBERS = 12        # members used for headline retrieval (by accumulated excess)
TOP_HEADLINES = 10
WIN_BEFORE, WIN_AFTER = pd.Timedelta(days=28), pd.Timedelta(days=7)
SEED, N_HUMAN = 42, 15
# controls for judge validation (known-real / known-noise); excluded from the precision estimate
CONTROL_REAL_MEMBERS = ("Coronavirus", "GameStop")
CONTROL_NOISE_MEMBERS = ("Earnings ESP",)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="lenient", help="config tag (lenient/nb/strict/tf...)")
    ap.add_argument("--tripl", nargs=2, default=TRIPL,
                    help="triple files for headline retrieval (tf arm uses the filtered ones)")
    a = ap.parse_args()
    tripl = a.tripl
    lf = pd.read_csv(OUT / f"theme_lifelines_{a.tag}.csv")
    mem = pd.read_csv(OUT / f"theme_members_{a.tag}.csv")
    em = lf[lf.status == "EMERGING"].copy()
    em["birth_dt"] = pd.to_datetime(em["birth"])
    print(f"{len(em)} EMERGING lifelines; {len(mem):,} membership rows")

    # per-lifeline retrieval members (top by accumulated excess, full untruncated names)
    mem = mem.sort_values("excess", ascending=False)
    top = {t: g.head(TOP_MEMBERS) for t, g in mem.groupby("theme")}
    name2themes = defaultdict(list)
    windows = {}
    for _, r in em.iterrows():
        t = r.theme
        if t not in top:
            continue
        windows[t] = (r.birth_dt - WIN_BEFORE, r.birth_dt + WIN_AFTER)
        for n in top[t]["name"].astype(str):
            name2themes[n].append(t)
    names = set(name2themes)
    print(f"{len(names):,} retrieval names")

    # one streaming pass over the dedup triple files: article -> (lifeline, members hit)
    hits = defaultdict(lambda: defaultdict(set))   # theme -> article id -> member names
    art_date = {}
    for path in tripl:
        for chunk in pd.read_csv(path, usecols=["id", "date", "h", "o"], chunksize=2_000_000):
            for col in ("h", "o"):
                m = chunk[chunk[col].isin(names)]
                if m.empty:
                    continue
                dts = pd.to_datetime(m["date"].str[:10], cache=True)
                for aid, dt, nm in zip(m["id"], dts, m[col]):
                    for t in name2themes[nm]:
                        lo, hi = windows[t]
                        if lo <= dt <= hi:
                            hits[t][aid].add(nm)
                            art_date[aid] = dt
        print(f"  scanned {path}")

    # titles for the winning articles only
    want = set()
    ranked = {}
    for t, arts in hits.items():
        best = sorted(arts.items(), key=lambda kv: (-len(kv[1]), abs((art_date[kv[0]] - windows[t][0]).days)))
        ranked[t] = best[:TOP_HEADLINES]
        want.update(aid for aid, _ in ranked[t])
    tit = pd.read_parquet(PARQUET, columns=["id", "title", "ticker", "body"])
    tit = tit[tit["id"].isin(want)].set_index("id")
    print(f"{len(want):,} articles titled")

    def snippet(aid, n=450):
        """Lede of the article body, whitespace-collapsed (the lede names the storyline)."""
        b = " ".join(str(tit.loc[aid, "body"]).split())
        return b[:n]

    packs = []
    for _, r in em.iterrows():
        t = r.theme
        if t not in top:
            continue
        members = [{"name": str(n), "excess": float(x)}
                   for n, x in zip(top[t]["name"], top[t]["excess"])]
        heads = []
        for i, (aid, hit) in enumerate(ranked.get(t, [])):
            if aid in tit.index:
                row = tit.loc[aid]
                h = {"date": str(art_date[aid].date()), "ticker": str(row["ticker"]),
                     "members_hit": sorted(hit), "title": str(row["title"])[:200]}
                if i < 3:   # body ledes for the top multi-member articles (headline ambiguity fix)
                    h["body_lede"] = snippet(aid)
                heads.append(h)
        packs.append({"theme": int(t), "birth_month": r.birth[:7], "n_members_total":
                      int((mem.theme == t).sum()), "members": members, "headlines": heads})
    with open(OUT / f"audit_packs_{a.tag}.jsonl" if a.tag != "lenient" else OUT / "audit_packs.jsonl", "w", encoding="utf-8") as f:
        for p in packs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"wrote {len(packs)} packs (tag={a.tag})")

    if a.tag != "lenient":
        return
    # human-audit sheet: seed-42 random sample + labelled controls
    rng = np.random.default_rng(SEED)
    ids = [p["theme"] for p in packs]
    sample = set(rng.choice(ids, size=min(N_HUMAN, len(ids)), replace=False).tolist())
    ctrl = {}
    for p in packs:
        mn = {m["name"] for m in p["members"]}
        if any(c in mn for c in CONTROL_REAL_MEMBERS):
            ctrl[p["theme"]] = "control_real"
        if any(c in mn for c in CONTROL_NOISE_MEMBERS):
            ctrl[p["theme"]] = "control_noise"
    rows = [{"theme": t, "role": ctrl.get(t, "sample"),
             "label_REAL_TEMPLATE_INCOHERENT": "", "notes": ""}
            for t in sorted(sample | set(ctrl))]
    pd.DataFrame(rows).to_csv(OUT / "audit_sample_human.csv", index=False)
    print(f"human sheet: {len(sample)} sampled + {len(ctrl)} controls -> audit_sample_human.csv")


if __name__ == "__main__":
    main()
