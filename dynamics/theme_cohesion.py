"""Lifeline-level precision audit, part 3: edge-level cohesion of each EMERGING theme.

A theme's members are guaranteed to have co-occurred (Louvain runs on the co-occurrence
subgraph), but co-occurrence can be substantive joint coverage or generic roundup/listicle
glue. This pass measures the binding directly, per theme, from the articles in the birth
window that mention at least two members:

  n_binding        articles mentioning >= 2 members
  multi3_share     share of binding articles mentioning >= 3 members (real joint coverage)
  pair_coverage    share of member pairs (top-12 members) linked by >= 1 joint article
  mean_members     mean distinct members per binding article

plus cohesion_packs.jsonl: per theme, the top member-pairs with a sample joint headline each,
for LLM/human judgment of whether the binding coverage substantively relates the members.

Run: KG_CORE_PATH=data/kg_600k_dedup_core .venv/Scripts/python -m dynamics.theme_cohesion
"""
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
OUT = Path("data/dynamics/themes")
TRIPL = ["data/tripl_600k_0_dedup.csv", "data/tripl_600k_1_dedup.csv"]
PARQUET = "data/subset_600k.parquet"
TOP_MEMBERS = 12
TOP_PAIRS = 8
WIN_BEFORE, WIN_AFTER = pd.Timedelta(days=28), pd.Timedelta(days=7)


def main():
    lf = pd.read_csv(OUT / "theme_lifelines_lenient.csv")
    mem = pd.read_csv(OUT / "theme_members_lenient.csv").sort_values("excess", ascending=False)
    em = lf[lf.status == "EMERGING"].copy()
    em["birth_dt"] = pd.to_datetime(em["birth"])
    top = {t: list(g.head(TOP_MEMBERS)["name"].astype(str)) for t, g in mem.groupby("theme")}
    windows = {r.theme: (r.birth_dt - WIN_BEFORE, r.birth_dt + WIN_AFTER)
               for _, r in em.iterrows() if r.theme in top}
    name2themes = defaultdict(list)
    for t, ns in top.items():
        if t in windows:
            for n in ns:
                name2themes[n].append(t)
    names = set(name2themes)

    # streaming pass: per (theme, article) -> set of members mentioned
    hits = defaultdict(lambda: defaultdict(set))
    art_date = {}
    for path in TRIPL:
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

    # per-theme cohesion stats + top binding pairs
    stats, want = [], set()
    pair_best = {}   # theme -> [(pair, count, best article id)]
    for t, members in top.items():
        if t not in windows:
            continue
        arts = {a: ms for a, ms in hits.get(t, {}).items() if len(ms) >= 2}
        n_bind = len(arts)
        multi3 = sum(1 for ms in arts.values() if len(ms) >= 3)
        pair_n, pair_art = defaultdict(int), {}
        for a, ms in arts.items():
            for p in combinations(sorted(ms), 2):
                pair_n[p] += 1
                if p not in pair_art or len(ms) > len(arts[pair_art[p]]):
                    pair_art[p] = a
        n_members = len(members)
        n_pairs_possible = n_members * (n_members - 1) // 2
        best = sorted(pair_n.items(), key=lambda kv: -kv[1])[:TOP_PAIRS]
        pair_best[t] = [(p, c, pair_art[p]) for p, c in best]
        want.update(a for _, _, a in pair_best[t])
        stats.append({
            "theme": t, "n_binding": n_bind,
            "multi3_share": round(multi3 / n_bind, 3) if n_bind else 0.0,
            "pair_coverage": round(len(pair_n) / n_pairs_possible, 3) if n_pairs_possible else 0.0,
            "mean_members": round(sum(len(ms) for ms in arts.values()) / n_bind, 2) if n_bind else 0.0,
        })
    sdf = pd.DataFrame(stats).sort_values("theme")
    sdf.to_csv(OUT / "cohesion_stats.csv", index=False)
    print(f"cohesion stats -> cohesion_stats.csv ({len(sdf)} themes)")
    print(sdf[["n_binding", "multi3_share", "pair_coverage", "mean_members"]].describe().round(3))

    tit = pd.read_parquet(PARQUET, columns=["id", "title", "body"])
    tit = tit[tit["id"].isin(want)].set_index("id")
    with open(OUT / "cohesion_packs.jsonl", "w", encoding="utf-8") as f:
        for t, best in pair_best.items():
            pairs = []
            for p, c, a in best:
                lede = " ".join(str(tit.loc[a, "body"]).split())[:450] if a in tit.index else ""
                pairs.append({"pair": list(p), "joint_articles": c,
                              "sample_title": str(tit.loc[a, "title"])[:200] if a in tit.index else "",
                              "body_lede": lede, "date": str(art_date[a].date())})
            f.write(json.dumps({"theme": int(t), "pairs": pairs}, ensure_ascii=False) + "\n")
    print(f"cohesion packs -> cohesion_packs.jsonl")


if __name__ == "__main__":
    main()
