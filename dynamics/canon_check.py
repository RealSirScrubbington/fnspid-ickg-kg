"""Deterministic ticker-anchored canonicalisation sensitivity check (ADDITIVE arm).

Upgrades the crude suffix merge of resolve_check.py to a type-constrained,
ticker-anchored alias table, then re-measures test novelty and the
training-free recurrence baseline under the merge. All frozen arms are
untouched; this is a sensitivity line, not a new canonical construction.

Protocol (fixed before running; deterministic, no learned components):
  Rule A  within-type merge on the suffix-stripped key (keynorm of
          resolve_check.py), restricted to corporate types
          {COMP, ORG, ORG/GOV, ORG/REG}. Unlike the crude merge, entities
          of different types NEVER merge.
  Rule B  (COMP only) ticker anchoring from provenance: an entity anchors
          to ticker T if >= 60% of its mention rows in the deduplicated
          triple files carry T and it has >= 10 mentions; an entity whose
          key equals the key of an SEC EDGAR company title anchors to that
          ticker unconditionally. Two anchored COMP entities merge iff they
          share the anchor ticker AND their keys share the same first
          token (guards against subsidiary fusion: Instagram never merges
          into Meta Platforms).
  Guards  (i) entities on EITHER noise list (the 291-entity loader
          blacklist or the 2,010-entity detector template list) never
          merge, so the noise controls stay identical across arms;
          (ii) anchor-conflict veto: same-key entities whose dominance
          anchors point to different tickers do not merge (this dissolves
          suffix-key collisions such as Prudential Financial [PRU] vs
          Prudential Plc [PUK]); within a conflicted key group, per-anchor
          subgroups merge and unanchored members stay unmerged;
          (iii) keys of <= 2 characters never merge; self-loops dropped
          after remapping; edge weights summed on collision.
          Guards (i) and (ii) were added after specimen review of the
          unguarded table (634 template-name members; one false merge
          found); the guarded run is the single quotable one.

Run:  KG_CORE_PATH=data/kg_600k_dedup_core \
        .venv/Scripts/python -m dynamics.canon_check
Outputs: data/dynamics/canon/{alias_table.csv, merge_groups.csv} for human
review, plus the before/after novelty and recurrence-MRR comparison.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from dynamics.loader import load_core
from dynamics.noise import noise_entity_ids
from dynamics.resolve_check import keynorm, eval_recurrence

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TRIPLE_FILES = ["data/tripl_600k_0_dedup.csv", "data/tripl_600k_1_dedup.csv"]
EDGAR_FILE = "data/company_tickers.json"
OUT = Path("data/dynamics/canon")
CORPORATE_TYPES = {"COMP", "ORG", "ORG/GOV", "ORG/REG"}
DOMINANCE = 0.60
MIN_MENTIONS = 10


def mention_ticker_counts():
    """(surface form, type) -> Counter of article tickers, from the dedup triple files."""
    counts = defaultdict(Counter)
    for f in TRIPLE_FILES:
        df = pd.read_csv(f, usecols=["ticker", "h", "h_type", "o", "o_type"],
                         keep_default_na=False, dtype=str)
        for name_col, type_col in (("h", "h_type"), ("o", "o_type")):
            g = df.groupby([name_col, type_col, "ticker"]).size()
            for (name, typ, tick), n in g.items():
                counts[(name, typ)][tick] += n
    return counts


def template_names():
    """The detector's 2,010-entity template list (names), if present."""
    f = Path("data/dynamics/themes/template_entities.txt")
    if not f.exists():
        return set()
    return {line.split("\t")[0] for line in f.read_text(encoding="utf-8").splitlines() if line}


def build_alias_table(kg):
    """Assign each entity a group id under Rules A + B. Returns a DataFrame."""
    noise = noise_entity_ids(kg.id2name)
    tmpl = template_names()
    edgar = json.load(open(EDGAR_FILE))
    title_key2ticker = {}
    for rec in edgar.values():
        title_key2ticker.setdefault(keynorm(rec["title"]), rec["ticker"])

    rows = []
    for i in range(kg.n_entities):
        name, typ = kg.id2name.get(i), kg.id2type.get(i, "?")
        if name is None:
            continue
        key = keynorm(name)
        rows.append({"id": i, "name": name, "type": typ, "key": key,
                     "mergeable": (typ in CORPORATE_TYPES and i not in noise
                                   and name not in tmpl and len(key) > 2)})
    tab = pd.DataFrame(rows)

    # Rule B anchors (COMP only): provenance dominance, then EDGAR titles.
    counts = mention_ticker_counts()
    anchors = {}
    for r in tab[tab.mergeable & (tab.type == "COMP")].itertuples():
        # An exact EDGAR-title match is an explicit legal identity and takes
        # precedence over mention dominance (a foreign company discussed
        # mainly in articles about a US near-namesake would otherwise
        # inherit the wrong anchor: Prudential Plc's mentions are dominated
        # by PRU-tagged articles, but its title anchors it to PUK).
        if r.key in title_key2ticker:
            anchors[r.id] = title_key2ticker[r.key]
            continue
        c = counts.get((r.name, "COMP"))
        if c:
            total = sum(c.values())
            tick, top = c.most_common(1)[0]
            if total >= MIN_MENTIONS and top / total >= DOMINANCE:
                anchors[r.id] = tick
    tab["ticker"] = tab["id"].map(anchors)

    # Union-find over mergeable entities.
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    m = tab[tab.mergeable]
    for _, grp in m.groupby(["type", "key"]):          # Rule A + anchor veto
        ticks = grp["ticker"].dropna().unique()
        if len(ticks) <= 1:
            ids = grp["id"].tolist()
            for other in ids[1:]:
                union(ids[0], other)
        else:  # conflicting anchors: merge per-anchor subgroups only
            for _, sub in grp.dropna(subset=["ticker"]).groupby("ticker"):
                ids = sub["id"].tolist()
                for other in ids[1:]:
                    union(ids[0], other)
    anchored = m[m.ticker.notna() & (m.type == "COMP")].copy()
    anchored["tok0"] = anchored["key"].str.split().str[0]
    for _, grp in anchored.groupby(["ticker", "tok0"]):  # Rule B
        ids = grp["id"].tolist()
        for other in ids[1:]:
            union(ids[0], other)

    tab["group"] = [find(i) if mg else i for i, mg in zip(tab["id"], tab.mergeable)]
    return tab


def main():
    kg = load_core(drop_noise=True)
    test_lo = kg.splits["test"][0]
    tab = build_alias_table(kg)

    # Human-review artifacts.
    OUT.mkdir(parents=True, exist_ok=True)
    tab.to_csv(OUT / "alias_table.csv", index=False)
    sizes = tab.groupby("group").size()
    multi = sizes[sizes > 1]
    weight = kg.edges.groupby("subj")["weight"].sum().add(
        kg.edges.groupby("obj")["weight"].sum(), fill_value=0)
    groups = (tab[tab.group.isin(multi.index)]
              .assign(w=lambda d: d["id"].map(weight).fillna(0))
              .sort_values(["group", "w"], ascending=[True, False])
              .groupby("group")
              .agg(size=("id", "size"), type=("type", "first"),
                   ticker=("ticker", "first"),
                   members=("name", lambda s: " | ".join(s)))
              .sort_values("size", ascending=False))
    groups.to_csv(OUT / "merge_groups.csv")

    # Remap edges (representative id = the group root; weights summed by
    # eval_recurrence's per-week iteration, self-loops and exact dups dropped).
    old2new = dict(zip(tab["id"], tab["group"]))
    compact = {g: j for j, g in enumerate(sorted(set(old2new.values())))}
    old2new = {i: compact[g] for i, g in old2new.items()}
    e = kg.edges.copy()
    e["subj"] = e["subj"].map(old2new)
    e["obj"] = e["obj"].map(old2new)
    e = (e[e["subj"] != e["obj"]]
         .drop_duplicates(["subj", "rel", "obj", "time"])
         .reset_index(drop=True))

    nov0, mrr0 = eval_recurrence(kg.edges, kg.n_entities, kg.n_times, test_lo)
    nov1, mrr1 = eval_recurrence(e, len(compact), kg.n_times, test_lo)

    n_anchor = tab.ticker.notna().sum()
    print("\n=== ticker-anchored canonicalisation (deterministic, additive arm) ===")
    print(f"mergeable corporate entities: {int(tab.mergeable.sum()):,} of {kg.n_entities:,}")
    print(f"ticker-anchored (COMP):       {n_anchor:,}")
    print(f"merge groups >1:              {len(multi):,} (largest {int(sizes.max())})")
    print(f"entities:        {kg.n_entities:,} -> {len(compact):,}")
    print(f"edge-instances:  {len(kg.edges):,} -> {len(e):,}")
    print(f"TEST novelty:    {nov0:.1%} -> {nov1:.1%}   (drop {100 * (nov0 - nov1):.1f} pts)")
    print("\nrecurrence MRR     unmerged     merged")
    for g in ("all", "recurring", "novel"):
        print(f"  {g:10}       {mrr0[g]:.4f}      {mrr1[g]:.4f}")
    print(f"\nreview artifacts: {OUT}/alias_table.csv, {OUT}/merge_groups.csv")


if __name__ == "__main__":
    main()
