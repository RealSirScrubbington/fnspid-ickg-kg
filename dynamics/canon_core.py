"""Build the canonicalised (merged) core for the detector sensitivity arm.

Reads the alias table produced by canon_check.py and rewrites the canonical
deduplicated core with merged entities: each merge group keeps the name,
type and type id of its highest-weight member (so gold-list name matching
and the name-based noise controls behave identically), edge weights are
summed on collision, and self-loops are dropped. The source core is never
modified; output goes to a sibling directory.

Run:  .venv/Scripts/python -m dynamics.canon_core
Then: KG_CORE_PATH=data/kg_600k_dedup_canon_core \
        .venv/Scripts/python -m dynamics.themes --tag canon
      KG_CORE_PATH=data/kg_600k_dedup_canon_core \
        .venv/Scripts/python -m dynamics.themes_eval canon
"""
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SRC = Path("data/kg_600k_dedup_core")
DST = Path("data/kg_600k_dedup_canon_core")
ALIAS = Path("data/dynamics/canon/alias_table.csv")


def main():
    tab = pd.read_csv(ALIAS, keep_default_na=False, na_values=[""])
    ent = pd.read_csv(SRC / "entity2id.txt", sep="\t", header=None,
                      names=["name", "id", "type", "tid"], keep_default_na=False)
    ew = pd.read_csv(SRC / "edges_weighted.tsv", sep="\t")

    # Total mention weight per entity decides each group's representative.
    w = (ew.groupby("subj")["weight"].sum()
         .add(ew.groupby("obj")["weight"].sum(), fill_value=0))
    tab = tab.merge(ent[["id", "tid"]], on="id", how="left")
    tab["w"] = tab["id"].map(w).fillna(0)
    reps = (tab.sort_values(["w", "id"], ascending=[False, True])
            .groupby("group").first().reset_index())
    reps = reps.sort_values("id").reset_index(drop=True)
    reps["newid"] = range(len(reps))
    group2new = dict(zip(reps["group"], reps["newid"]))
    old2new = {i: group2new[g] for i, g in zip(tab["id"], tab["group"])}

    DST.mkdir(parents=True, exist_ok=True)
    out_ent = reps[["name", "newid", "type", "tid"]]
    out_ent.to_csv(DST / "entity2id.txt", sep="\t", header=False, index=False)

    e = ew.copy()
    e["subj"] = e["subj"].map(old2new)
    e["obj"] = e["obj"].map(old2new)
    e = (e[e["subj"] != e["obj"]]
         .groupby(["subj", "rel", "obj", "time_id"], as_index=False)["weight"].sum())
    e.to_csv(DST / "edges_weighted.tsv", sep="\t", index=False)

    n_rel, n_times = (int(x) for x in (SRC / "stat.txt").read_text().split()[1:3])
    (DST / "stat.txt").write_text(f"{len(reps)} {n_rel} {n_times}\n")
    for f in ("relation2id.txt", "time2date.txt"):
        shutil.copy(SRC / f, DST / f)

    print(f"entities {len(ent):,} -> {len(reps):,} | "
          f"edge-week rows {len(ew):,} -> {len(e):,} | wrote {DST}")


if __name__ == "__main__":
    main()
