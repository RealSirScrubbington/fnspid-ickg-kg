"""Produce a denoised 'core' KG from the full FinDKG graph by thresholding on edge
support (mention weight) and entity degree, then re-indexing + re-splitting. This is
a NON-DESTRUCTIVE convenience artifact for the dynamics analysis -- the full graph and
raw triplets are untouched. Same temporal axis as the full graph (comparable time_ids).

  python build_core.py --kg data/kg --out data/kg_core --min-weight 2 --min-degree 2
"""
import argparse, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ickg_kg.schema import RELATION2ID


def main() -> None:
    """Threshold edges by weight, iterate the entity-degree filter to a fixed point,
    re-index survivors by frequency (low ids = frequent), re-split chronologically on the
    UNCHANGED time axis, and write the core FinDKG files."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", default="data/kg")
    ap.add_argument("--out", default="data/kg_core")
    ap.add_argument("--min-weight", type=int, default=2, help="keep edges supported by >= N articles")
    ap.add_argument("--min-degree", type=int, default=2, help="keep entities in >= N (surviving) edges")
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--valid-frac", type=float, default=0.15)
    args = ap.parse_args()
    kg, out = Path(args.kg), Path(args.out); out.mkdir(parents=True, exist_ok=True)

    ent = pd.read_csv(kg / "entity2id.txt", sep="\t", header=None, names=["name", "id", "type", "tid"])
    id2 = ent.set_index("id")[["name", "type", "tid"]].to_dict("index")
    e = pd.read_csv(kg / "edges_weighted.tsv", sep="\t")
    n0_ent, n0_edge = len(ent), len(e)

    e = e[e["weight"] >= args.min_weight]
    while True:                                   # iterate degree filter to a fixed point
        deg = pd.concat([e["subj"], e["obj"]]).value_counts()
        keep = set(deg[deg >= args.min_degree].index)
        e2 = e[e["subj"].isin(keep) & e["obj"].isin(keep)]
        if len(e2) == len(e):
            break
        e = e2

    survivors = list(pd.concat([e["subj"], e["obj"]]).value_counts().index)  # freq desc
    old2new = {o: i for i, o in enumerate(survivors)}
    e = e.assign(subj=e["subj"].map(old2new), obj=e["obj"].map(old2new))

    t2d = (kg / "time2date.txt").read_text(encoding="utf-8")
    n_ts = sum(1 for _ in t2d.splitlines() if _.strip())
    n_ent, n_rel = len(survivors), len(RELATION2ID)
    t_tr = int(n_ts * args.train_frac); t_va = int(n_ts * (args.train_frac + args.valid_frac))
    e = e.sort_values(["time_id", "subj", "rel", "obj"])
    train = e[e["time_id"] < t_tr]; valid = e[(e["time_id"] >= t_tr) & (e["time_id"] < t_va)]; test = e[e["time_id"] >= t_va]

    (out / "stat.txt").write_text(f"{n_ent}\t{n_rel}\t{n_ts}\n", encoding="utf-8")
    with (out / "entity2id.txt").open("w", encoding="utf-8") as f:
        for o in survivors:
            r = id2[o]; f.write(f"{r['name']}\t{old2new[o]}\t{r['type']}\t{r['tid']}\n")
    with (out / "relation2id.txt").open("w", encoding="utf-8") as f:
        for r, i in RELATION2ID.items():
            f.write(f"{r}\t{i}\n")
    (out / "time2date.txt").write_text(t2d, encoding="utf-8")
    for name, part in (("train", train), ("valid", valid), ("test", test)):
        with (out / f"{name}.txt").open("w", encoding="utf-8") as f:
            for idx, (s, r, o, t) in enumerate(part[["subj", "rel", "obj", "time_id"]].itertuples(index=False, name=None)):
                f.write(f"{s}\t{r}\t{o}\t{t}\t{idx}\n")
    e[["subj", "rel", "obj", "time_id", "weight"]].to_csv(out / "edges_weighted.tsv", sep="\t", index=False)

    print(f"[core] weight>={args.min_weight}, degree>={args.min_degree}")
    print(f"[core] entities {n0_ent:,} -> {n_ent:,} ({100*n_ent/n0_ent:.1f}%)")
    print(f"[core] edges    {n0_edge:,} -> {len(e):,} ({100*len(e)/n0_edge:.1f}%)")
    print(f"[core] split: train={len(train):,} valid={len(valid):,} test={len(test):,}")
    print("[core] type dist:", pd.Series([id2[o]['type'] for o in survivors]).value_counts().to_dict())
    print("CORE_DONE")


if __name__ == "__main__":
    main()
