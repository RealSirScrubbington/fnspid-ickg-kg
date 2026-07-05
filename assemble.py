"""assemble.py -- Stage 3: merge the shard triplet CSVs into FinDKG-format files.

Light surface-form entity normalisation only (whitespace/strip + a tiny alias map);
full entity resolution is OUT OF SCOPE (a parallel student's thesis). Buckets triplets
by a configurable time resolution (weekly default) using the article PUBLICATION date,
integer-encodes entities/relations, splits train/valid/test CHRONOLOGICALLY by time-step,
writes the files, and round-trip validates (counts + no future edges leaking into earlier
splits).

  python assemble.py --inputs data/tripl_full_0.csv data/tripl_full_1.csv \
                     --outdir data/kg --time-resolution week
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ickg_kg.schema import RELATION2ID, ENTITY_TYPE2ID, normalize_relation, normalize_entity_type

# Intentionally tiny "obvious alias" map -- NOT a canonicalisation service.
ALIASES: dict[str, str] = {}


def norm_entity(s) -> str:
    """Whitespace-collapsed surface form, alias-mapped by lowercase key (original casing kept)."""
    s = " ".join(str(s).split()).strip()           # collapse all whitespace (TSV-safe)
    return ALIASES.get(s.lower(), s)


def bucket_start(dt: pd.Series, res: str) -> pd.Series:
    """Start timestamp of the week/fortnight/month bucket containing each date."""
    if res == "week":
        return dt.dt.to_period("W").dt.start_time
    if res == "month":
        return dt.dt.to_period("M").dt.start_time
    if res == "fortnight":
        base = dt.min().normalize()
        return base + pd.to_timedelta((dt - base).dt.days // 14 * 14, unit="D")
    raise SystemExit(f"unknown resolution: {res}")


def main() -> None:
    """Merge the shard CSVs, normalise entities/relations/types, bucket by publication date,
    integer-encode, split chronologically by time-step, write the FinDKG flat files, and
    round-trip validate the output."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--time-resolution", default="week", choices=["week", "fortnight", "month"])
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--valid-frac", type=float, default=0.15)
    args = ap.parse_args()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    # 1. load + merge, keep only schema-valid triplets
    df = pd.concat([pd.read_csv(p) for p in args.inputs], ignore_index=True)
    df = df[df["valid"].astype(str).str.lower() == "true"].copy()
    df["h"] = df["h"].map(norm_entity); df["o"] = df["o"].map(norm_entity)
    df["h_type"] = df["h_type"].map(normalize_entity_type)
    df["o_type"] = df["o_type"].map(normalize_entity_type)
    df["r"] = df["r"].map(normalize_relation)
    df = df[(df["h"] != "") & (df["o"] != "")].dropna(subset=["h_type", "o_type", "r"])
    print(f"[assemble] valid triplets after normalisation: {len(df)}")

    # 2. entity -> single type (majority vote over all head+tail mentions) + frequency
    et = pd.concat([
        df[["h", "h_type"]].rename(columns={"h": "e", "h_type": "t"}),
        df[["o", "o_type"]].rename(columns={"o": "e", "o_type": "t"})], ignore_index=True)
    ent_type = (et.groupby(["e", "t"]).size().reset_index(name="n")
                  .sort_values("n", ascending=False).drop_duplicates("e").set_index("e")["t"])
    entities = list(et["e"].value_counts().index)          # most frequent first -> low ids
    ent2id = {e: i for i, e in enumerate(entities)}

    # 3. time buckets from PUBLICATION date
    df["dt"] = pd.to_datetime(df["date"].astype(str).str.replace(" UTC", "", regex=False), errors="coerce")
    df = df.dropna(subset=["dt"])
    df["bstart"] = bucket_start(df["dt"], args.time_resolution)
    buckets = sorted(df["bstart"].unique())
    time2id = {b: i for i, b in enumerate(buckets)}
    df["time_id"] = df["bstart"].map(time2id)

    # 4. quadruples (dedup per (s,r,o,t)); keep mention-count as edge weight
    df["subj"] = df["h"].map(ent2id); df["obj"] = df["o"].map(ent2id); df["rel"] = df["r"].map(RELATION2ID)
    wq = (df.groupby(["subj", "rel", "obj", "time_id"]).size()
            .reset_index(name="weight").sort_values(["time_id", "subj", "rel", "obj"]))
    n_ent, n_rel, n_ts = len(entities), len(RELATION2ID), len(buckets)
    print(f"[assemble] entities={n_ent} relations={n_rel} timesteps={n_ts} edges={len(wq)}")

    # 5. chronological split by time-step
    t_tr = int(n_ts * args.train_frac); t_va = int(n_ts * (args.train_frac + args.valid_frac))
    train = wq[wq["time_id"] < t_tr]; valid = wq[(wq["time_id"] >= t_tr) & (wq["time_id"] < t_va)]
    test = wq[wq["time_id"] >= t_va]

    # 6. write FinDKG-format files (tab-separated)
    (out / "stat.txt").write_text(f"{n_ent}\t{n_rel}\t{n_ts}\n", encoding="utf-8")
    with (out / "entity2id.txt").open("w", encoding="utf-8") as f:
        for e in entities:
            t = ent_type[e]
            f.write(f"{e}\t{ent2id[e]}\t{t}\t{ENTITY_TYPE2ID[t]}\n")
    with (out / "relation2id.txt").open("w", encoding="utf-8") as f:
        for r, i in RELATION2ID.items():
            f.write(f"{r}\t{i}\n")
    with (out / "time2date.txt").open("w", encoding="utf-8") as f:
        for b, i in time2id.items():
            f.write(f"{i}\t{pd.Timestamp(b).date()}\n")
    for name, part in (("train", train), ("valid", valid), ("test", test)):
        with (out / f"{name}.txt").open("w", encoding="utf-8") as f:
            for idx, (s, r, o, t) in enumerate(part[["subj", "rel", "obj", "time_id"]].itertuples(index=False, name=None)):
                f.write(f"{s}\t{r}\t{o}\t{t}\t{idx}\n")
    wq[["subj", "rel", "obj", "time_id", "weight"]].to_csv(out / "edges_weighted.tsv", sep="\t", index=False)

    _validate(out, n_ent, n_rel, n_ts, len(train), len(valid), len(test), t_tr, t_va)
    print(f"[assemble] wrote KG to {out}  (train={len(train)} valid={len(valid)} test={len(test)})")


def _validate(out, n_ent, n_rel, n_ts, ntr, nva, nte, t_tr, t_va) -> None:
    """Re-read the written files and assert row counts, id ranges, and split time ordering
    (no future edges in earlier splits)."""
    e = sum(1 for _ in (out / "entity2id.txt").open(encoding="utf-8"))
    r = sum(1 for _ in (out / "relation2id.txt").open(encoding="utf-8"))
    ts = sum(1 for _ in (out / "time2date.txt").open(encoding="utf-8"))
    assert e == n_ent, f"entity2id rows {e} != {n_ent}"
    assert r == n_rel == 15, f"relation count {r}"
    assert ts == n_ts, f"time rows {ts} != {n_ts}"
    times = {}
    for name in ("train", "valid", "test"):
        tids = [int(l.split("\t")[3]) for l in (out / f"{name}.txt").open(encoding="utf-8") if l.strip()]
        for l in (out / f"{name}.txt").open(encoding="utf-8"):
            s, rel, o, t, _ = l.split("\t")
            assert int(s) < n_ent and int(o) < n_ent and int(rel) < n_rel and int(t) < n_ts, "id out of range"
        times[name] = (min(tids), max(tids)) if tids else (None, None)
    # no future edges leaking earlier: max(train) < min(valid) <= ... < min(test)
    if times["train"][1] is not None and times["valid"][0] is not None:
        assert times["train"][1] < times["valid"][0] == t_tr or times["train"][1] < t_tr, "train/valid time overlap"
    if times["valid"][1] is not None and times["test"][0] is not None:
        assert times["valid"][1] < t_va <= times["test"][0], "valid/test time overlap"
    print(f"[assemble] round-trip OK | time ranges train={times['train']} valid={times['valid']} test={times['test']}")


if __name__ == "__main__":
    main()
