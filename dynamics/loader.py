"""Load the denoised core KG (FinDKG flat files) into time-indexed structures.

The core is small (~46k entities / ~308k edge-week rows / 365 weekly buckets), so everything
fits comfortably in memory. Edge ids are the core's contiguous re-indexed ids.

Terminology used throughout the dynamics code:
  - edge-instance  : one row of edges_weighted.tsv = a (subj, rel, obj) seen in a given week.
  - relationship   : a distinct (subj, rel, obj) triple, ignoring time.
  - formation      : the FIRST week a relationship appears (the edge-formation event).
"""
from dataclasses import dataclass
import os
from pathlib import Path
import pandas as pd

# Causal/outcome relations flagged by the blinded-extraction test as entity-dependent (~30% drop
# under blinding). The other 10 relations are structural (who operates where, controls/owns what).
IMPACT_RELATIONS = {"Impact", "Positive_Impact_On", "Negative_Impact_On", "Raise", "Decrease"}


@dataclass
class CoreKG:
    edges: pd.DataFrame          # columns: subj, rel, obj, time, weight  (all int ids; time = week id)
    id2name: dict                # entity id -> surface form
    id2type: dict                # entity id -> entity type (ORG, COMP, GPE, ...)
    id2rel: dict                 # relation id -> relation name
    times: pd.Series             # time_id -> pd.Timestamp (week-bucket start)
    n_entities: int
    n_relations: int
    n_times: int
    splits: dict                 # 'train'/'valid'/'test' -> (lo_time, hi_time)  half-open

    def date(self, t: int) -> str:
        return self.times.get(t, pd.NaT).date().isoformat()

    def split_of(self, t: int) -> str:
        for name, (lo, hi) in self.splits.items():
            if lo <= t < hi:
                return name
        return "?"


def load_core(path: str | None = None, train_frac: float = 0.70, valid_frac: float = 0.15,
              drop_noise: bool = False, drop_relations: set | None = None, verbose: bool = True) -> CoreKG:
    path = path or os.environ.get("KG_CORE_PATH", "data/kg_core")   # env switch: e.g. data/kg_600k_core
    p = Path(path)
    ent = pd.read_csv(p / "entity2id.txt", sep="\t", header=None, names=["name", "id", "type", "tid"],
                      keep_default_na=False)   # an entity literally named "nan" must stay a string
    rel = pd.read_csv(p / "relation2id.txt", sep="\t", header=None, names=["name", "id"])
    t2d = pd.read_csv(p / "time2date.txt", sep="\t", header=None, names=["tid", "date"])
    times = pd.to_datetime(t2d.set_index("tid")["date"])

    ew = pd.read_csv(p / "edges_weighted.tsv", sep="\t").rename(columns={"time_id": "time"})
    edges = ew[["subj", "rel", "obj", "time", "weight"]].copy()
    id2name = dict(zip(ent["id"], ent["name"]))

    if drop_noise:
        from dynamics.noise import noise_entity_ids
        nz = noise_entity_ids(id2name)
        before = len(edges)
        edges = edges[~edges["subj"].isin(nz) & ~edges["obj"].isin(nz)].reset_index(drop=True)
        if verbose:
            print(f"[noise] dropped {len(nz)} templated entities, "
                  f"{before-len(edges):,} edge-instances ({100*(before-len(edges))/before:.1f}%)")

    if drop_relations:
        drop_ids = set(rel[rel["name"].isin(drop_relations)]["id"])
        before = len(edges)
        edges = edges[~edges["rel"].isin(drop_ids)].reset_index(drop=True)
        if verbose:
            print(f"[rel-filter] dropped {len(drop_ids)} relations, {before-len(edges):,} edge-instances")

    n_ent, n_rel, n_ts = (int(x) for x in (p / "stat.txt").read_text().split())
    t_tr = int(n_ts * train_frac)
    t_va = int(n_ts * (train_frac + valid_frac))
    splits = {"train": (0, t_tr), "valid": (t_tr, t_va), "test": (t_va, n_ts)}

    return CoreKG(
        edges=edges,
        id2name=id2name,
        id2type=dict(zip(ent["id"], ent["type"])),
        id2rel=dict(zip(rel["id"], rel["name"])),
        times=times,
        n_entities=n_ent, n_relations=n_rel, n_times=n_ts,
        splits=splits,
    )


if __name__ == "__main__":
    kg = load_core()
    print(f"{kg.n_entities:,} entities | {len(kg.edges):,} edge-instances | {kg.n_times} weeks")
    print(f"dates: {kg.date(0)} -> {kg.date(kg.n_times-1)}")
    print(f"splits (weeks): {kg.splits}")
