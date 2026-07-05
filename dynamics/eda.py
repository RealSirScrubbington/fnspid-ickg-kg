"""Temporal EDA of the core KG: per-week activity, edge formation, graph growth, and
relation mix over time. Prints a summary, writes reusable weekly aggregates, and saves an
overview figure. Run: `.venv/Scripts/python -m dynamics.eda`
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dynamics.loader import load_core

OUT = Path("data/dynamics/eda")
OUT.mkdir(parents=True, exist_ok=True)


def weekly_aggregates(kg) -> pd.DataFrame:
    """Per-week activity/formation/growth table. Returns (df indexed by week, first) where
    `first` holds each relationship's first-appearance row (formation = min time per (s,r,o))."""
    e = kg.edges
    idx = pd.RangeIndex(kg.n_times, name="time")

    edges_wk = e.groupby("time").size().reindex(idx, fill_value=0)            # edge-instances / week
    formation_time = e.groupby(["subj", "rel", "obj"])["time"].transform("min")
    first = e[e["time"] == formation_time]
    formations_wk = first.groupby("time").size().reindex(idx, fill_value=0)   # NEW relationships / week

    ent_long = pd.concat([e[["time", "subj"]].rename(columns={"subj": "ent"}),
                          e[["time", "obj"]].rename(columns={"obj": "ent"})], ignore_index=True)
    active_ent_wk = ent_long.groupby("time")["ent"].nunique().reindex(idx, fill_value=0)
    ent_first = ent_long.groupby("ent")["time"].min()
    new_ent_wk = ent_first.value_counts().reindex(idx, fill_value=0).sort_index()

    df = pd.DataFrame({
        "date": kg.times.reindex(idx).values,
        "edge_instances": edges_wk.values,
        "formations": formations_wk.values,         # first appearance of a (s,r,o)
        "active_entities": active_ent_wk.values,
        "new_entities": new_ent_wk.values,
        "cum_relationships": formations_wk.cumsum().values,
        "cum_entities": new_ent_wk.cumsum().values,
    }, index=idx)
    df["split"] = [kg.split_of(t) for t in idx]
    return df, first


def relation_mix(kg) -> pd.DataFrame:
    """Week x relation-name matrix of edge-instance counts."""
    e = kg.edges
    idx = pd.RangeIndex(kg.n_times, name="time")
    mix = e.groupby(["time", "rel"]).size().unstack(fill_value=0).reindex(idx, fill_value=0)
    mix.columns = [kg.id2rel[c] for c in mix.columns]
    return mix


def shade_splits(ax, kg):
    """Shade the train/valid/test week ranges as background bands on ax."""
    colors = {"train": "#e8f0fe", "valid": "#fff3e0", "test": "#fde8e8"}
    for name, (lo, hi) in kg.splits.items():
        ax.axvspan(lo, hi - 1, color=colors[name], alpha=0.6, zorder=0)


def main():
    """Print the summary stats, save weekly_stats.csv and the 4-panel overview figure."""
    kg = load_core(drop_noise=True)   # align with every other dynamics analysis (audit fix)
    df, first = weekly_aggregates(kg)
    mix = relation_mix(kg)
    df.to_csv(OUT / "weekly_stats.csv")

    n_rel_distinct = len(first)
    h = kg.n_times // 2
    print(f"=== CORE temporal EDA ===")
    print(f"{kg.n_entities:,} entities | {len(kg.edges):,} edge-instances | "
          f"{n_rel_distinct:,} distinct relationships | {kg.n_times} weeks "
          f"({kg.date(0)} -> {kg.date(kg.n_times-1)})")
    print(f"edge-instances/wk : mean {df.edge_instances.mean():.0f}  med {df.edge_instances.median():.0f}  "
          f"min {df.edge_instances.min()}  max {df.edge_instances.max()}  zero-weeks {int((df.edge_instances==0).sum())}")
    print(f"formations/wk     : mean {df.formations.mean():.0f}  "
          f"H1 {df.formations[:h].mean():.0f}  H2 {df.formations[h:].mean():.0f}")
    print(f"active entities/wk: mean {df.active_entities.mean():.0f}  max {df.active_entities.max()}")
    print(f"recurring fraction: {1 - n_rel_distinct/len(kg.edges):.1%} of edge-instances repeat a prior relationship")
    print(f"edge weight       : mean {kg.edges.weight.mean():.2f}  max {int(kg.edges.weight.max())}")
    for s in ("train", "valid", "test"):
        sub = df[df.split == s]
        print(f"  {s:5}: weeks {sub.index.min()}-{sub.index.max()} | "
              f"edge-instances {int(sub.edge_instances.sum()):,} | formations {int(sub.formations.sum()):,}")
    print("top relations:", kg.edges.groupby("rel").size().sort_values(ascending=False)
          .head(6).rename(kg.id2rel).to_dict())

    # --- overview figure ---
    x = df.index.values
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    a = axes[0, 0]; shade_splits(a, kg)
    a.plot(x, df.edge_instances, lw=1.0, color="#1a73e8", label="edge-instances")
    a.plot(x, df.formations, lw=1.0, color="#d93025", label="formations (new)")
    a.set_title("Weekly activity vs edge formation"); a.set_xlabel("week"); a.legend(fontsize=8)

    a = axes[0, 1]; shade_splits(a, kg)
    a.plot(x, df.cum_relationships, color="#188038", label="distinct relationships")
    a.plot(x, df.cum_entities, color="#9334e6", label="distinct entities")
    a.set_title("Cumulative graph growth"); a.set_xlabel("week"); a.legend(fontsize=8)

    a = axes[1, 0]; shade_splits(a, kg)
    a.plot(x, df.active_entities, color="#e37400")
    a.set_title("Active entities per week"); a.set_xlabel("week")

    a = axes[1, 1]
    top = mix.sum().sort_values(ascending=False).head(8).index
    share = mix.div(mix.sum(axis=1).replace(0, np.nan), axis=0)
    a.stackplot(x, *[share[c].fillna(0).rolling(4, min_periods=1).mean() for c in top],
                labels=list(top))
    a.set_title("Relation mix over time (top 8, 4-wk smoothed)"); a.set_xlabel("week")
    a.set_ylim(0, 1); a.legend(fontsize=6, loc="upper left", ncol=2)

    fig.suptitle(f"Core KG temporal signal — {kg.n_entities:,} entities, {kg.n_times} weeks "
                 f"({kg.date(0)} to {kg.date(kg.n_times-1)})", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "temporal_overview.png", dpi=110)
    print(f"\nsaved {OUT/'temporal_overview.png'} and {OUT/'weekly_stats.csv'}")


if __name__ == "__main__":
    main()
