"""Lightweight theme-quality filter distilled from the precision audit's judge labels.

PRE-REGISTERED PROTOCOL (fixed before any model was fit; mirrors the frozen-rubric discipline).
Task: predict the audit's binary label (REAL vs noise) for a lenient EMERGING lifeline from
cheap, text-free features, so the audit becomes a deployable final pipeline stage with no LLM
at inference.

Data: the 282 judge-labelled lenient lifelines ONLY (other arms share storylines with lenient,
so pooling would leak; the template-filtered arm is reserved as a possible transfer test).
Labels are the validated judge's (11/11 controls, 93%/87.5% human agreement) - the filter is
a DISTILLATION of the judge and is disclosed as such; the 14 human-labelled lenient packs act
as an additional external check.

Features (all computable from frozen artifacts; windows match the audit's birth window, i.e. a
one-week decision delay in deployment; membership = the audit pack's top-12 accumulated members,
so the filter scores the same object the judge scored - both disclosed):
  detector outputs at birth: score, rank, size, freshness
  cohesion: n_binding, multi3_share, pair_coverage, mean_members
  composition: comp_share / concept_share of top-12 member types, debut_share
               (members whose first formation lies within 13 weeks before birth)
  template: tmpl_share_binding (share of binding articles carrying a v2 fingerprint)

Models: standardized logistic regression (class_weight balanced; interpretable headline model)
and HistGradientBoosting (flexibility comparator). No hyperparameter search.

Evaluation: 5-fold GroupKFold with birth-year groups, plus a strict temporal holdout
(train births < 2022, test >= 2022). Metrics: PR-AUC and precision at 68.2% REAL retention
(comparable to the fixed pair_coverage >= 0.5 baseline at 47.4%). Fixed comparison points:
that single-threshold baseline, the ~50% structural-filter ceiling, and the judge (teacher).
Hypothesis stated in advance: structure alone saturates near 50%, so the filter beats the
threshold baseline only if composition/template features carry semantic signal.

Run: KG_CORE_PATH=data/kg_600k_dedup_core .venv/Scripts/python -m dynamics.theme_filter
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
OUT = Path("data/dynamics/themes")
TRIPL = ["data/tripl_600k_0_dedup.csv", "data/tripl_600k_1_dedup.csv"]
CONCEPT_TYPES = {"CONCEPT", "FIN_INSTRUMENT", "ECON_INDICATOR"}
WIN_BEFORE, WIN_AFTER = pd.Timedelta(days=28), pd.Timedelta(days=7)


def build_features():
    from dynamics.loader import load_core
    from dynamics.velocity import formation_events, entity_week_matrix
    kg = load_core(drop_noise=True)

    lf = pd.read_csv(OUT / "theme_lifelines_lenient.csv")
    em = lf[lf.status == "EMERGING"].copy()
    em["birth_dt"] = pd.to_datetime(em["birth"])
    wk = pd.read_csv(OUT / "theme_weeks_lenient.csv")
    mem = pd.read_csv(OUT / "theme_members_lenient.csv").sort_values("excess", ascending=False)
    coh = pd.read_csv(OUT / "cohesion_stats.csv").set_index("theme")
    lab = pd.read_csv(OUT / "theme_judge_labels.csv").set_index("theme")
    tmpl_articles = set(open(OUT.parent.parent / "dynamics/themes/template_articles.txt"
                             if False else OUT / "template_articles.txt").read().split())

    # first-formation week per entity (for debut_share)
    M = entity_week_matrix(formation_events(kg), kg.n_entities, kg.n_times)
    first_week = np.full(kg.n_entities, 10**9)
    nz = np.argwhere(M > 0)
    for e, w in nz[np.argsort(nz[:, 1])][::-1]:
        first_week[e] = w
    date_of = {w: pd.Timestamp(kg.date(w)) for w in range(kg.n_times)}
    week_of = {v: k for k, v in date_of.items()}

    top = {t: g.head(12) for t, g in mem.groupby("theme")}
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

    # streaming pass: binding articles per theme + their template share
    arts = defaultdict(set)
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
                            arts[t].add(aid)
        print(f"  scanned {path}")

    rows = []
    for _, r in em.iterrows():
        t = r.theme
        if t not in top or t not in lab.index:
            continue
        bw = week_of.get(r.birth_dt)
        wrow = wk[(wk.theme == t) & (wk.week == bw)]
        g = top[t]
        ids = g["ent"].to_numpy()
        types = [str(kg.id2type.get(int(e), "?")) for e in ids]
        a = arts.get(t, set())
        rows.append({
            "theme": t,
            "birth_year": r.birth_dt.year,
            "label": int(lab.loc[t, "label"] == "REAL"),
            "score": float(wrow["score"].iloc[0]) if len(wrow) else 0.0,
            "rank": float(wrow["rank"].iloc[0]) if len(wrow) else 10.0,
            "size": float(wrow["size"].iloc[0]) if len(wrow) else len(ids),
            "fresh": float(r.fresh),
            "n_binding": float(coh.loc[t, "n_binding"]) if t in coh.index else 0.0,
            "multi3_share": float(coh.loc[t, "multi3_share"]) if t in coh.index else 0.0,
            "pair_coverage": float(coh.loc[t, "pair_coverage"]) if t in coh.index else 0.0,
            "mean_members": float(coh.loc[t, "mean_members"]) if t in coh.index else 0.0,
            "comp_share": np.mean([ty == "COMP" for ty in types]),
            "concept_share": np.mean([ty in CONCEPT_TYPES for ty in types]),
            "debut_share": np.mean([(bw - first_week[int(e)]) <= 13 and first_week[int(e)] <= bw
                                    for e in ids]) if bw is not None else 0.0,
            "tmpl_share_binding": (np.mean([aid in tmpl_articles for aid in a]) if a else 0.0),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "filter_features_lenient.csv", index=False)
    print(f"features: {len(df)} lifelines x {df.shape[1] - 3} features -> filter_features_lenient.csv")
    return df


FEATS = ["score", "rank", "size", "fresh", "n_binding", "multi3_share", "pair_coverage",
         "mean_members", "comp_share", "concept_share", "debut_share", "tmpl_share_binding"]


def precision_at_retention(y, p, retention):
    """Precision when the score threshold retains `retention` of the positives."""
    pos = np.sort(p[y == 1])[::-1]
    thr = pos[min(len(pos) - 1, int(np.ceil(retention * len(pos))) - 1)]
    kept = p >= thr
    return (y[kept] == 1).mean(), kept.sum()


def evaluate(df):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.metrics import average_precision_score, roc_auc_score

    X = df[FEATS].to_numpy()
    y = df["label"].to_numpy()
    groups = df["birth_year"].to_numpy()
    base_rate = y.mean()
    print(f"\nn={len(y)}, REAL rate {base_rate:.1%} (chance PR-AUC)")

    models = {
        "logistic": make_pipeline(StandardScaler(),
                                  LogisticRegression(max_iter=2000, class_weight="balanced")),
        "hist-gb": HistGradientBoostingClassifier(random_state=0),
    }
    for name, model in models.items():
        cv = GroupKFold(n_splits=5)
        p = cross_val_predict(model, X, y, cv=cv, groups=groups, method="predict_proba")[:, 1]
        pr = average_precision_score(y, p)
        roc = roc_auc_score(y, p)
        p68, n68 = precision_at_retention(y, p, 0.682)
        p50, n50 = precision_at_retention(y, p, 0.50)
        print(f"{name:9} grouped-CV: PR-AUC {pr:.3f} | ROC-AUC {roc:.3f} | "
              f"precision@68%REAL {p68:.1%} (keeps {n68}) | @50%REAL {p50:.1%} (keeps {n50})")
        if name == "logistic":
            model.fit(X, y)
            lr = model.named_steps["logisticregression"]
            coefs = sorted(zip(FEATS, lr.coef_[0]), key=lambda kv: -abs(kv[1]))
            print("  standardized coefficients:",
                  ", ".join(f"{f}={c:+.2f}" for f, c in coefs))
        df[f"p_{name}"] = p

    # strict temporal holdout
    tr, te = df.birth_year < 2022, df.birth_year >= 2022
    print(f"\ntemporal holdout: train {tr.sum()} (<2022), test {te.sum()} (>=2022), "
          f"test REAL rate {y[te.to_numpy()].mean():.1%}")
    for name, model in models.items():
        model.fit(X[tr.to_numpy()], y[tr.to_numpy()])
        p = model.predict_proba(X[te.to_numpy()])[:, 1]
        yt = y[te.to_numpy()]
        pr = average_precision_score(yt, p)
        p68, n68 = precision_at_retention(yt, p, 0.682)
        print(f"{name:9} holdout: PR-AUC {pr:.3f} | precision@68%REAL {p68:.1%} (keeps {n68})")

    # external check: the 14 human-labelled lenient packs
    hum = pd.read_csv(OUT / "audit_sample_human.csv")
    hum = hum[hum.role == "sample"].copy()
    hum["human"] = (hum["label_REAL_TEMPLATE_INCOHERENT"].astype(str).str.strip().str.upper()
                    == "REAL").astype(int)
    j = df.set_index("theme").join(hum.set_index("theme")[["human"]], how="inner")
    agree = ((j["p_logistic"] >= 0.5).astype(int) == j["human"]).mean()
    print(f"\nhuman check (n={len(j)} lenient packs): filter-vs-human binary agreement {agree:.0%}")

    print("\nfixed baselines: pair_coverage>=0.5 single threshold = 47.4% @ 68.2% retention; "
          "structural-sweep ceiling ~50%; judge (teacher, uses text) = the labels themselves.")


if __name__ == "__main__":
    f = OUT / "filter_features_lenient.csv"
    df = pd.read_csv(f) if f.exists() else build_features()
    evaluate(df)
