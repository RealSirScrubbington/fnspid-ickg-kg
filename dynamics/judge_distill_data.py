"""Build the distillation dataset for the small fine-tuned judge.

PRE-REGISTERED PROTOCOL (fixed before any training):
- Examples: every judge-labelled audit pack from all five arms (lenient, tf, strict, nb,
  tf_r15), rendered EXACTLY as the teacher saw them (same rubric system prompt, same pack
  rendering from theme_judge.py), completion = the teacher's three-way label.
- Leakage control: packs are assigned STORYLINE GROUPS = connected components of cross-arm
  member overlap (Jaccard >= 0.3 on top-12 member-name sets). Train/test splits are always
  group-disjoint; the primary held-out set is additionally temporal (birth >= 2022).
- Evaluation for the fine-tuned judge (fixed): (1) the same 11-control gate every judge faces
  (fail = not quotable); (2) three-way and binary agreement with the teacher on held-out
  groups; (3) binary agreement with the 24 human-labelled packs; (4) precision-at-retention
  as a filter. Baselines: zero-shot small models (failed the gate), the feature filter
  (PR-AUC 0.602), the 14B teacher (ceiling).

Outputs: data/dynamics/themes/distill_{train,test}.jsonl (+ groups CSV).
Run: .venv/Scripts/python -m dynamics.judge_distill_data
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
OUT = Path("data/dynamics/themes")

from dynamics.theme_judge import RUBRIC, render  # identical rendering to the teacher

ARMS = [("lenient", "audit_packs.jsonl", "theme_judge_labels.csv", "theme_lifelines_lenient.csv"),
        ("tf", "audit_packs_tf.jsonl", "theme_judge_labels_tf.csv", "theme_lifelines_tf.csv"),
        ("strict", "audit_packs_strict.jsonl", "theme_judge_labels_strict.csv",
         "theme_lifelines_strict.csv"),
        ("nb", "audit_packs_nb.jsonl", "theme_judge_labels_nb.csv", "theme_lifelines_nb.csv"),
        ("tf_r15", "audit_packs_tf_r15.jsonl", "theme_judge_labels_tf_r15.csv",
         "theme_lifelines_tf_r15.csv")]


def main():
    rows = []
    for arm, packs_f, labels_f, lf_f in ARMS:
        labels = pd.read_csv(OUT / labels_f).set_index("theme")
        births = pd.read_csv(OUT / lf_f).set_index("theme")["birth"]
        for line in open(OUT / packs_f, encoding="utf-8"):
            p = json.loads(line)
            t = p["theme"]
            if t not in labels.index or labels.loc[t, "label"] == "PARSE_FAIL":
                continue
            rows.append({"arm": arm, "theme": t, "label": labels.loc[t, "label"],
                         "birth": str(births.get(t, "")),
                         "members": frozenset(m["name"] for m in p["members"]),
                         "prompt": render(p)})
    df = pd.DataFrame(rows)
    print(f"{len(df)} labelled packs across {df.arm.nunique()} arms | "
          f"labels: {df.label.value_counts().to_dict()}")

    # storyline groups: union-find over cross-arm member overlap (Jaccard >= 0.3)
    parent = list(range(len(df)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    mems = df["members"].tolist()
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            a, b = mems[i], mems[j]
            if len(a & b) / len(a | b) >= 0.3:
                parent[find(i)] = find(j)
    df["group"] = [find(i) for i in range(len(df))]
    print(f"{df.group.nunique()} storyline groups (largest {df.group.value_counts().iloc[0]})")

    # primary split: temporal (birth >= 2022 -> test), then enforce group-disjointness by
    # moving any group that straddles the boundary entirely into TEST (conservative)
    df["birth_dt"] = pd.to_datetime(df["birth"], errors="coerce")
    df["is_test"] = df["birth_dt"] >= pd.Timestamp("2022-01-01")
    straddle = df.groupby("group")["is_test"].transform(lambda s: s.any())
    df.loc[straddle & ~df.is_test, "is_test"] = True   # whole group goes to test
    tr, te = df[~df.is_test], df[df.is_test]
    print(f"train {len(tr)} packs ({tr.group.nunique()} groups) | "
          f"test {len(te)} packs ({te.group.nunique()} groups) | "
          f"test label mix {te.label.value_counts().to_dict()}")

    for name, part in (("train", tr), ("test", te)):
        with open(OUT / f"distill_{name}.jsonl", "w", encoding="utf-8") as f:
            for _, r in part.iterrows():
                f.write(json.dumps({
                    "arm": r.arm, "theme": int(r.theme), "group": int(r.group),
                    "label": r.label,
                    "messages": [{"role": "system", "content": RUBRIC},
                                 {"role": "user", "content": r.prompt},
                                 {"role": "assistant",
                                  "content": '{"label": "' + r.label + '", "rationale": ""}'}],
                }, ensure_ascii=False) + "\n")
    df.drop(columns=["members", "prompt"]).to_csv(OUT / "distill_groups.csv", index=False)
    print("wrote distill_train.jsonl / distill_test.jsonl / distill_groups.csv")


if __name__ == "__main__":
    main()
