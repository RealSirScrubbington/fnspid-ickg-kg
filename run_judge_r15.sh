#!/usr/bin/env bash
# Judge the tf_r15 (Louvain resolution 1.5 on the template-filtered substrate) audit packs.
# Outputs: audit/theme_judge_labels_tf_r15.csv, audit/R15_DONE
set -e
P=/cs/student/projects3/csml/2025/dmaruev/fnspid-ickg-kg
cd "$P"
source goldbug_env.sh 2>/dev/null || true
export TMPDIR="$P/cache/tmp"; mkdir -p "$TMPDIR"
rm -f audit/R15_DONE audit/r15_failures.txt
cd "$P/audit"
echo "=== tf_r15 $(date) ==="
"$P/.venv-judge/bin/python" theme_judge.py --packs audit_packs_tf_r15.jsonl \
    --out theme_judge_labels_tf_r15.csv --mode themes \
    || echo "FAILED tf_r15" >> r15_failures.txt
touch R15_DONE
echo "=== done $(date) ==="
