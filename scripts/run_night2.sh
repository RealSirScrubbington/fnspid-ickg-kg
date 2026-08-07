#!/bin/bash
# Follow-up batch: (A) 1.5B judge scores for the remaining three arms;
# (B/C) tf-substrate RE-GCN retrains at seeds 1 and 2 for a mean +/- spread.
set -u
P=/cs/student/projects3/csml/2025/dmaruev/fnspid-ickg-kg
cd "$P"
exec >> night2_run.log 2>&1
echo "=== night2 start $(date) on $(hostname) ==="

PYTHONPATH="$P" .venv-ada/bin/python -m dynamics.judge_score --data-dir audit \
  --adapter judge_lora --arms strict,nb,tf_r15 --out audit/judge_scores_extra.csv
A=$?
echo "stage A exit $A $(date)"

KG_CORE_PATH=data/kg_600k_dedup_tf_core PYTHONPATH="$P" \
  .venv-ada/bin/python -m dynamics.regcn --tag tf_s1 --seed 1
B=$?
echo "stage B exit $B $(date)"

KG_CORE_PATH=data/kg_600k_dedup_tf_core PYTHONPATH="$P" \
  .venv-ada/bin/python -m dynamics.regcn --tag tf_s2 --seed 2
C=$?
echo "stage C exit $C $(date)"

touch NIGHT2_DONE
echo "=== night2 end $(date) (A=$A B=$B C=$C) ==="
