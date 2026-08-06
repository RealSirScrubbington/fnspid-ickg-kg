#!/bin/bash
# Chained night run: (A) fine-tuned judge scores every audit pack; (B) RE-GCN retrain
# on the template-filtered core. Each stage gets its own DONE flag so a partial night
# still yields stage-A results. Launched detached by launch_night.sh.
set -u
P=/cs/student/projects3/csml/2025/dmaruev/fnspid-ickg-kg
cd "$P"
exec >> night_run.log 2>&1
echo "=== night run start $(date) on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.used --format=csv,noheader

echo "--- stage A: judge scoring sweep ---"
PYTHONPATH="$P" .venv-ada/bin/python -m dynamics.judge_score --data-dir audit --adapter judge_lora
A=$?
echo "stage A exit $A $(date)"
[ "$A" -eq 0 ] && touch JUDGE_SCORES_DONE

echo "--- stage B: RE-GCN retrain on tf core ---"
KG_CORE_PATH=data/kg_600k_dedup_tf_core PYTHONPATH="$P" \
  .venv-ada/bin/python -m dynamics.regcn --tag tf
B=$?
echo "stage B exit $B $(date)"
[ "$B" -eq 0 ] && touch REGCN_TF_DONE

touch NIGHT_DONE
echo "=== night run end $(date) (A=$A B=$B) ==="
