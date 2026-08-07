#!/usr/bin/env bash
# Velocity-feature ablation on the canonical dedup core: 2 arms x 3 seeds,
# early stopping on validation weeks per run, sequential (single GPU).
# Baseline arm (flag off) is bit-identical to the frozen model. Outputs:
# data/dynamics/linkpred/regcn_results_abl_{base,vel}_s{0,1,2}.csv
# data/dynamics/linkpred/regcn_ranks_abl_{base,vel}_s{0,1,2}.npz (per-query, paired)
set -u
P=/cs/student/projects3/csml/2025/dmaruev/fnspid-ickg-kg
cd "$P"
source goldbug_env.sh
PY=.venv-ada/bin/python
export KG_CORE_PATH=data/kg_600k_dedup_core
mkdir -p data/dynamics/linkpred logs_abl
rm -f ABLATION_DONE
for s in 0 1 2; do
  echo "=== base seed $s $(date) ==="
  $PY -m dynamics.regcn --epochs 60 --seed "$s" --tag "abl_base_s$s" \
      > "logs_abl/base_s$s.log" 2>&1 || echo "FAILED base_s$s" >> logs_abl/failures.txt
  echo "=== vel seed $s $(date) ==="
  $PY -m dynamics.regcn --epochs 60 --seed "$s" --velocity-features --tag "abl_vel_s$s" \
      > "logs_abl/vel_s$s.log" 2>&1 || echo "FAILED vel_s$s" >> logs_abl/failures.txt
done
touch ABLATION_DONE
echo "all done $(date)"
