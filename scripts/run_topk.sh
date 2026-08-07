#!/usr/bin/env bash
# Worked-example dump: retrain canonical RE-GCN (seed 0) with top-10 candidate dumping.
# Outputs: data/dynamics/linkpred/regcn_ranks_topk.npz (+results CSV), TOPK_DONE
set -e
P=/cs/student/projects3/csml/2025/dmaruev/fnspid-ickg-kg
cd "$P"
source goldbug_env.sh 2>/dev/null || true
export TMPDIR="$P/cache/tmp"; mkdir -p "$TMPDIR"
rm -f TOPK_DONE
export KG_CORE_PATH=data/kg_600k_dedup_core
.venv-ada/bin/python -m dynamics.regcn --epochs 60 --seed 0 --tag topk --dump-topk 10 \
    > topk_run.log 2>&1
touch TOPK_DONE
echo done
