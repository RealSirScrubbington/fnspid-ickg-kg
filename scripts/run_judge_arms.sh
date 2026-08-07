#!/usr/bin/env bash
# Judge the tf / nb / strict audit packs (themes mode only) with the cached local model.
# Assumes .venv-judge and the HF model cache already exist on projects3 (built 2026-07-10).
# Outputs: audit/theme_judge_labels_{tf,nb,strict}.csv, audit/ARMS_DONE
set -e
P=/cs/student/projects3/csml/2025/dmaruev/fnspid-ickg-kg
cd "$P"
source goldbug_env.sh 2>/dev/null || true
export TMPDIR="$P/cache/tmp"; mkdir -p "$TMPDIR"
rm -f audit/ARMS_DONE audit/arms_failures.txt
.venv-judge/bin/python -c "import torch; assert torch.cuda.is_available(); print('cuda ok:', torch.cuda.get_device_name(0))"
cd "$P/audit"
for tag in tf nb strict; do
  echo "=== $tag $(date) ==="
  "$P/.venv-judge/bin/python" theme_judge.py --packs "audit_packs_${tag}.jsonl" \
      --out "theme_judge_labels_${tag}.csv" --mode themes \
      || echo "FAILED $tag" >> arms_failures.txt
done
touch ARMS_DONE
echo "=== all done $(date) ==="
