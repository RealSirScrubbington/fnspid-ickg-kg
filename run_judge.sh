#!/usr/bin/env bash
# One-shot precision-audit judge run (any CS lab GPU host with the projects3 mount).
# Builds .venv-judge (vLLM) if missing, then judges audit_packs.jsonl (themes mode) and
# cohesion_packs.jsonl (pairs mode) with a locally hosted Qwen2.5-14B-Instruct-AWQ.
# Outputs: audit/theme_judge_labels.csv, audit/pair_judge_verdicts.csv, audit/JUDGE_DONE
set -e
P=/cs/student/projects3/csml/2025/dmaruev/fnspid-ickg-kg
cd "$P"
source goldbug_env.sh 2>/dev/null || true
export TMPDIR="$P/cache/tmp"; mkdir -p "$TMPDIR"
rm -f audit/JUDGE_DONE audit/failures.txt

if [ ! -x .venv-judge/bin/python ]; then
  echo "=== building .venv-judge $(date) ==="
  rm -rf .venv-judge
  "$P/uv-python/cpython-3.12-linux-x86_64-gnu/bin/python3.12" -m venv .venv-judge
  .venv-judge/bin/pip install --no-cache-dir --quiet --upgrade pip
  # vllm 0.9.x pins torch 2.7 (cu126 default) - matches the 12.6 driver on the lab hosts;
  # latest vllm ships torch for newer CUDA and fails cuda_init on these machines
  .venv-judge/bin/pip install --no-cache-dir --quiet "vllm==0.9.2"
fi
# vllm 0.9.x clashes with transformers >= 4.54 (duplicate 'aimv2' config registration)
.venv-judge/bin/pip install --no-cache-dir --quiet "transformers==4.53.2"
.venv-judge/bin/python -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

cd "$P/audit"
echo "=== themes mode $(date) ==="
"$P/.venv-judge/bin/python" theme_judge.py --packs audit_packs.jsonl \
    --out theme_judge_labels.csv --mode themes \
    || echo "FAILED themes" >> failures.txt
echo "=== pairs mode $(date) ==="
"$P/.venv-judge/bin/python" theme_judge.py --packs cohesion_packs.jsonl \
    --out pair_judge_verdicts.csv --mode pairs \
    || echo "FAILED pairs" >> failures.txt
touch JUDGE_DONE
echo "=== all done $(date) ==="
