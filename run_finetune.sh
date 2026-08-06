#!/usr/bin/env bash
# Fine-tune the small judge (one GPU session). Reuses .venv-ada (torch cu126); installs the
# training deps into it on first run. Outputs: judge_lora/ + finetune log with the gate report.
set -e
P=/cs/student/projects3/csml/2025/dmaruev/fnspid-ickg-kg
cd "$P"
source goldbug_env.sh 2>/dev/null || true
export TMPDIR="$P/cache/tmp"; mkdir -p "$TMPDIR"
rm -f FINETUNE_DONE
.venv-ada/bin/python -c "import peft" 2>/dev/null || \
  .venv-ada/bin/pip install --no-cache-dir --quiet "transformers==4.53.2" peft accelerate
PYTHONPATH="$P" .venv-ada/bin/python -m dynamics.judge_finetune \
    --data-dir audit --out-dir judge_lora > finetune_run.log 2>&1
touch FINETUNE_DONE
echo done
