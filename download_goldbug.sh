#!/usr/bin/env bash
# Download Qwen2.5-14B-Instruct (base) + ICKG-v4.2 adapter into projects3 HF cache.
set -euo pipefail
PROJ=/cs/student/projects3/csml/2025/dmaruev/fnspid-ickg-kg
CACHE=$PROJ/cache
export HF_HOME=$CACHE/hf TRITON_CACHE_DIR=$CACHE/triton XDG_CACHE_HOME=$CACHE/xdg PIP_CACHE_DIR=$CACHE/pip
. "$PROJ/.venv-blackwell/bin/activate"
python - <<'PY'
from huggingface_hub import snapshot_download
print("downloading base: unsloth/Qwen2.5-14B-Instruct (safetensors) ...", flush=True)
p = snapshot_download("unsloth/Qwen2.5-14B-Instruct",
                      allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.txt"])
print("base ->", p, flush=True)
print("downloading adapter: victorlxh/ICKG-v4.2 ...", flush=True)
p2 = snapshot_download("victorlxh/ICKG-v4.2")
print("adapter ->", p2, flush=True)
PY
echo "DOWNLOAD_DONE"
