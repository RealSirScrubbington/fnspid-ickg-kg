#!/usr/bin/env bash
# Blackwell (2x RTX PRO 6000) env on goldbug via uv + Python 3.12 + vLLM.
# Everything (caches, python, uv) lives under projects3 -- never HOME (quota-full).
set -euo pipefail
PROJ=/cs/student/projects3/csml/2025/dmaruev/fnspid-ickg-kg
CACHE=$PROJ/cache
export HF_HOME=$CACHE/hf TRITON_CACHE_DIR=$CACHE/triton
export XDG_CACHE_HOME=$CACHE/xdg XDG_DATA_HOME=$CACHE/xdg-data PIP_CACHE_DIR=$CACHE/pip
export UV_CACHE_DIR=$CACHE/uv UV_PYTHON_INSTALL_DIR=$PROJ/uv-python UV_INSTALL_DIR=$PROJ/uv-bin
mkdir -p "$HF_HOME" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$XDG_DATA_HOME" \
         "$PIP_CACHE_DIR" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$UV_INSTALL_DIR"
cd "$PROJ"

echo "[0/4] reclaiming stale Ada venv"
rm -rf "$PROJ/.venv" "$PROJ/.venv-blackwell" 2>/dev/null || true

echo "[1/4] installing uv (no sudo, no HOME writes) -> $UV_INSTALL_DIR"
curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --no-modify-path
UV="$UV_INSTALL_DIR/uv"
"$UV" --version

echo "[2/4] creating Python 3.12 venv"
"$UV" venv --python 3.12 "$PROJ/.venv-blackwell"
. "$PROJ/.venv-blackwell/bin/activate"

echo "[3/4] installing vllm + data libs (pulls Blackwell-capable torch; large)"
"$UV" pip install vllm pandas pyarrow datasets huggingface_hub

echo "[4/4] verifying torch + BOTH Blackwell GPUs + a real bf16 op"
python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda", torch.version.cuda,
      "| avail", torch.cuda.is_available(), "| ndev", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU{i}: {p.name} | {round(p.total_memory/1e9,1)} GB | sm {p.major}.{p.minor}")
for i in range(torch.cuda.device_count()):
    x = torch.randn(4096, 4096, device=f"cuda:{i}", dtype=torch.bfloat16)
    _ = (x @ x).sum().item()
    print(f"  bf16 matmul on cuda:{i} OK")
import vllm; print("vllm", vllm.__version__)
PY
echo "SETUP_DONE"
