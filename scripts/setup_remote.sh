#!/usr/bin/env bash
# One-shot environment setup on sideswipe (/scratch0). Logs progress; idempotent-ish.
set -e
PROJ=/scratch0/dmaruev/fnspid-ickg-kg
export HF_HOME=/scratch0/dmaruev/hf-cache
cd "$PROJ"

echo "[1/5] creating venv (python3.9)"
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip -q

echo "[2/5] installing torch (cu124)"
pip install -q torch --index-url https://download.pytorch.org/whl/cu124

echo "[3/5] installing ML stack"
pip install -q -r requirements.txt

echo "[4/5] verifying torch + CUDA"
python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda_avail", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("GPU:", p.name, "| VRAM_GB", round(p.total_memory/1e9, 2), "| cc", f"{p.major}.{p.minor}")
import bitsandbytes, transformers, peft
print("transformers", transformers.__version__, "| peft", peft.__version__, "| bnb", bitsandbytes.__version__)
PY

echo "[5/5] downloading base model + ICKG adapter (~14 GB) into $HF_HOME"
python download_model.py

echo "ALL_DONE"
