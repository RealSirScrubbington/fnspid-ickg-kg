"""Pre-fetch ICKG-v3.2 (LoRA adapter) + its base Mistral-7B-Instruct-v0.2 into
the HF cache. safetensors only (skip the duplicate .bin shards)."""
import os
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
from huggingface_hub import snapshot_download

print("downloading base: mistralai/Mistral-7B-Instruct-v0.2 (safetensors only) ...")
p1 = snapshot_download(
    "mistralai/Mistral-7B-Instruct-v0.2",
    allow_patterns=["*.json", "*.safetensors", "tokenizer.model"],
)
print("base ->", p1)

print("downloading adapter: victorlxh/ICKG-v3.2 ...")
p2 = snapshot_download("victorlxh/ICKG-v3.2")
print("adapter ->", p2)
print("DOWNLOAD DONE")
