"""Embed core-KG entity names with a HuggingFace model (ChronoBERT encoder or Qwen decoder),
mean-pooling the last hidden states. Saves [N, d] float16 embeddings for the text-based
link-prediction lookahead-bias baseline.

  python -m dynamics.text_embed --model manelalab/chrono-bert-v1-20221231 --out data/dynamics/linkpred/emb_chrono.npy
  python -m dynamics.text_embed --model unsloth/Qwen2.5-14B-Instruct       --out data/dynamics/linkpred/emb_qwen.npy
"""
import argparse
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--kg", default="data/kg_core")
a = ap.parse_args()
DEV = "cuda" if torch.cuda.is_available() else "cpu"

ent = pd.read_csv(f"{a.kg}/entity2id.txt", sep="\t", header=None,
                  names=["name", "id", "type", "tid"], keep_default_na=False).sort_values("id")
names = [str(n) for n in ent["name"].tolist()]
N = len(names)
print(f"embedding {N:,} entity names with {a.model}", flush=True)

tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModel.from_pretrained(a.model, trust_remote_code=True, torch_dtype=torch.bfloat16).to(DEV).eval()

out = []
with torch.no_grad():
    for i in range(0, N, a.batch):
        enc = tok(names[i:i + a.batch], padding=True, truncation=True, max_length=32, return_tensors="pt").to(DEV)
        hs = model(**enc).last_hidden_state                      # [B, T, d]
        mask = enc["attention_mask"].unsqueeze(-1).to(hs.dtype)
        pooled = (hs * mask).sum(1) / mask.sum(1).clamp(min=1)   # mean-pool over real tokens
        out.append(pooled.float().cpu().numpy())
        if (i // a.batch) % 20 == 0:
            print(f"  {i}/{N}", flush=True)

E = np.concatenate(out).astype(np.float16)
np.save(a.out, E)
print(f"saved {a.out} shape {E.shape}")
