"""Blinded-extraction robustness check for the PIT / knowledge-cutoff-leakage question.

Procedure on a fixed sample of articles:
  1. extract UNMASKED -> record the entities the model found per article;
  2. RE-EXTRACT with those entity names (distinct placeholders) + dates masked.
If the extractor is text-grounded rather than injecting entity-knowledge hindsight, the RELATION
structure should survive blinding: similar valid-triple count, similar relation-type mix, and a
stable share of causal "impact" relations (Impact / Pos|Neg_Impact_On / Raise / Decrease) -- those
are the ones most prone to hindsight, since they assert outcomes. A large drop would flag
entity-recognition-driven extraction.

Run on goldbug GPU: `python blinded_check.py`
"""
import sys
import re
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ickg_kg.ickg_extractor import ICKG_PROMPT, parse_triplets

N = 300
IMPACT = {"Impact", "Positive_Impact_On", "Negative_Impact_On", "Raise", "Decrease"}
DATE = re.compile(r"\b(?:19|20)\d{2}\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|"
                  r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}\b", re.I)

df = pd.read_parquet("data/subset_200k.parquet").sample(N, random_state=11).reset_index(drop=True)
from huggingface_hub import snapshot_download
adapter = snapshot_download("victorlxh/ICKG-v4.2")
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(model="unsloth/Qwen2.5-14B-Instruct", enable_lora=True, max_lora_rank=16,
          tensor_parallel_size=1, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.90)
lora = LoRARequest("ickg", 1, adapter); sp = SamplingParams(temperature=0.0, max_tokens=1280)


def extract(bodies):
    convs = [[{"role": "user", "content": ICKG_PROMPT.replace("<input_text>", str(b)[:6000])}] for b in bodies]
    outs = llm.chat(convs, sp, lora_request=lora, use_tqdm=False)
    return [parse_triplets(o.outputs[0].text) for o in outs]


def mask_body(body, ents):
    b = str(body)
    uniq = sorted({e for e in ents if e and len(str(e)) > 2}, key=lambda x: len(str(x)), reverse=True)
    for k, e in enumerate(uniq):
        b = re.sub(re.escape(str(e)), f"[ENT{k % 60}]", b, flags=re.I)
    return DATE.sub("[DATE]", b)


def stats(trips_list):
    valid = [[t for t in ts if t.valid] for ts in trips_list]
    per_art = sum(len(v) for v in valid) / len(valid)
    rels = pd.Series([t.r for v in valid for t in v])
    dist = rels.value_counts(normalize=True)
    impact = dist.reindex(list(IMPACT)).fillna(0).sum()
    return per_art, dist, impact


un = extract(df["body"].tolist())
masked_bodies = [mask_body(df["body"][i], [t.h for t in ts if t.valid] + [t.o for t in ts if t.valid])
                 for i, ts in enumerate(un)]
bl = extract(masked_bodies)

un_pa, un_d, un_imp = stats(un)
bl_pa, bl_d, bl_imp = stats(bl)
allr = un_d.index.union(bl_d.index)
tv = 0.5 * (un_d.reindex(allr).fillna(0) - bl_d.reindex(allr).fillna(0)).abs().sum()

print(f"\n=== blinded extraction ({N} articles) ===")
print(f"valid triples/article : unmasked {un_pa:.1f} | blinded {bl_pa:.1f}  ({100*bl_pa/un_pa:.0f}% retained)")
print(f"causal 'impact' share : unmasked {un_imp:.1%} | blinded {bl_imp:.1%}")
print(f"relation-mix total-variation distance: {tv:.3f}  (0 = identical mix)")
print("\ntop relations (unmasked vs blinded share):")
for r in un_d.head(8).index:
    print(f"  {r:20} {un_d.get(r,0):.3f}  ->  {bl_d.get(r,0):.3f}")
print("\nBLINDED_DONE")
