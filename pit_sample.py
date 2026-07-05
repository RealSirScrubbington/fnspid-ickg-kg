"""Stage 4 blinded-vs-unblinded PIT hook. Runs ICKG on a small sample with the
article dates + the ticker MASKED vs UNMASKED, saves both, and measures how much
the extracted graph changes -- a proxy for how much the extractor leans on
recognisable entities / world knowledge rather than the text in front of it.
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ickg_kg.ickg_extractor import ICKG_PROMPT, parse_triplets
from extract import mask_article

N = 100
df = pd.read_parquet("data/subset_200k.parquet").sample(N, random_state=7).reset_index(drop=True)
from huggingface_hub import snapshot_download
ap = snapshot_download("victorlxh/ICKG-v4.2")
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(model="unsloth/Qwen2.5-14B-Instruct", enable_lora=True, max_lora_rank=16,
          tensor_parallel_size=1, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.90)
lora = LoRARequest("ickg", 1, ap); sp = SamplingParams(temperature=0.0, max_tokens=1280)


def gen(masked: bool):
    convs = []
    for _, r in df.iterrows():
        b = mask_article(str(r["body"]), str(r["ticker"])) if masked else str(r["body"])
        convs.append([{"role": "user", "content": ICKG_PROMPT.replace("<input_text>", b[:6000])}])
    return [parse_triplets(o.outputs[0].text) for o in llm.chat(convs, sp, lora_request=lora, use_tqdm=False)]


def edges(ts):
    return {(t.h.lower(), t.r, t.o.lower()) for t in ts if t.valid}


un = gen(False); ma = gen(True)
rows = []; ov_tot = 0; un_tot = 0
for i, (_, r) in enumerate(df.iterrows()):
    ue, me = edges(un[i]), edges(ma[i])
    ov = len(ue & me); ov_tot += ov; un_tot += len(ue)
    rows.append({"id": r["id"], "ticker": r["ticker"], "n_unmasked": len(ue),
                 "n_masked": len(me), "overlap": ov})
pd.DataFrame(rows).to_csv("data/pit_sample_compare.csv", index=False)
print(f"[pit] unmasked mean valid edges/art: {un_tot/N:.1f}")
print(f"[pit] masked   mean valid edges/art: {sum(r['n_masked'] for r in rows)/N:.1f}")
print(f"[pit] edge overlap (masked keeps an unmasked edge): {100*ov_tot/max(1,un_tot):.1f}%")
print("PIT_SAMPLE_DONE")
