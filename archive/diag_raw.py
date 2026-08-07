"""Diagnostic: load ICKG-v4.2 once, print AND save the RAW model output for a
handful of sample articles, so we can see what format Qwen2.5-14B emits (why 65%
don't parse) and iterate the parser offline without re-loading the 14B."""
import json
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ickg_kg.ickg_extractor import ICKG_PROMPT, parse_triplets

df = pd.read_parquet("data/sample_1k.parquet").head(15)
from huggingface_hub import snapshot_download
adapter = snapshot_download("victorlxh/ICKG-v4.2")
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(model="unsloth/Qwen2.5-14B-Instruct", enable_lora=True, max_lora_rank=16,
          tensor_parallel_size=1, dtype="bfloat16", max_model_len=8192,
          gpu_memory_utilization=0.90)
lora = LoRARequest("ickg", 1, adapter)
sp = SamplingParams(temperature=0.0, max_tokens=1024)

convs = [[{"role": "user", "content": ICKG_PROMPT.replace("<input_text>", str(r["body"])[:6000])}]
         for _, r in df.iterrows()]
outs = llm.chat(convs, sp, lora_request=lora, use_tqdm=False)

recs = []
for (_, r), o in zip(df.iterrows(), outs):
    raw = o.outputs[0].text
    recs.append({"ticker": str(r["ticker"]), "raw": raw, "parsed": len(parse_triplets(raw))})
    print("=" * 90)
    print(f"TICKER={r['ticker']}  parsed={len(parse_triplets(raw))}  raw_len={len(raw)}")
    print(raw[:1400])

Path("data/diag_raw.jsonl").write_text("\n".join(json.dumps(x) for x in recs))
print("SAVED data/diag_raw.jsonl")
