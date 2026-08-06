"""Score EVERY audit pack with the fine-tuned small judge (Qwen2.5-1.5B + LoRA adapter).

Turns the distilled judge from an evaluation artifact into a filter: for each EMERGING
lifeline pack (lenient + tf arms) it records
  label_gen  greedy free-generation label parsed exactly like the teacher's
             (comparable to theme_judge_labels*.csv),
  label_mc   argmax over the three constrained completions '{"label": "<L>"}',
  p_real / p_template / p_incoherent
             softmax over the three completion log-probs (ID-level concat, so token
             alignment is exact).
p_real is the ranking score for the precision-retention curve; the hard labels feed the
benchmark-survival check.

Run (GPU): PYTHONPATH=. python -m dynamics.judge_score --data-dir audit --adapter judge_lora
Output: <data-dir>/judge_scores.csv  (one row per (arm, theme))
"""
import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ARM_FILES = {"lenient": "audit_packs.jsonl", "tf": "audit_packs_tf.jsonl",
             "strict": "audit_packs_strict.jsonl", "nb": "audit_packs_nb.jsonl",
             "tf_r15": "audit_packs_tf_r15.jsonl"}
LABELS = ["REAL", "TEMPLATE", "INCOHERENT"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="audit")
    ap.add_argument("--adapter", default="judge_lora")
    ap.add_argument("--arms", default="lenient,tf", help="comma-separated arm tags")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    arms = [(t, ARM_FILES[t]) for t in a.arms.split(",")]

    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from dynamics.theme_judge import RUBRIC, render
    from dynamics.judge_finetune import MODEL, parse_label

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                                 device_map="cuda")
    model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()

    rows = []
    for arm, packs_f in arms:
        path = Path(a.data_dir) / packs_f
        packs = [json.loads(l) for l in open(path, encoding="utf-8")]
        print(f"[{arm}] {len(packs)} packs", flush=True)
        for i, p in enumerate(packs):
            msgs = [{"role": "system", "content": RUBRIC},
                    {"role": "user", "content": render(p)}]
            prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            pid = tok(prompt, return_tensors="pt").input_ids.to("cuda")
            with torch.no_grad():
                gen = model.generate(pid, max_new_tokens=30, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
                label_gen = parse_label(tok.decode(gen[0, pid.shape[1]:]))
                lps = []
                for lab in LABELS:
                    comp = tok('{"label": "' + lab + '"', add_special_tokens=False,
                               return_tensors="pt").input_ids.to("cuda")
                    full = torch.cat([pid, comp], dim=1)
                    n_new = comp.shape[1]
                    logits = model(full).logits[0, -n_new - 1:-1]
                    lp = F.log_softmax(logits.float(), dim=-1)
                    lps.append(lp[torch.arange(n_new), comp[0]].sum())
                probs = torch.softmax(torch.stack(lps), dim=0).tolist()
            rows.append({"arm": arm, "theme": int(p["theme"]), "label_gen": label_gen,
                         "label_mc": LABELS[int(np.argmax(probs))],
                         "p_real": probs[0], "p_template": probs[1],
                         "p_incoherent": probs[2]})
            if (i + 1) % 25 == 0:
                print(f"  {i + 1} scored", flush=True)

    df = pd.DataFrame(rows)
    out = Path(a.out) if a.out else Path(a.data_dir) / "judge_scores.csv"
    df.to_csv(out, index=False)
    for arm, g in df.groupby("arm"):
        print(f"[{arm}] gen labels {g.label_gen.value_counts().to_dict()} | "
              f"mc labels {g.label_mc.value_counts().to_dict()} | "
              f"mean p_real {g.p_real.mean():.3f}")
    print(f"wrote {out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
