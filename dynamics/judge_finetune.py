"""LoRA fine-tune of a small judge (Qwen2.5-1.5B-Instruct) on the teacher's pack labels,
with the pre-registered evaluation of dynamics/judge_distill_data.py.

Stages (all in one run):
  1. Zero-shot baseline of the base 1.5B on the control packs (expected to fail the gate,
     as the zero-shot 7B and 14B did).
  2. LoRA SFT on distill_train.jsonl (completion-only loss; r=16, lr 1e-4, 3 epochs, no search).
  3. Evaluation: (a) the 11-pack control gate; (b) 3-way + binary agreement with the teacher
     on the group-disjoint temporal test set; (c) binary agreement with the 24 human labels;
     all with greedy decoding.

Run (GPU host): python -m dynamics.judge_finetune --data-dir audit --out-dir judge_lora
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
CONTROL_NOISE = [2, 21, 25, 34, 58, 66, 301, 315, 349]   # lenient themes -> TEMPLATE expected
CONTROL_REAL = [121, 131, 136]                            # lenient themes -> REAL expected
HUMAN_LENIENT = {36: "TEMPLATE", 42: "REAL", 78: "REAL", 85: "REAL", 146: "TEMPLATE",
                 169: "INCOHERENT", 183: "INCOHERENT", 200: "REAL", 218: "REAL",
                 245: "TEMPLATE", 267: "TEMPLATE", 269: "INCOHERENT", 274: "INCOHERENT",
                 345: "INCOHERENT"}
HUMAN_TF = {58: "REAL", 75: "INCOHERENT", 89: "REAL", 107: "INCOHERENT", 123: "INCOHERENT",
            145: "INCOHERENT", 159: "REAL", 180: "INCOHERENT", 251: "REAL", 313: "INCOHERENT"}


def load_packs(path):
    return {json.loads(l)["theme"]: json.loads(l) for l in open(path, encoding="utf-8")}


def parse_label(txt):
    m = re.search(r'"label"\s*:\s*"(REAL|TEMPLATE|INCOHERENT)"', txt)
    if m:
        return m.group(1)
    m = re.search(r"\b(REAL|TEMPLATE|INCOHERENT)\b", txt)
    return m.group(1) if m else "PARSE_FAIL"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="audit")
    ap.add_argument("--out-dir", default="judge_lora")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-len", type=int, default=3072)
    a = ap.parse_args()
    D = Path(a.data_dir)

    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments)
    from peft import LoraConfig, get_peft_model
    from dynamics.theme_judge import RUBRIC, render

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                                 device_map="cuda")

    def judge(m, packs, themes):
        outs = {}
        m.eval()
        with torch.no_grad():
            for t in themes:
                if t not in packs:
                    continue
                msgs = [{"role": "system", "content": RUBRIC},
                        {"role": "user", "content": render(packs[t])}]
                ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                              return_tensors="pt").to("cuda")
                out = m.generate(ids, max_new_tokens=30, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
                outs[t] = parse_label(tok.decode(out[0, ids.shape[1]:]))
        return outs

    def gate_report(tag, preds):
        ok_n = sum(preds.get(t) == "TEMPLATE" for t in CONTROL_NOISE)
        ok_r = sum(preds.get(t) == "REAL" for t in CONTROL_REAL)
        print(f"[{tag}] CONTROL GATE: noise {ok_n}/{len(CONTROL_NOISE)} -> TEMPLATE, "
              f"real {ok_r}/{len(CONTROL_REAL)} -> REAL | "
              f"{'PASS' if ok_n >= 8 and ok_r == 3 else 'FAIL'}")
        return ok_n, ok_r

    len_packs = load_packs(D / "audit_packs.jsonl")
    tf_packs = load_packs(D / "audit_packs_tf.jsonl")
    ctrl_themes = CONTROL_NOISE + CONTROL_REAL

    print("=== stage 1: zero-shot base model on controls ===")
    gate_report("zero-shot", judge(model, len_packs, ctrl_themes))

    print("=== stage 2: LoRA SFT ===")
    train_rows = [json.loads(l) for l in open(D / "distill_train.jsonl", encoding="utf-8")]
    print(f"{len(train_rows)} training packs")

    def encode(row):
        msgs = row["messages"]
        full = tok.apply_chat_template(msgs, tokenize=False)
        prompt = tok.apply_chat_template(msgs[:2], add_generation_prompt=True, tokenize=False)
        fi = tok(full, truncation=True, max_length=a.max_len)["input_ids"]
        pi = tok(prompt, truncation=True, max_length=a.max_len)["input_ids"]
        labels = [-100] * min(len(pi), len(fi)) + fi[min(len(pi), len(fi)):]
        return {"input_ids": fi, "labels": labels[:len(fi)]}

    ds = [encode(r) for r in train_rows]

    def collate(batch):
        L = max(len(b["input_ids"]) for b in batch)
        pad = tok.pad_token_id or tok.eos_token_id
        return {
            "input_ids": torch.tensor([b["input_ids"] + [pad] * (L - len(b["input_ids"]))
                                       for b in batch]),
            "attention_mask": torch.tensor([[1] * len(b["input_ids"]) +
                                            [0] * (L - len(b["input_ids"])) for b in batch]),
            "labels": torch.tensor([b["labels"] + [-100] * (L - len(b["labels"]))
                                    for b in batch]),
        }

    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    args = TrainingArguments(output_dir=a.out_dir, num_train_epochs=a.epochs,
                             per_device_train_batch_size=2, gradient_accumulation_steps=8,
                             learning_rate=1e-4, bf16=True, logging_steps=20,
                             save_strategy="no", report_to=[], seed=0,
                             lr_scheduler_type="cosine", warmup_ratio=0.05)
    Trainer(model=model, args=args, train_dataset=ds, data_collator=collate).train()
    model.save_pretrained(a.out_dir)

    print("=== stage 3: evaluation ===")
    preds = judge(model, len_packs, ctrl_themes)
    gate_report("fine-tuned", preds)

    test_rows = [json.loads(l) for l in open(D / "distill_test.jsonl", encoding="utf-8")]
    n3 = nb = n = 0
    for r in test_rows:
        msgs = r["messages"][:2]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=30, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        pred = parse_label(tok.decode(out[0, ids.shape[1]:]))
        n += 1
        n3 += pred == r["label"]
        nb += (pred == "REAL") == (r["label"] == "REAL")
    print(f"[fine-tuned] TEACHER AGREEMENT on held-out ({n} packs, group-disjoint, 2022+): "
          f"3-way {n3 / n:.1%} | binary {nb / n:.1%}")

    hums = [(len_packs, HUMAN_LENIENT), (tf_packs, HUMAN_TF)]
    hn = hb = 0
    for packs, labels in hums:
        p = judge(model, packs, list(labels))
        for t, hl in labels.items():
            hn += 1
            hb += (p.get(t) == "REAL") == (hl == "REAL")
    print(f"[fine-tuned] HUMAN AGREEMENT (24 packs, binary): {hb / hn:.1%}")


if __name__ == "__main__":
    main()
