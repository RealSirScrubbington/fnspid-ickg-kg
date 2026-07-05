"""extract.py -- batched, checkpointed FinDKG-triplet extraction from FNSPID
articles using ICKG-v4.2 (Qwen2.5-14B-Instruct + LoRA) served by vLLM.

Stage 1 (throughput gate):  python extract.py --input sample_1k.parquet --out tripl_1k.parquet --report
Stage 2 (full build):       python extract.py --input subset_200k.parquet --out tripl_full.parquet
Blinded PIT path:           add --masked  (redacts dates + the ticker before extraction)

Reuses the VERBATIM FinDKG prompt and the parenthesis-aware triplet parser from
ickg_kg (only the chat wrapper differs for Qwen vs Mistral -- vLLM applies the
model's own chat template via .chat()).
"""
from __future__ import annotations
import argparse, csv, json, re, sys, time
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ickg_kg.ickg_extractor import ICKG_PROMPT, parse_triplets
from ickg_kg.schema import ENTITY_TYPES

TRIPLET_COLS = ["id", "date", "ticker", "publisher", "h", "h_type", "r", "o", "o_type", "valid"]

# --- masking for the blinded-vs-unblinded hindsight diagnostic ---------------
_DATE_RE = re.compile(
    r"\b(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{2,4})\b")
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")


def mask_article(body: str, ticker: str = "", names: list[str] | None = None) -> str:
    """Redact dates, 4-digit years, the ticker and any given entity names -> the
    'blinded' input for measuring how much the extractor leans on world knowledge."""
    t = _DATE_RE.sub("[DATE]", body or "")
    t = _YEAR_RE.sub("[YEAR]", t)
    if ticker:
        t = re.sub(rf"\b{re.escape(ticker)}\b", "[ENT]", t)
    for n in names or []:
        if n and len(n) > 2:
            t = re.sub(rf"\b{re.escape(n)}\b", "[ENT]", t, flags=re.IGNORECASE)
    return t


def build_prompt(body: str, max_chars: int) -> str:
    """Insert the (truncated) article body into the verbatim FinDKG/ICKG prompt."""
    return ICKG_PROMPT.replace("<input_text>", (body or "")[:max_chars])


def load_subset(path: str) -> pd.DataFrame:
    """Read the article subset (parquet/json/jsonl); requires id/date/body, defaults ticker/publisher."""
    p = Path(path)
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_json(p, lines=p.suffix == ".jsonl")
    needed = {"id", "date", "body"}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"subset missing columns: {missing}; has {list(df.columns)}")
    for c in ("ticker", "publisher"):
        if c not in df.columns:
            df[c] = ""
    return df


def done_ids(stats_path: Path) -> set[str]:
    """Article ids already checkpointed in the stats file (lets a rerun resume, not redo)."""
    if not stats_path.exists():
        return set()
    return {json.loads(l)["id"] for l in stats_path.read_text().splitlines() if l.strip()}


def main() -> None:
    """Batched vLLM extraction with per-chunk checkpointing: skips articles already recorded
    in .stats.jsonl, then appends the triplet CSV and stats after every chunk, so a killed
    run loses at most one chunk."""
    ap = argparse.ArgumentParser(description="vLLM ICKG extraction (checkpointed)")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True, help="triplets parquet (checkpointed alongside .stats.jsonl)")
    ap.add_argument("--base", default="unsloth/Qwen2.5-14B-Instruct")
    ap.add_argument("--adapter", default="victorlxh/ICKG-v4.2")
    ap.add_argument("--tp", type=int, default=2, help="tensor_parallel_size (#GPUs)")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--max-input-chars", type=int, default=6000)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--chunk", type=int, default=2000, help="checkpoint chunk size")
    ap.add_argument("--limit", type=int, default=0, help="0=all; else first N (Stage 1)")
    ap.add_argument("--masked", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shard", default="0/1", help="i/N: process rows i::N (data-parallel split)")
    args = ap.parse_args()

    df = load_subset(args.input)
    si, sn = (int(x) for x in args.shard.split("/"))
    if sn > 1:
        df = df.iloc[si::sn].reset_index(drop=True)  # interleaved -> even temporal coverage per shard
    if args.limit:
        df = df.head(args.limit)
    out_path = Path(args.out)
    stats_path = out_path.with_suffix(".stats.jsonl")
    done = done_ids(stats_path)
    todo = df[~df["id"].astype(str).isin(done)].to_dict("records")
    print(f"[extract] input={len(df)} done={len(done)} todo={len(todo)} masked={args.masked}", flush=True)

    if todo:
        from huggingface_hub import snapshot_download
        adapter_path = snapshot_download(args.adapter)
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
        print(f"[extract] loading vLLM: base={args.base} adapter={args.adapter} tp={args.tp} dtype={args.dtype}", flush=True)
        llm = LLM(model=args.base, enable_lora=True, max_lora_rank=16,
                  tensor_parallel_size=args.tp, dtype=args.dtype,
                  max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_mem_util,
                  seed=args.seed)
        lora = LoRARequest("ickg-v4.2", 1, adapter_path)
        sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)

        t0 = time.time(); seen = 0
        for i in range(0, len(todo), args.chunk):
            chunk = todo[i:i + args.chunk]
            convs = []
            for r in chunk:
                body = mask_article(r["body"], str(r.get("ticker", ""))) if args.masked else r["body"]
                convs.append([{"role": "user", "content": build_prompt(body, args.max_input_chars)}])
            outs = llm.chat(convs, sp, lora_request=lora, use_tqdm=False)
            trip_rows, stat_rows = [], []
            for r, o in zip(chunk, outs):
                raw = o.outputs[0].text
                trips = parse_triplets(raw)
                for t in trips:
                    trip_rows.append({"id": str(r["id"]), "date": r["date"],
                                      "ticker": r.get("ticker", ""), "publisher": r.get("publisher", ""),
                                      "h": t.h, "h_type": t.h_type, "r": t.r,
                                      "o": t.o, "o_type": t.o_type, "valid": t.valid})
                nv = sum(t.valid for t in trips)
                stat_rows.append({"id": str(r["id"]), "date": str(r["date"]),
                                  "n_parsed": len(trips), "n_valid": nv,
                                  "malformed": len(trips) == 0, "raw_len": len(raw),
                                  "raw": raw})  # keep raw so triplets can be re-parsed offline
            _append_csv(out_path, trip_rows)
            with stats_path.open("a") as f:
                for s in stat_rows:
                    f.write(json.dumps(s) + "\n")
            seen += len(chunk)
            el = time.time() - t0
            print(f"  {seen}/{len(todo)}  {seen/el:.2f} art/s  (chunk valid={sum(s['n_valid'] for s in stat_rows)})", flush=True)
        print(f"[extract] done {len(todo)} articles in {(time.time()-t0)/60:.1f} min", flush=True)

    if args.report:
        report(out_path, stats_path, args)


def _append_csv(path: Path, rows: list[dict]) -> None:
    """True append (O(chunk)) -- avoids the O(n^2) read+rewrite of parquet over a
    large checkpointed build."""
    if not rows:
        return
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=TRIPLET_COLS)
        if write_header:
            w.writeheader()
        w.writerows(rows)


def report(out_path: Path, stats_path: Path, args) -> None:
    """Print the Stage-1 quality report (success/malformed rates, type/relation dists) from
    the triplet CSV + stats file."""
    stats = [json.loads(l) for l in stats_path.read_text().splitlines() if l.strip()]
    df = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame(columns=TRIPLET_COLS)
    valid = df[df["valid"].astype(str).str.lower() == "true"]
    n = len(stats)
    with_t = sum(1 for s in stats if s["n_valid"] > 0)
    total_valid = sum(s["n_valid"] for s in stats)
    malformed = sum(1 for s in stats if s["malformed"])
    ent = Counter(list(valid["h_type"]) + list(valid["o_type"]))
    rel = Counter(valid["r"])
    gpe = valid[(valid["h_type"] == "GPE") | (valid["o_type"] == "GPE")]
    print("\n" + "=" * 64)
    print(f"STAGE-1 REPORT  n={n}  model={args.adapter}  tp={args.tp}  dtype={args.dtype}")
    print("=" * 64)
    print(f"success rate (>=1 valid triplet): {100*with_t/n:.1f}%")
    print(f"malformed-output rate           : {100*malformed/n:.1f}%")
    print(f"mean valid triplets/article     : {total_valid/n:.2f}")
    print(f"GPE edge density                : {100*len(gpe)/max(1,total_valid):.2f}%  ({len(gpe)} edges)")
    print(f"entity-type dist : {dict(ent.most_common())}")
    print(f"missing types    : {[t for t in ENTITY_TYPES if t not in ent]}")
    print(f"relation dist    : {dict(rel.most_common())}")
    print("15 sample triplets:")
    for _, r in valid.head(15).iterrows():
        print(f"   ({r['h']} | {r['h_type']} | {r['r']} | {r['o']} | {r['o_type']})")


if __name__ == "__main__":
    main()
