"""Stage 0 verification: CUDA/GPU, the 4-bit stack, range-GET sampling, and the
ICKG/FinDKG facts. Prints a checklist; exits non-zero on a hard failure."""
from __future__ import annotations
import sys, traceback

ok = True

print("== torch / CUDA ==")
try:
    import torch
    print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"GPU: {p.name} | VRAM: {p.total_memory/1e9:.2f} GB | cc {p.major}.{p.minor}")
        print("cuda runtime:", torch.version.cuda)
    else:
        ok = False; print("!! CUDA not available")
except Exception:
    ok = False; traceback.print_exc()

print("\n== 4-bit stack ==")
for mod in ("transformers", "accelerate", "peft", "bitsandbytes", "sentencepiece"):
    try:
        m = __import__(mod)
        print(f"{mod}: {getattr(m, '__version__', '?')}")
    except Exception as e:
        ok = False; print(f"!! {mod} import failed: {e}")

print("\n== bitsandbytes 4-bit smoke (tiny nf4 linear on GPU) ==")
try:
    import torch, bitsandbytes as bnb
    lin = bnb.nn.Linear4bit(64, 64, bias=False, compute_dtype=torch.float16,
                            quant_type="nf4").cuda()
    x = torch.randn(2, 64, device="cuda", dtype=torch.float16)
    y = lin(x)
    print("nf4 linear forward OK, out shape", tuple(y.shape))
except Exception:
    ok = False; traceback.print_exc()

print("\n== FNSPID range-GET + record recovery ==")
try:
    from ickg_kg.config import Config
    from ickg_kg.fnspid_sample import FnspidSampler
    cfg = Config()
    s = FnspidSampler(cfg.fnspid_repo, cfg.news_path)
    print(f"size: {s.size/1e9:.1f} GB | columns: {s.header}")
    arts = s.sample(3, seed=123, min_body_chars=200, verbose=False)
    for a in arts:
        print(f"  [{a.date}] {a.symbol}: {a.title[:60]!r}  body={len(a.body)} chars  off={a.byte_offset}")
    if len(arts) < 3:
        ok = False; print("!! sampled fewer than 3 records")
except Exception:
    ok = False; traceback.print_exc()

print("\n== ICKG/FinDKG schema ==")
try:
    from ickg_kg.schema import RELATION2ID, ENTITY_TYPE2ID
    from ickg_kg.ickg_extractor import ICKG_PROMPT
    print(f"relations: {len(RELATION2ID)} | entity types: {len(ENTITY_TYPE2ID)}")
    print("prompt ends with INPUT_TEXT placeholder:", ICKG_PROMPT.rstrip().endswith("<input_text>"))
except Exception:
    ok = False; traceback.print_exc()

print("\n== RESULT:", "ALL OK" if ok else "FAILURES ABOVE", "==")
sys.exit(0 if ok else 1)
