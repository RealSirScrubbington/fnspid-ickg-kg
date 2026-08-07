"""Smoke test: load ICKG once and dump RAW output + parsed triplets for the first
few cached trial articles. Confirms the verbatim prompt + Mistral [INST] framing
+ parser actually work, and measures per-article latency to project trial time."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ickg_kg.config import Config, SAMPLE_DIR
from ickg_kg.fnspid_sample import Article
from ickg_kg.ickg_extractor import IckgExtractor


def main(k: int = 5) -> None:
    cfg = Config(n=500, seed=42)
    recs = json.loads((SAMPLE_DIR / "trial_seed42_n500.json").read_text(encoding="utf-8"))
    arts = [Article(**r) for r in recs[:k]]

    print("loading ICKG (4-bit) ...")
    ex = IckgExtractor(cfg)
    import torch
    print(f"loaded in {ex.load_s:.1f}s on {ex.device} | "
          f"VRAM alloc {torch.cuda.memory_allocated()/1e9:.2f} GB "
          f"reserved {torch.cuda.memory_reserved()/1e9:.2f} GB")

    lat = []
    for a in arts:
        res = ex.extract(a.article_id, a.body)
        lat.append(res.latency_s)
        print("=" * 84)
        print(f"{a.date}  {a.symbol}  | {a.title[:72]}")
        print(f"latency={res.latency_s:.1f}s  parsed={res.n_parsed}  valid={res.n_valid}  "
              f"malformed={res.malformed}")
        print("RAW (first 700 chars):")
        print("  " + res.raw[:700].replace("\n", "\n  "))
        print("PARSED TRIPLETS:")
        for t in res.triplets[:14]:
            flag = "" if t.valid else "  <-- INVALID type/relation"
            print(f"   ({t.h} | {t.h_type} | {t.r} | {t.o} | {t.o_type}){flag}")
    print("=" * 84)
    print(f"latency: mean={sum(lat)/len(lat):.1f}s  min={min(lat):.1f}s  max={max(lat):.1f}s")
    print(f"=> projected 500-article trial: ~{500*sum(lat)/len(lat)/60:.0f} min")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
