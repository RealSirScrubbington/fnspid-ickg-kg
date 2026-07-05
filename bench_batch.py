"""Benchmark batched extraction throughput vs VRAM on this GPU, to pick the
fastest batch size that fits. Loads ICKG once, runs the same K articles at
several batch sizes, reports art/s + peak VRAM + total valid triplets (a
consistency check that batching doesn't corrupt outputs)."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ickg_kg.config import Config, SAMPLE_DIR
from ickg_kg.fnspid_sample import Article
from ickg_kg.ickg_extractor import IckgExtractor


def chunks(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def main(k: int = 8, batch_sizes=(1, 2, 4, 6)) -> None:
    import torch
    cfg = Config(n=500, seed=42)
    recs = json.loads((SAMPLE_DIR / "trial_seed42_n500.json").read_text(encoding="utf-8"))
    arts = [Article(**r) for r in recs[:k]]
    items = [(a.article_id, a.body) for a in arts]

    print(f"loading ICKG (4-bit) ... timing {k} articles per batch size")
    ex = IckgExtractor(cfg)
    print(f"loaded in {ex.load_s:.1f}s | weights VRAM {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # warm up (CUDA/kernels init) so the first timed batch isn't penalised
    ex.extract_batch(items[:2])

    print(f"\n{'batch':>5} {'art/s':>7} {'s/art':>7} {'peakVRAM':>9} {'valid':>6}")
    for b in batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            t0 = time.time()
            total_valid = 0
            for ch in chunks(items, b):
                for res in ex.extract_batch(ch):
                    total_valid += res.n_valid
            dt = time.time() - t0
            peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"{b:>5} {k/dt:>7.3f} {dt/k:>7.1f} {peak:>8.2f}G {total_valid:>6}")
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            msg = str(e).split("\n")[0][:60]
            print(f"{b:>5}   OOM/ERR: {msg}")
            torch.cuda.empty_cache()
            break


if __name__ == "__main__":
    main()
