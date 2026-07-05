"""Pre-sample the Stage-1 trial articles (model not required) and report the
draw's temporal / symbol spread, so the random subset can be sanity-checked
independently of extraction. Writes the exact cache file trial.py reuses."""
import collections
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ickg_kg.config import Config, SAMPLE_DIR, ensure_dirs
from ickg_kg.fnspid_sample import FnspidSampler, articles_to_records


def main(n: int = 500, seed: int = 42) -> None:
    ensure_dirs()
    cfg = Config(n=n, seed=seed)
    s = FnspidSampler(cfg.fnspid_repo, cfg.news_path)
    print(f"corpus {cfg.news_path}: {s.size/1e9:.1f} GB | columns ok")
    arts = s.sample(cfg.n, cfg.seed, cfg.min_body_chars,
                    min_body_words=cfg.min_body_words)
    cache = SAMPLE_DIR / f"trial_seed{cfg.seed}_n{cfg.n}.json"
    cache.write_text(json.dumps(articles_to_records(arts)), encoding="utf-8")

    years = collections.Counter(a.date[:4] for a in arts)
    syms = collections.Counter(a.symbol for a in arts)
    print(f"\nsampled {len(arts)} articles -> {cache}")
    print("year distribution:", dict(sorted(years.items())))
    print(f"distinct symbols: {len(syms)} | top: {syms.most_common(8)}")
    print(f"body chars: mean={int(statistics.mean(len(a.body) for a in arts))} "
          f"min={min(len(a.body) for a in arts)} max={max(len(a.body) for a in arts)}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    main(n, seed)
