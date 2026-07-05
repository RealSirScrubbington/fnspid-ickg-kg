"""Select a reproducible FNSPID subset (random offset sampling, date + word
filtered) and write it as a parquet for extract.py. Used for the Stage-1 1k
throughput sample; the Stage-2 200k stratified build is a separate (later) step.
"""
import argparse
import collections
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ickg_kg.config import Config
from ickg_kg.fnspid_sample import FnspidSampler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--year-start", type=int, default=2017)
    ap.add_argument("--year-end", type=int, default=2023)
    ap.add_argument("--min-words", type=int, default=200)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = Config()
    s = FnspidSampler(cfg.fnspid_repo, cfg.news_path)
    print(f"corpus {cfg.news_path}: {s.size/1e9:.1f} GB | columns ok", flush=True)
    arts = s.sample(args.n, args.seed, min_body_chars=200, min_body_words=args.min_words,
                    year_range=(args.year_start, args.year_end))
    df = pd.DataFrame([{"id": a.article_id, "date": a.date, "ticker": a.symbol,
                        "publisher": a.publisher, "title": a.title, "body": a.body,
                        "byte_offset": a.byte_offset} for a in arts])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)

    years = collections.Counter(str(d)[:4] for d in df["date"])
    print(f"\nselected {len(df)} -> {args.out}")
    print("year distribution:", dict(sorted(years.items())))
    print(f"distinct tickers: {df['ticker'].nunique()}")
    wl = df["body"].str.split().str.len()
    print(f"words: mean={int(wl.mean())} median={int(wl.median())} min={int(wl.min())}")


if __name__ == "__main__":
    main()
