"""One-off exploratory script: random-offset sample raw FNSPID rows to measure the Article word-count distribution (sized the >=200-word cut). Not part of the pipeline."""
import sys, random, statistics, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ickg_kg.config import Config
from ickg_kg.fnspid_sample import FnspidSampler

cfg = Config()
s = FnspidSampler(cfg.fnspid_repo, cfg.news_path)
ai = s._col.get("Article")
rng = random.Random(7)
N = 200
got = tries = 0
buckets = {"<50":0, "50-99":0, "100-199":0, "200-499":0, "500-999":0, "1000+":0}
words = []
t0 = time.time()
while got < N and tries < N*30:
    tries += 1
    off = rng.randint(s._header_end, s.size - 262144)
    try:
        row = s._record_at(off)
    except Exception:
        continue
    if row is None or len(row) <= ai:
        continue
    got += 1
    w = len((row[ai] or "").split()); words.append(w)
    if w < 50: buckets["<50"] += 1
    elif w < 100: buckets["50-99"] += 1
    elif w < 200: buckets["100-199"] += 1
    elif w < 500: buckets["200-499"] += 1
    elif w < 1000: buckets["500-999"] += 1
    else: buckets["1000+"] += 1
ge200 = sum(1 for w in words if w >= 200)
print(f"sampled {got} raw records ({tries} reads, {time.time()-t0:.0f}s)")
print(f"word-count buckets: {buckets}")
print(f"% >= 200 words: {100*ge200/got:.1f}%   (=> ~{ge200/got*15.7:.1f}M of 15.7M would pass)")
print(f"words: mean={int(statistics.mean(words))} median={int(statistics.median(words))} p10={sorted(words)[got//10]} max={max(words)}")
