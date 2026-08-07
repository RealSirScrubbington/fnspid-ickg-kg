"""One-off exploratory script: random-offset sample raw FNSPID rows to measure Article body-length coverage (informed the min-words subset filter). Not part of the pipeline."""
import sys, random, statistics, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ickg_kg.config import Config
from ickg_kg.fnspid_sample import FnspidSampler

cfg = Config()
s = FnspidSampler(cfg.fnspid_repo, cfg.news_path)
ai = s._col.get("Article"); ti = s._col.get("Article_title")
print(f"columns: {s.header}")
print(f"Article idx={ai}  Article_title idx={ti}")
rng = random.Random(2024)
N = 200
got = tries = title_only = 0
b = {"empty":0, "1-49":0, "50-199":0, ">=200":0}
lens = []
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
    body = (row[ai] or "").strip()
    title = (row[ti] or "").strip() if ti is not None and ti < len(row) else ""
    L = len(body); lens.append(L)
    if L == 0: b["empty"] += 1; title_only += 1 if title else 0
    elif L < 50: b["1-49"] += 1
    elif L < 200: b["50-199"] += 1
    else: b[">=200"] += 1
print(f"\nsampled {got} RAW records in {time.time()-t0:.0f}s ({tries} reads)")
print(f"Article length buckets: {b}")
print(f"% with usable body (>=200 chars): {100*b['>=200']/got:.1f}%")
print(f"% empty Article: {100*b['empty']/got:.1f}%  (of which {title_only} have a headline only)")
nz = [x for x in lens if x>0]
if nz:
    print(f"non-empty body chars: mean={int(statistics.mean(nz))} median={int(statistics.median(nz))} max={max(nz)}")
