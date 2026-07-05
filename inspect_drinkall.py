"""One-off exploratory script: profile the Drinkall financial-news dataset (field coverage, body lengths, dates, dupes) as a candidate corpus. Not part of the pipeline."""
import json, lzma, urllib.request, statistics, collections
from pathlib import Path

BASE = "https://raw.githubusercontent.com/felixdrinkall/financial-news-dataset/main/data"
YEARS = [2017, 2018, 2019, 2020, 2021, 2022, 2023]
OUT = Path("C:/Users/denma/fnspid-ickg-kg/data/drinkall"); OUT.mkdir(parents=True, exist_ok=True)

def load_year(y):
    p = OUT / f"{y}_processed.json.xz"
    if not p.exists():
        urllib.request.urlretrieve(f"{BASE}/{y}_processed.json.xz", p)
    raw = lzma.open(p).read().decode("utf-8", "replace").strip()
    try:
        d = json.loads(raw)
        if isinstance(d, dict): d = list(d.values())
    except json.JSONDecodeError:
        d = [json.loads(l) for l in raw.splitlines() if l.strip()]
    return d

recs = []
for y in YEARS:
    d = load_year(y); print(f"{y}: {len(d):>6} records"); recs += d
print(f"TOTAL: {len(recs)}")
s = recs[len(recs)//2]
print("FIELDS:", list(s.keys()))
body = lambda r: (r.get("maintext") or r.get("text") or "")
w = [len(body(r).split()) for r in recs]; nz = [x for x in w if x > 0]
print(f"full-text present: {100*len(nz)/len(recs):.1f}%  | words mean={int(statistics.mean(nz))} median={int(statistics.median(nz))} >=200w={100*sum(1 for x in nz if x>=200)/len(recs):.1f}%")
ts = [(r.get("date_publish") or "") for r in recs]
print(f"date_publish present {100*sum(1 for t in ts if t)/len(ts):.1f}% | example {next((t for t in ts if t), None)!r} | with real time-of-day {100*sum(1 for t in ts if t and ' ' in t and t[-8:]!='00:00:00')/len(ts):.1f}%")
src = collections.Counter((r.get("source_domain") or r.get("news_outlet") or "?") for r in recs)
print("sources top5:", src.most_common(5))
tt = [(r.get("title") or "").strip().lower() for r in recs]; tc = collections.Counter(t for t in tt if t)
print(f"titles: {len(tc)} unique / {sum(1 for t in tt if t)} -> exact-dup rate {100*(sum(1 for t in tt if t)-len(tc))/max(1,sum(1 for t in tt if t)):.1f}%")
for k in ["named_entities","mentioned_companies","related_companies","sentiment","emotion","industries"]:
    print(f"  enrich '{k}': populated {100*sum(1 for r in recs if r.get(k))/len(recs):.0f}%")
