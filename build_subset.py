"""Stream the full FNSPID news CSV ONCE and build a reproducible, time-stratified
subset: even per-month coverage over [year_start, year_end], body >= min_words,
via per-month reservoir sampling (fixed seed). Saves the subset parquet + the
exact selected row-index list, and reports the per-month distribution vs what was
available (so recency-skew is visible and documented).
"""
import argparse, collections, csv, io, sys, time, random
from pathlib import Path
import requests
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ickg_kg.config import Config

csv.field_size_limit(10**8)  # some FNSPID article bodies exceed the 128 KB csv default


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-month", type=int, default=2400, help="reservoir size per YYYY-MM")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--year-start", type=int, default=2017)
    ap.add_argument("--year-end", type=int, default=2023)
    ap.add_argument("--min-words", type=int, default=200)
    ap.add_argument("--out", default="data/subset_200k.parquet")
    ap.add_argument("--source-file", default=None, help="read a local CSV instead of streaming the URL")
    args = ap.parse_args()

    cfg = Config()
    rng = random.Random(args.seed)
    if args.source_file:
        print(f"reading local file {args.source_file}", flush=True)
        reader = csv.reader(open(args.source_file, encoding="utf-8", errors="replace"))
    else:
        url = f"https://huggingface.co/datasets/{cfg.fnspid_repo}/resolve/main/{cfg.news_path}"
        sess = requests.Session(); sess.headers.update({"User-Agent": "ickg/0.1"})
        print(f"streaming {url}", flush=True)
        r = sess.get(url, stream=True, timeout=180); r.raise_for_status(); r.raw.decode_content = True
        reader = csv.reader(io.TextIOWrapper(r.raw, encoding="utf-8", errors="replace"))
    header = next(reader)
    col = {c: i for i, c in enumerate(header)}
    iId, iDate, iArt = 0, col["Date"], col["Article"]
    iSym, iPub, iTitle = col["Stock_symbol"], col["Publisher"], col["Article_title"]

    res: dict[str, list] = {}
    seen = collections.Counter()
    t0 = time.time(); nrows = 0
    it = iter(reader)
    while True:
        try:
            row = next(it)
        except StopIteration:
            break
        except Exception as ex:  # CDN closed the stream near EOF -- keep what we sampled
            print(f"[warn] stream ended after {nrows} rows: {type(ex).__name__}: {ex}", flush=True)
            break
        nrows += 1
        if nrows % 1_000_000 == 0:
            kept = sum(len(v) for v in res.values())
            print(f"  {nrows//10**6}M rows | {time.time()-t0:.0f}s | kept={kept}", flush=True)
        if len(row) <= iArt:
            continue
        date = row[iDate]
        if len(date) < 7 or not date[:4].isdigit():
            continue
        y = int(date[:4])
        if not (args.year_start <= y <= args.year_end):
            continue
        if len(row[iArt].split()) < args.min_words:
            continue
        m = date[:7]
        seen[m] += 1
        rec = {"fnspid_idx": row[iId], "date": date, "ticker": row[iSym],
               "publisher": row[iPub], "title": row[iTitle], "body": row[iArt]}
        b = res.setdefault(m, [])
        if len(b) < args.per_month:
            b.append(rec)
        else:
            j = rng.randint(0, seen[m] - 1)
            if j < args.per_month:
                b[j] = rec

    recs = [x for m in sorted(res) for x in res[m]]
    df = pd.DataFrame(recs)
    df.insert(0, "id", [f"a{i:07d}" for i in range(len(df))])   # unique processing id
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    out.with_suffix(".ids.txt").write_text("\n".join(str(x["fnspid_idx"]) for x in recs))

    print(f"\nTOTAL selected: {len(df)} from {nrows} rows in {(time.time()-t0)/60:.1f} min "
          f"(seed={args.seed}, per_month={args.per_month})", flush=True)
    print("per-month  selected/available:")
    for m in sorted(seen):
        print(f"  {m}: {len(res.get(m, [])):>4} / {seen[m]}")
    print("SUBSET_DONE")


if __name__ == "__main__":
    main()
