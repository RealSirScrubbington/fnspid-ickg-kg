"""Template-article fingerprinting, v2.

Slot-normalise each article's title and body lede (tickers, numbers, dates -> placeholders);
a normalised skeleton recurring >= MIN_RECUR times is a template fingerprint; an article
matching a template fingerprint (title or lede) is a template article; an entity with
>= ENT_SHARE of its triple mentions (valid rows) in template articles is template-dominated.

STATUS: the original (v1) one-off script that produced the canonical template_entities.txt
was never committed; reconstruction was attempted against the committed run-log anchors
(987/826 fps, 159,781 articles = 26.4%, 2,010 entities) and did NOT converge (best: 678/508
fps, 151,930 = 25.1%, 4,310 entities, 38% list overlap) - the original's exact normalisation
and counting basis are unrecoverable. This v2 is therefore a documented re-implementation:
- the v1 entity list stays CANONICAL for every frozen configuration (themes.py reads it);
- v2's ARTICLE flags (template_articles.txt) are used only by the template-filtered
  detector arm introduced after the precision audit.
Known coverage property: skeletons whose variable slot is a ticker symbol are caught ~100%
(e.g. "Interesting <TICK> Put And Call Options For <date>"); skeletons embedding the full
company NAME are under-caught (~7% for "Options Now Available For <Company>"), so the
filtered arm's measured gains are conservative. Entity output goes to
template_entities_RECON.txt for comparison only - never overwrite the canonical list.
Run: .venv/Scripts/python -m dynamics.template_flags
"""
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
OUT = Path("data/dynamics/themes")
PARQUET = "data/subset_600k.parquet"
TRIPL = ["data/tripl_600k_0.csv", "data/tripl_600k_1.csv"]
MIN_RECUR = 25
ENT_SHARE = 0.60
ENT_MIN = 50
LEDE = 120

MONTHS = r"january|february|march|april|may|june|july|august|september|october|november|december"
RE_TICK = re.compile(r"\b[A-Z]{1,5}\b")
RE_MON = re.compile(rf"\b(?:{MONTHS})\b", re.I)
RE_NUM = re.compile(r"\d+(?:[.,]\d+)*(?:st|nd|rd|th)?")
RE_PUNCT = re.compile(r"[^\w@ ]+")
RE_WS = re.compile(r"\s+")


def norm(s):
    """Slot-normalise: tickers -> @T, month names -> @M, numbers/ordinals -> @N."""
    s = RE_TICK.sub("@T", str(s))
    s = RE_MON.sub("@M", s)
    s = RE_NUM.sub("@N", s)
    s = RE_PUNCT.sub(" ", s)
    return RE_WS.sub(" ", s.lower()).strip()


def main():
    t0 = time.time()
    df = pd.read_parquet(PARQUET, columns=["id", "title", "body"])
    df["nt"] = df["title"].map(norm)
    df["nb"] = df["body"].str[:LEDE].map(norm)
    print(f"tokenized {len(df):,} in {time.time()-t0:.0f}s")

    flagged = pd.Series(False, index=df.index)
    for label, col in (("pass", "nt"), ("pass", "nb")):
        c = Counter(df[col])
        fps = {k for k, n in c.items() if n >= MIN_RECUR}
        new = df[col].isin(fps) & ~flagged
        flagged |= df[col].isin(fps)
        print(f"{label}: {len(fps)} template fps, union now {int(flagged.sum()):,} (+{int(new.sum()):,})")
    print(f"UNION templated articles: {int(flagged.sum()):,} ({flagged.mean():.1%})")
    tset = set(df.loc[flagged, "id"])
    (OUT / "template_articles.txt").write_text("\n".join(sorted(tset)), encoding="utf-8")

    tot, tmp = defaultdict(int), defaultdict(int)
    for path in TRIPL:
        for chunk in pd.read_csv(path, usecols=["id", "h", "o", "valid"], chunksize=2_000_000):
            chunk = chunk[chunk["valid"].astype(str) == "True"]
            is_t = chunk["id"].isin(tset)
            for col in ("h", "o"):
                for nm, it in zip(chunk[col], is_t):
                    tot[nm] += 1
                    if it:
                        tmp[nm] += 1
        print(f"  scanned {path}")
    rows = [(nm, tot[nm], tmp[nm] / tot[nm]) for nm in tot if tot[nm] >= ENT_MIN
            and tmp[nm] / tot[nm] >= ENT_SHARE]
    rows.sort(key=lambda r: -r[1])
    print(f"template-dominated entities: {len(rows):,}")
    with open(OUT / "template_entities_RECON.txt", "w", encoding="utf-8") as f:
        for nm, n, sh in rows:
            f.write(f"{nm}\t{n}\t{sh:.2f}\n")
    print("wrote template_articles.txt + template_entities_RECON.txt "
          "(compare against committed template_entities.txt before use)")


if __name__ == "__main__":
    main()
