"""Stage 1 -- TRIAL go/no-go gate.

Runs ICKG-v3.2 over ~500 randomly-sampled FNSPID articles and reports the
numbers needed to decide whether to scale to the bounded build:

  * extraction success rate (% articles with >=1 parseable triplet)
  * mean triplets / article
  * entity-type and relation-type distributions
  * GPE (country) edge density   [the brief's weakest relation]
  * throughput (articles/sec) + projected full-build time
  * malformed-output rate
  * 15 sample triplets to eyeball

It is checkpointed/resumable: triplets stream to CSV and per-article stats to a
JSONL as they complete, so a restart (e.g. after VRAM OOM) continues where it
left off. Then it STOPS -- scaling is a separate, gated decision.

Usage:
  python trial.py --n 500 --seed 42 --precision 4bit
  python trial.py --n 8  --seed 1   --smoke           # quick end-to-end check
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ickg_kg.config import Config, add_common_args, ensure_dirs, SAMPLE_DIR, TRIPLET_DIR, REPORT_DIR
from ickg_kg.fnspid_sample import FnspidSampler, Article, articles_to_records
from ickg_kg.schema import ENTITY_TYPES

TRIPLET_COLS = ["article_id", "date", "symbol", "byte_offset",
                "h", "h_type", "r", "o", "o_type", "valid"]


def load_or_sample(cfg: Config, smoke: bool) -> list[Article]:
    tag = "smoke" if smoke else "trial"
    cache = SAMPLE_DIR / f"{tag}_seed{cfg.seed}_n{cfg.n}.json"
    if cache.exists():
        recs = json.loads(cache.read_text(encoding="utf-8"))
        print(f"reusing cached sample: {cache} ({len(recs)} articles)")
        return [Article(**r) for r in recs]
    print(f"sampling {cfg.n} random FNSPID articles (seed={cfg.seed}) ...")
    sampler = FnspidSampler(cfg.fnspid_repo, cfg.news_path)
    print(f"  corpus: {cfg.news_path}  size={sampler.size/1e9:.1f} GB")
    print(f"  REAL columns: {sampler.header}")
    arts = sampler.sample(cfg.n, cfg.seed, cfg.min_body_chars,
                          min_body_words=cfg.min_body_words)
    cache.write_text(json.dumps(articles_to_records(arts)), encoding="utf-8")
    print(f"  saved sample -> {cache}")
    return arts


def done_ids(stats_path: Path) -> set[str]:
    if not stats_path.exists():
        return set()
    ids = set()
    for line in stats_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            ids.add(json.loads(line)["article_id"])
    return ids


def run(cfg: Config, smoke: bool) -> None:
    ensure_dirs()
    tag = "smoke" if smoke else "trial"
    triplet_csv = TRIPLET_DIR / f"{tag}_triplets_seed{cfg.seed}_n{cfg.n}.csv"
    stats_jsonl = TRIPLET_DIR / f"{tag}_stats_seed{cfg.seed}_n{cfg.n}.jsonl"

    articles = load_or_sample(cfg, smoke)
    already = done_ids(stats_jsonl)
    if already:
        print(f"resuming: {len(already)} articles already done")

    if not triplet_csv.exists():
        with triplet_csv.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=TRIPLET_COLS).writeheader()

    from ickg_kg.ickg_extractor import IckgExtractor
    print(f"loading ICKG ({cfg.precision}) base={cfg.base_model} ...")
    extractor = IckgExtractor(cfg)
    print(f"  model loaded in {extractor.load_s:.1f}s on {extractor.device}")
    _report_vram(extractor)

    todo = [a for a in articles if a.article_id not in already]
    bs = max(1, cfg.batch_size)
    print(f"extracting over {len(todo)} articles (batch_size={bs}) ...")
    t0 = time.time()
    done = 0
    for batch in _chunks(todo, bs):
        results = extractor.extract_batch([(a.article_id, a.body) for a in batch])
        with triplet_csv.open("a", newline="", encoding="utf-8") as f, \
                stats_jsonl.open("a", encoding="utf-8") as sf:
            w = csv.DictWriter(f, fieldnames=TRIPLET_COLS)
            for art, res in zip(batch, results):
                for t in res.triplets:
                    w.writerow({"article_id": art.article_id, "date": art.date,
                                "symbol": art.symbol, "byte_offset": art.byte_offset,
                                **t.as_row()})
                sf.write(json.dumps({"article_id": art.article_id, "date": art.date,
                                     "n_parsed": res.n_parsed, "n_valid": res.n_valid,
                                     "malformed": res.malformed, "latency_s": res.latency_s,
                                     "raw_len": len(res.raw)}) + "\n")
        done += len(batch)
        el = time.time() - t0
        print(f"  {done}/{len(todo)}  {done/el:.2f} art/s  "
              f"last batch valid={sum(r.n_valid for r in results)}")
    report(cfg, tag, stats_jsonl, triplet_csv, articles)


def _chunks(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def _report_vram(extractor) -> None:
    try:
        torch = extractor._torch
        if torch.cuda.is_available():
            used = torch.cuda.memory_allocated() / 1e9
            res = torch.cuda.memory_reserved() / 1e9
            print(f"  VRAM allocated={used:.2f} GB reserved={res:.2f} GB")
    except Exception:
        pass


def report(cfg, tag, stats_jsonl, triplet_csv, articles) -> None:
    stats = []
    for l in stats_jsonl.read_text(encoding="utf-8").splitlines():
        if l.strip():
            try:
                stats.append(json.loads(l))
            except json.JSONDecodeError:
                pass  # tolerate a partial trailing line when read mid-append
    rows = [r for r in csv.DictReader(triplet_csv.open(encoding="utf-8"))
            if r.get("valid") in ("True", "False")]
    valid_rows = [r for r in rows if r["valid"] == "True"]

    n = len(stats)
    with_triplets = sum(1 for s in stats if s["n_valid"] > 0)
    malformed = sum(1 for s in stats if s["malformed"])
    total_valid = sum(s["n_valid"] for s in stats)
    total_parsed = sum(s["n_parsed"] for s in stats)
    extract_time = sum(s["latency_s"] for s in stats)
    thr = n / extract_time if extract_time else 0.0

    ent_types = Counter()
    for r in valid_rows:
        ent_types[r["h_type"]] += 1
        ent_types[r["o_type"]] += 1
    rel_types = Counter(r["r"] for r in valid_rows)
    gpe_rows = [r for r in valid_rows if "GPE" in (r["h_type"], r["o_type"])]
    gpe_entities = set()
    for r in gpe_rows:
        if r["h_type"] == "GPE":
            gpe_entities.add(r["h"].lower())
        if r["o_type"] == "GPE":
            gpe_entities.add(r["o"].lower())

    dates = sorted(s["date"] for s in stats if s.get("date"))
    proj = {k: _fmt_eta(k / thr) for k in (10_000, 30_000, 50_000)} if thr else {}

    R = {
        "tag": tag, "n_articles": n, "seed": cfg.seed, "precision": cfg.precision,
        "max_input_tokens": cfg.max_input_tokens, "max_new_tokens": cfg.max_new_tokens,
        "date_range": [dates[0], dates[-1]] if dates else None,
        "success_rate": round(with_triplets / n, 4) if n else 0,
        "malformed_rate": round(malformed / n, 4) if n else 0,
        "mean_valid_triplets_per_article": round(total_valid / n, 3) if n else 0,
        "mean_parsed_triplets_per_article": round(total_parsed / n, 3) if n else 0,
        "valid_triplet_fraction": round(total_valid / total_parsed, 4) if total_parsed else 0,
        "total_valid_triplets": total_valid,
        "throughput_art_per_s": round(thr, 3),
        "projected_build_time": proj,
        "gpe_edge_count": len(gpe_rows),
        "gpe_edge_density": round(len(gpe_rows) / total_valid, 4) if total_valid else 0,
        "distinct_gpe_entities": len(gpe_entities),
        "entity_type_distribution": dict(ent_types.most_common()),
        "relation_type_distribution": dict(rel_types.most_common()),
        "missing_entity_types": [t for t in ENTITY_TYPES if t not in ent_types],
    }
    sample_trips = [f"({r['h']}, {r['h_type']}, {r['r']}, {r['o']}, {r['o_type']})"
                    for r in valid_rows[:15]]

    out_json = REPORT_DIR / f"{tag}_report_seed{cfg.seed}_n{cfg.n}.json"
    out_json.write_text(json.dumps({**R, "sample_triplets": sample_trips}, indent=2),
                        encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"STAGE 1 {tag.upper()} REPORT  (n={n}, seed={cfg.seed}, {cfg.precision})")
    print("=" * 70)
    print(f"date range sampled     : {R['date_range']}")
    print(f"extraction success rate: {R['success_rate']*100:.1f}%  (>=1 valid triplet)")
    print(f"malformed-output rate  : {R['malformed_rate']*100:.1f}%  (0 parseable tuples)")
    print(f"mean valid triplets/art: {R['mean_valid_triplets_per_article']}  "
          f"(parsed {R['mean_parsed_triplets_per_article']})")
    print(f"valid-tuple fraction   : {R['valid_triplet_fraction']*100:.1f}%")
    print(f"throughput             : {R['throughput_art_per_s']} art/s")
    print(f"projected full build   : {R['projected_build_time']}")
    print(f"GPE edge density       : {R['gpe_edge_density']*100:.2f}%  "
          f"({R['gpe_edge_count']} edges, {R['distinct_gpe_entities']} distinct GPEs)")
    print(f"\nentity-type distribution:\n  {R['entity_type_distribution']}")
    print(f"missing entity types   : {R['missing_entity_types']}")
    print(f"\nrelation-type distribution:\n  {R['relation_type_distribution']}")
    print("\n15 sample triplets:")
    for s in sample_trips:
        print("  " + s)
    print(f"\nfull report -> {out_json}")
    print(f"triplets     -> {triplet_csv}")
    _verdict(R)


def _verdict(R) -> None:
    ok = (R["success_rate"] >= 0.70 and R["mean_valid_triplets_per_article"] >= 3.0
          and R["gpe_edge_count"] > 0)
    print("\n" + "-" * 70)
    print(f"GATE: success>=70%? {R['success_rate']*100:.1f}%   "
          f"triplets/art>=3? {R['mean_valid_triplets_per_article']}   "
          f"GPE present? {R['gpe_edge_count']>0}")
    print(f"=> {'LOOKS SANE -- candidate GO (await go-ahead)' if ok else 'BELOW TARGET -- inspect prompt/parse before scaling'}")
    print("-" * 70)


def _fmt_eta(seconds: float) -> str:
    h = int(seconds // 3600); m = int((seconds % 3600) // 60)
    return f"{h}h{m:02d}m"


def main() -> None:
    p = argparse.ArgumentParser(description="Stage 1 ICKG trial (go/no-go gate)")
    add_common_args(p)
    p.add_argument("--smoke", action="store_true", help="tiny end-to-end check")
    args = p.parse_args()
    cfg = Config.from_args(args)
    if args.smoke and not args.n:
        cfg.n = 8
    run(cfg, smoke=args.smoke)


if __name__ == "__main__":
    main()
