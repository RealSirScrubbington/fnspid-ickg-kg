# Backburner: re-run at ~600k articles

The pipeline is parametrized — a 600k re-run is a config change, **writes to separate
output dirs** (the current 200k results are untouched), and is fully resumable.
**ETA ~42 h** on the 2× RTX PRO 6000 (safely inside the 3-day GPU cap).

## Launch (on goldbug, in the project dir)
```bash
cd /cs/student/projects3/csml/2025/dmaruev/fnspid-ickg-kg
source goldbug_env.sh && . .venv-blackwell/bin/activate

# 1. Stratified subset: 7,200/month × 84 ≈ 604,800 (sparsest month has ~7,327, so even
#    monthly coverage still holds). Streams the 23 GB once (~4 min). Unique ids auto-assigned.
nohup python build_subset.py --per-month 7200 --seed 42 --out data/subset_600k.parquet > subset_600k.log 2>&1 &

# 2. Data-parallel extraction (~42 h, resumable). Writes data/tripl_600k_{0,1}.csv (+ logs).
#    Re-run the SAME line to resume after any interruption.
nohup bash run_build.sh data/subset_600k.parquet data/tripl_600k > build_600k.log 2>&1 &

# 3. Assemble + denoised core (non-GPU, minutes)
python assemble.py --inputs data/tripl_600k_0.csv data/tripl_600k_1.csv --outdir data/kg_600k --time-resolution week
python build_core.py --kg data/kg_600k --out data/kg_600k_core --min-weight 3 --min-degree 3
```

## Notes
- **Even stratification preserved**: 7,200/month is below the sparsest month's availability
  (~7,327 after the ≥200-word filter), so every month still contributes 7,200.
- **Core thresholds raised to weight≥3, degree≥3**: with 3× the articles there's more support
  per edge, so a stricter core stays *trainable* and *higher quality*. Tune `--min-weight/--min-degree`.
- **1M (~70 h)** is right at the 72-h cap — only attempt split across two resumable sessions.
- **Monitor** extraction with the `bash -s` watchdog pattern (stall/fatal/heartbeat) — remote
  greps must run under `bash`, not the tcsh login shell.
- Higher-leverage than more articles for the *velocity* signal: **entity resolution** (merging the
  fragmented head entities, e.g. Alibaba's 20 surface forms) — the parallel thesis.
