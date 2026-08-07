# Reproducing the results

Every stage below lists the command, the hardware it needs, and the numbers
you should see. Stages 1–3 rebuild the substrate from the public FNSPID
distribution; if you only want to reproduce the *analyses*, you still need
the substrate first, since the graph constructions are too large for git.

Determinism notes up front: the detector, evaluation, and audit pipelines
are exactly seeded and reproduce to the digit. GPU *training* runs (RE-GCN,
ComplEx, the judge fine-tune) are seeded but subject to CUDA
non-determinism; expect results inside the quoted three-seed bands rather
than bit-identical checkpoints.

## 0. Environment

Python 3.12. Install PyTorch first from the CUDA index, then the rest:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Extraction (Stage 2) and the 14B audit judge (Stage 6) additionally need
`vllm` on a Linux GPU host; everything else runs on Windows or Linux with a
single 6 GB GPU.

## 1. Corpus subset (CPU, minutes)

Download the FNSPID news file (`Zihan1004/FNSPID`, file
`Stock_news/nasdaq_exteral_data.csv`; ~23 GB) and run one streaming pass:

```bash
python build_subset.py --per-month 2400 --seed 42 --out data/subset_200k.parquet --source-file <path-to-nasdaq_exteral_data.csv>
```

```bash
python build_subset.py --per-month 7200 --seed 42 --out data/subset_600k.parquet --source-file <path-to-nasdaq_exteral_data.csv>
```

Window 2017–2023, bodies ≥ 200 words, per-month reservoir sampling.
**Expect:** 201,600 and 604,800 articles. Verify against the committed
row-id lists — `data/subset_200k.ids.txt` must equal
`release/subset_200k.ids.txt` (and likewise for 600k). If your pandas
version breaks reservoir determinism, select the `release/` row ids
directly instead; they are the ground truth.

## 2. Extraction (2× large GPUs, ~14 h / ~43 h)

ICKG-v4.2 over the subset via vLLM, one instance per GPU over interleaved
shards, CSV-append checkpointed (re-run the same line to resume):

```bash
bash scripts/run_build.sh data/subset_600k.parquet data/tripl_600k
```

**Expect:** 0% malformed generations, ~18.5 valid triples per article,
~11.5M raw triples at 600k. The parser accepts both output formats the
model emits (JSON arrays ~65%, Python tuples ~35%).

## 3. Assemble the graphs (CPU, minutes)

```bash
python assemble.py --inputs data/tripl_600k_0.csv data/tripl_600k_1.csv --outdir data/kg_600k --time-resolution week
```

```bash
python build_core.py --kg data/kg_600k --out data/kg_600k_core --min-weight 3 --min-degree 3
```

For the **canonical deduplicated core**: drop the FNSPID rows listed in
`release/dedup_drop_ids_600k.txt` from the subset (one copy kept per exact
article body; 147,181 rows → 457,619 unique articles), re-extract or filter
the triple CSVs by the surviving article ids, then re-run `assemble.py` and
`build_core.py` with the same w≥3/d≥3 thresholds into
`data/kg_600k_dedup_core`. The 200k pipeline is identical with
`--min-weight 2 --min-degree 2`.

**Expect** (entities / edge-weeks / test novelty): 200k core 46,026 /
307,647 / 72.4%; 600k core 41,888 / 493,238 / 64.0%; dedup core **21,216 /
212,294 / 61.6%**. Splits are fixed by week: train 0–254, valid 255–309,
test 310–364.

## 4. Theme detector and gold-list validation (CPU/6 GB GPU, minutes)

The single most important check in the repository — the canary. Run:

```bash
KG_CORE_PATH=data/kg_600k_dedup_core python -m dynamics.themes
```

```bash
KG_CORE_PATH=data/kg_600k_dedup_core python -m dynamics.themes_eval lenient
```

**Expect exactly: `detected 8/10 | median lead -1wk` with placebo
`p = 0.0001`.** Variants: `--strict` (8/10, p = 2e-5), `--nb` (7/10, chance
0.9/10), window grid via `--recent {2,4,8} --base {13,26,52}` (all cells
7–8/10, p < 4e-4). Burst-detector cross-check:
`python -m dynamics.burst` and `python -m dynamics.event_validate`
(Kleinberg 14/18, CUSUM 15/18, BOCD 17/18, union 18/18).

## 5. Precision audit (16 GB GPU for the judge)

```bash
KG_CORE_PATH=data/kg_600k_dedup_core python -m dynamics.theme_audit
```

```bash
KG_CORE_PATH=data/kg_600k_dedup_core python -m dynamics.theme_judge
```

The judge (Qwen2.5-14B-Instruct-AWQ, greedy, frozen prompt and rubric) must
pass the control gate before its labels count: 11/11 controls (8
earnings-ritual themes → TEMPLATE, 3 COVID lifelines → REAL).
**Expect precision (share REAL):** lenient 37.9% [32.5, 43.7] of 282;
strict 42.9%; NB 37.1%; template-filtered **44.7%** [38.6, 50.9] of 244.
The template-filtered substrate comes from `dynamics/template_flags.py`
(v2 fingerprints, 25.1% of articles) removed before graph assembly; its
gold list is unchanged (8/10, −1 wk, p = 5e-5). `theme_cohesion.py` and
`theme_filter.py` reproduce the structural-ceiling result (text-free
filtering saturates near 50%; logistic PR-AUC 0.602 vs base 0.379).

## 6. Distilled 1.5B judge (6 GB GPU, ~8 min training)

```bash
KG_CORE_PATH=data/kg_600k_dedup_core python -m dynamics.judge_distill_data
```

```bash
KG_CORE_PATH=data/kg_600k_dedup_core python -m dynamics.judge_finetune
```

```bash
KG_CORE_PATH=data/kg_600k_dedup_core python -m dynamics.judge_score --arms lenient,tf
```

Qwen2.5-1.5B-Instruct + LoRA (r=16, α=32, lr 1e-4, 3 epochs,
completion-only loss, seed 0); splits are storyline-grouped and temporal
(held-out groups born 2022+). **Expect:** zero-shot base FAILS the control
gate; fine-tuned judge passes 9/9 + 3/3. Held-out teacher agreement 82.6%
binary / 69.1% three-way; 20/24 vs the human packs (κ = 0.66). Ranking by
P(REAL): held-out PR-AUC 0.766 (base 0.237); on the 46 held-out lenient
packs 0.756 vs the retrained text-free filter's 0.406 (Δ +0.35, paired
bootstrap CI [+0.13, +0.55]). Calibration: Brier 0.112, ECE 0.074.

## 7. Link-prediction benchmark (6 GB GPU, hours)

```bash
KG_CORE_PATH=data/kg_600k_dedup_core python -m dynamics.linkpred_eval
```

with `python -m dynamics.complex_kge`, `python -m dynamics.temporal_heur`,
`python -m dynamics.regcn` (early stopping on validation weeks),
`python -m dynamics.text_embed` + `text_linkpred` (ChronoBERT), and
`python -m dynamics.bootstrap_ci` for CIs. Protocol: raw (unfiltered)
ranking, mean-rank ties, update-after-scoring, history strictly < t.

**Expect on the canonical core (MRR all / novel / Hits@10):** recurrence
0.196 / 0.000 / 0.296; ComplEx 0.099 / 0.025; RE-GCN 0.117 / 0.031;
ChronoBERT-2022 0.105 / 0.035; **backoff rec→ChronoBERT 0.218
[.215, .221] / 0.035 / 0.342** (best). Novel-edge ordering ChronoBERT >
RE-GCN ≥ ComplEx ≫ CN/AA must replicate on all three cores
(`KG_CORE_PATH` selects). Lookahead check: the 2022-vs-2024 ChronoBERT gap
is +0.0005 [−0.0009, +0.0019] — a bound, not a detection.

## 8. Ablation and template-filtered retrain (6 GB GPU)

```bash
KG_CORE_PATH=data/kg_600k_dedup_core python -m dynamics.regcn --velocity-features
```

```bash
KG_CORE_PATH=data/kg_600k_dedup_core python -m dynamics.ablation_eval
```

Three seeds per arm, identical hyperparameters, paired per-query bootstrap
on seed-averaged reciprocal ranks. **Expect:** velocity features degrade
every subset — all −0.015 [−.016, −.014], recurring −0.025, novel −0.009
(seed spread ≤ 0.008). Retraining the identical configuration with
`KG_CORE_PATH=data/kg_600k_dedup_tf_core` (three seeds): within-tf backoff
mean 0.1767, novel mean 0.0299; paired on the 21,133 shared test queries
the pooled seed-averaged delta is **−0.005** [−0.006, −0.004], an order of
magnitude below the 0.038 aggregate difference — composition, not skill.

## 9. Point-in-time diagnostics

`python pit_sample.py` and `python blinded_check.py` (300-article blinded
re-extraction: relation-mix total-variation distance 0.10; impact-relation
share 27.5% → 21.1%); the time-shuffle placebo on the 200k core leaves
RE-GCN at 0.090 → 0.091 (no temporal-order signal at that density).

## 10. Figures

```bash
KG_CORE_PATH=data/kg_600k_dedup_core python -m dynamics.figures
```

plus `dynamics.figures2` for the audit-era figures. Vector PDFs land in
`figures/`, matching the thesis one-to-one.
