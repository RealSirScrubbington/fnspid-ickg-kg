# 600k-Article Run — Results & Methods (Phase-2 Re-analysis)

Companion to the 200k deliverables (`dynamics/README.md`, `FNSPID_ICKG_project_report.pdf`, and the
lit-review Google Doc). **Same pipeline and methods as the 200k run**; this records the 3× larger
re-run and its results. Generated 2026-06-28.

All raw numbers in this file are reproduced from the run logs in this folder:
`reanalysis600.log` (bootstrap CIs + RE-GCN seeds), `linkpred_unified_600k.csv` (baseline table),
`burst_catalog_600k.csv` + `global_bursts_600k.png` (burst detection).

---

## 1. What changed vs the 200k run
- **Sampling:** `build_subset.py --per-month 7200` (vs 2400) → **604,800 articles**, even 7,200/month
  across all 84 months (2017–2023), seed 42, body ≥200 words.
- **Extraction:** identical — ICKG-v4.2 (LoRA on Qwen2.5-14B-Instruct) via vLLM, both RTX PRO 6000
  GPUs on goldbug, ~43 h, **11.51 M triplets**, 0 errors, 0% malformed.
- **Assembly:** `assemble.py` → full `kg_600k`; core via `build_core.py --min-weight 3 --min-degree 3`
  (stricter than the 200k's 2/2, to keep the *denser* core trainable and high-support).
- **Analysis:** identical `dynamics/` code, pointed at the 600k core via a new **`KG_CORE_PATH`** env
  switch added to `loader.load_core`. GPU link-pred ran on goldbug; CPU descriptive scripts ran locally.

## 2. Graph size — the 600k core is DENSER
| | 200k | 600k |
|---|---:|---:|
| valid triplets | 3.82 M | **11.51 M** |
| full entities | 628,765 | **1,298,833** |
| full edges | 2.38 M | **6.13 M** |
| core thresholds | w≥2, d≥2 | w≥3, d≥3 |
| **core entities** | 46,026 | 41,888 |
| **core edges** | 307,647 | **493,238** |
| **edges / entity** | 6.7 | **11.8** |

Despite stricter thresholds, 3× the articles give a core with **11.8 vs 6.7 edges/entity**.

## 3. Link prediction (PIT forecasting, raw rank)
Test queries per direction: 70,742 | **recurring 36% / novel 64%** (200k was 28% / 72%).
Point estimates from `linkpred_unified_600k.csv`; 95% CIs (B=1000, 141,484 directions) from `reanalysis600.log`.

| method | MRR/all | MRR/recurring | MRR/novel | H@10 |
|---|---:|---:|---:|---:|
| recurrence | 0.167 | 0.464 | 0.000 | 0.260 |
| popularity | 0.039 | 0.071 | 0.021 | 0.088 |
| ComplEx | 0.094 | 0.212 | 0.028 | 0.179 |
| **RE-GCN** (3-seed mean) | 0.120 ±.003 | 0.273 | **0.034 ±.001** | 0.203 |
| ChronoBERT-2022 (PIT-clean) | 0.090 | 0.189 | 0.039 | 0.176 |
| ChronoBERT-2024 (lookahead) | 0.093 | — | 0.041 | — |
| backoff(rec→ComplEx) | 0.185 | 0.464 | 0.028 | 0.298 |
| backoff(rec→RE-GCN) | 0.189 | 0.464 | 0.034 | 0.304 |
| **backoff(rec→ChronoBERT) — BEST** | **0.192** | 0.464 | 0.039 | **0.312** |

**95% bootstrap CIs (novel):** ComplEx 0.0284 [0.0276, 0.0292] · ChronoBERT-2022 0.0389 [0.0380, 0.0399]
· ChronoBERT-2024 0.0406 [0.0397, 0.0416] · RE-GCN seeds 0.0338–0.0355.

**Key gaps (paired query-bootstrap, with the seed-replication caveat added 2026-06-29):**
- **Lookahead (CORRECTED — a bound, not a detection):** the +0.0017 [+0.0009, +0.0024] gap tested SIG
  under the query bootstrap, but that bootstrap conditions on ONE training run. A 3-seed scorer
  replication shows training noise is the same size (chrono-2022 novel range .0382–.0386, chrono-2024
  .0374–.0385 — fully overlapping, 2024 not consistently higher). Correct claim: **lookahead bias is
  below measurement resolution (< ~0.002 MRR)**.
- **Semantic** (ChronoBERT-2022 − ComplEx, novel): **+0.0106 [+0.0096, +0.0115]** — an order of
  magnitude above seed noise and the lookahead bound; robust.
- **New best** (backoff ChronoBERT − ComplEx, all): **+0.0068 [+0.0061, +0.0074]** — robust.
- **Temporal** (RE-GCN 3-seed novel 0.0338–0.0355 vs ComplEx CI [0.0276, 0.0292]): RE-GCN beats
  ComplEx on novel edges — an edge **absent** in the 200k run. (Seed-range separation is indicative;
  a paired per-query test is the fully rigorous version.)

## 4. Headline findings (corrected 2026-06-29 after audit)
1. **Novelty 72% → 64% → 61.6% (deduplicated)** — denser sampling converts novel→recurring; the
   "forecasting-hard" ceiling was **partly a data-density artifact**, not purely intrinsic.
2. **Within-core method rankings preserved and strengthened.** CAVEAT: absolute MRRs are **not
   comparable across cores** (raw ranking over 21k–46k candidates differs mechanically) — quote the
   novelty trajectory and within-core rankings, not cross-core MRR deltas.
3. **A small temporal signal emerges on denser data** — RE-GCN > ComplEx on novel (0.034 vs 0.028;
   seed-range vs CI separation, indicative); on the sparse 200k core the temporal GNN had **no** advantage.
4. **Semantic prior still wins on novel** — ChronoBERT (0.039) > RE-GCN (0.034) > ComplEx (0.028); best
   system unchanged in kind (recurrence → ChronoBERT backoff).
5. **Lookahead bias: below measurement resolution (< ~0.002 MRR)** — the initially-reported "significant
   +0.0017" was retracted after seed replication showed training noise of the same size (see §3).

## 4b. Duplication audit & deduplicated rebuild (sensitivity — added 2026-06-29)
FNSPID stores one row per article-ticker pair; syndicated bodies recur verbatim. **24.3% of the 600k
subset (10.5% of 200k) are exact duplicate bodies** (101,346/102,426 dup groups differ only by ticker;
only 476 span >1 week, so the cross-week "fake recurrence" channel is small — the main effect is
same-week weight inflation, which biased core membership at min-weight ≥ 3).
Deduplicated rebuild (`data/kg_600k_dedup_core`, same thresholds): 21,216 entities / 212,294 edges.
**Conclusions survive and strengthen:** novelty **61.6%**, recurrence MRR **0.196**. The deduplicated
core is CANONICAL (themes + benchmark). Artifacts: `data/dedup_drop_ids_600k.txt`,
`data/tripl_600k_{0,1}_dedup.csv`.

**FULL CANONICAL BENCHMARK (dedup core, completed 2026-07-03, all local on 6GB GPU):**
| method | MRR all [95% CI] | MRR novel [CI] | H@10 |
|---|---|---|---|
| recurrence | 0.196 [.193,.199] | 0.000 | 0.296 |
| popularity | 0.053 | 0.025 | 0.119 |
| CN / AA (self-candidate-fixed) | 0.025 / 0.029 | 0.0076 / 0.0080 | 0.059 / 0.060 |
| ComplEx | 0.099 | 0.0245 [.0235,.0257] | 0.193 |
| RE-GCN (early-stopped ON this core → epoch 22; seed 0) | 0.117 | 0.0308 | 0.196 |
| ChronoBERT-2022 | 0.105 [.103,.107] | 0.0354 [.0340,.0369] | 0.186 |
| backoff(rec→RE-GCN) | 0.215 | 0.0308 | 0.337 |
| **backoff(rec→ChronoBERT) — best** | **0.218 [.215,.221]** | 0.0354 | **0.342** |

Key gaps on the canonical core (paired bootstrap, 49,620 directions):
- **Lookahead (ChronoBERT-2024 − 2022, novel): +0.0005 [−0.0009, +0.0019] n.s.** — on the cleanest
  core the gap is not significant even before the seed-noise correction; the "< ~0.002 MRR bound"
  conclusion is confirmed empirically.
- Semantic (ChronoBERT − ComplEx, novel): +0.0109 [+0.0094, +0.0125] SIG — consistent (~+0.011) across
  all three cores.
- Method ranking (ChronoBERT > RE-GCN > ComplEx on novel; recurrence backoff best) is IDENTICAL on the
  200k, 600k, and dedup cores — the cross-corpus replication story.
- RE-GCN early stopping on this core selected epoch 22 (vs the 200k-tuned 21 previously transplanted):
  the audit concern was handled and proved immaterial.

## 5. Resolution sensitivity (crude corporate-suffix merge)
Entities 41,888 → 33,773 (merge 19%); **novelty 64% → 60.1%**; recurrence MRR 0.167 → 0.177.
Fragmentation is real but secondary; the conclusions survive.

## 6. Burst detection — two detectors (Kleinberg + CUSUM)
- **Kleinberg** (discrete events): COVID crash 2020-04, vaccine/election 2020-11, **GameStop 2021-01-25**,
  UAW 2023-09, AI surge 2023-07 (full catalogue: `burst_catalog_600k.csv`, 4,793 bursts / 3,031 entities).
- **CUSUM** (macro-regimes): just two intervals — **2019-12 → 2022-05 (129 wk)** COVID→recovery→inflation,
  and **2023-07 → 2023-12 (22 wk)** AI/rates.
- **Agreement:** 97% of Kleinberg burst-weeks nest inside CUSUM (Jaccard 0.22) → detected epochs are
  **not algorithm-dependent**; the two capture complementary scales. Figure: `global_bursts_600k.png`.

## 7. Event recovery (18-event gold list)
In-KG 18/18; **RECALL 94% (17/18)** (up from 200k's 89%); timing median 2 wk; global velocity not
elevated at event weeks (p=0.21 — events are entity-localised, as in the 200k run).

## 8. Reproducibility
Every 600k result regenerates with the env switch:
```
KG_CORE_PATH=data/kg_600k_core  python -m dynamics.<script>     # link-pred needs a GPU
# ChronoBERT/Qwen link-pred + bootstrap take the 600k embeddings:
KG_CORE_PATH=data/kg_600k_core EMB22=data/dynamics/linkpred/emb600_chrono22.npy \
  EMB24=data/dynamics/linkpred/emb600_chrono24.npy python -m dynamics.bootstrap_ci
```
Inputs (all local): `data/kg_600k/`, `data/kg_600k_core/`,
`data/dynamics/linkpred/emb600_{chrono22,chrono24,qwen}.npy`, raw `data/tripl_600k_{0,1}.csv`,
`data/subset_600k.parquet` (+ `.ids.txt`).
