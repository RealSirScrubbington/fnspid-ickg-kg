# FNSPID → ICKG → FinDKG: a time-indexed financial knowledge graph

Companion repository for the UCL MSc thesis *Inductive Temporal Theme Velocity
Discovery on a Knowledge Graph* (COMP0098, MSc Computational Statistics and
Machine Learning). A temporal financial knowledge graph is built from public
news by running the open-source **ICKG-v4.2** extractor over a time-stratified
subset of the **FNSPID** corpus, output in **FinDKG-compatible quadruple
format**, and used for three experiments under strict point-in-time (PIT)
discipline:

1. **Building the substrate** (thesis Ch. 3): construction at two corpus
   scales, a syndication-duplication audit, and a canonical deduplicated core.
2. **Walk-forward theme discovery** (Ch. 4): causal acceleration scoring,
   burst detection, theme clustering into dated lifelines, gold-list and
   placebo validation, a lifeline-level precision audit, and a distilled
   lightweight judge.
3. **Temporal link prediction** (Ch. 5): a replicated benchmark of edge-level
   predictability, a velocity-feature ablation, and a retraining on the
   template-filtered substrate.

**To reproduce the results, start with [REPRODUCING.md](REPRODUCING.md)** —
it walks the full pipeline stage by stage with commands and expected outputs.

---

## Headline results

**Substrate.** Extraction is clean at scale: 0% malformed generations,
~18.5 valid triples per article, 83% entity-grounding precision (lower
bound) on a 300-triple spot check. A duplication audit shows 10.5% (200k
subset) and 24.3% (600k subset) of articles are verbatim syndication
duplicates; removing them halves the denoised core, and the deduplicated
core is adopted as canonical.

| construction | articles | entities | edge-weeks | test novelty |
|---|---:|---:|---:|---:|
| 200k core (w≥2, d≥2) | 201,600 | 46,026 | 307,647 | 72.4% |
| 600k core (w≥3, d≥3) | 604,800 | 41,888 | 493,238 | 64.0% |
| **600k dedup core (canonical)** | 457,619 unique | **21,216** | **212,294** | **61.6%** |

**Theme discovery.** The walk-forward detector finds **8/10** major
2020–2023 themes at a **median lead of −1 week** against a placebo
expectation of ~2.2/10 (**p ≈ 1e-4**); anticipatable themes come weeks early
(COVID −3, vaccine race −8), shocks sit at the +1-week news-flow floor.
A blind LLM audit puts stream precision at 37.9% (lenient) rising to
**44.7%** under a template-article filter at zero recall cost; a fine-tuned
1.5B judge reproduces the audit labels (held-out binary agreement 82.6%,
PR-AUC 0.766) and retains 8/10 gold detections when used as a REAL-only
stream filter.

**Link prediction.** Edge-level forecasting is bimodal: recurring edges
reach MRR ≈ 0.5 while novel edges sit near a low ceiling for every method.
Best system: recurrence→ChronoBERT backoff, **MRR 0.218** [.215, .221],
Hits@10 0.342 on the canonical core; the method ranking replicates on all
three constructions. Injecting the detector's velocity features into the
temporal network **degrades** forecasting on every subset (−0.015 MRR
overall); retraining on the template-filtered graph moves per-edge skill by
only −0.005 on shared queries while the aggregate falls by 0.038 — template
noise inflates apparent forecastability. Extractor lookahead bias is bounded
below 0.002 MRR by a cutoff-controlled ChronoBERT comparison.

---

## Repository layout

```
├── README.md                  this file
├── REPRODUCING.md             stage-by-stage reproduction guide
├── requirements.txt           analysis environment (see REPRODUCING.md §0)
│
├── build_subset.py            FNSPID → time-stratified article subset (seeded reservoir)
├── select_fnspid.py           FNSPID download/selection helper
├── extract.py                 ICKG-v4.2 via vLLM → raw typed triples (resumable)
├── assemble.py                triples → FinDKG flat files (weekly buckets, chrono splits)
├── build_core.py              denoised core (edge-weight + degree thresholds)
├── pit_sample.py              blinded (masked) re-extraction sample
├── blinded_check.py           blinded-vs-unblinded extraction comparison
├── generate_report.py         regenerates FNSPID_ICKG_project_report.pdf
│
├── dynamics/                  all Experiment 2 & 3 analysis modules (see table below)
├── ickg_kg/                   early extraction package (config, sampler, extractor wrapper)
├── release/                   FNSPID row-id lists: the licensed-clean reproduction path
├── figures/                   thesis figures (vector PDF + PNG previews)
├── thesis/                    LaTeX source of the dissertation
├── results_600k/              consolidated 600k-scale results (RESULTS_600k.md + CSVs)
├── scripts/                   cluster launch scripts (records of the exact GPU invocations)
├── archive/                   pilot scripts and run logs, kept for provenance only
└── data/                      NOT in git: corpora, graphs, model outputs (see .gitignore)
```

Everything runs from the repo root. The environment variable `KG_CORE_PATH`
selects the graph construction (canonical: `data/kg_600k_dedup_core`), so
every module runs unchanged on any core.

## Code map: `dynamics/`

| module | what it does | thesis |
|---|---|---|
| `loader.py` | loads a core into time-indexed structures; `drop_noise=True` removes templated-media entities | 3.8 |
| `eda.py` | descriptive weekly aggregates | 4 |
| `velocity.py` | descriptive velocity/acceleration (centred windows) + per-entity z leaderboards | 4.2 |
| `burst.py` | three burst detectors: Kleinberg, CUSUM, BOCD (Gamma–Poisson, back-dated onsets) | 4.4 |
| `event_validate.py` | 18-event recovery, per detector vs its chance level (dilated coverage) | 4.4 |
| `themes.py` | **the walk-forward theme detector** (trailing Poisson/NB surprise → Louvain → lifelines; template + habituation controls; `--strict`, `--nb`, window flags) | 4.3–4.6 |
| `themes_eval.py` | frozen 10-event gold list: lead times + 100k-draw placebo control | 4.7 |
| `theme_audit.py` | builds blind audit packs (members + provenance headlines, no detector scores) | 4.8 |
| `theme_judge.py` | LLM judge (Qwen2.5-14B-AWQ via vLLM) over audit packs; control-gated | 4.8 |
| `theme_cohesion.py` | structural cohesion statistics (binding articles, pair coverage) | 4.8 |
| `theme_filter.py` | text-free logistic filter distilled from audit labels (pre-registered protocol) | 4.8 |
| `template_flags.py` | v2 template-article fingerprinter → template-filtered substrate | 4.6, 4.8 |
| `judge_distill_data.py` | assembles group-disjoint temporal train/held-out splits from teacher labels | 4.8 |
| `judge_finetune.py` | LoRA fine-tune of the 1.5B judge (completion-only loss, ~8 min consumer GPU) | 4.8 |
| `judge_score.py` | deploys the 1.5B judge: greedy label + exact P(REAL) from completion log-probs | 4.8 |
| `linkpred.py` / `linkpred_eval.py` | PIT ranking harness (raw ranks, mean-rank ties, update-after-scoring) + recurrence/popularity/backoff baselines | 5 |
| `complex_kge.py` | ComplEx/DistMult trained on pre-test weeks only | 5.4 |
| `temporal_heur.py` | Common-Neighbours / Adamic–Adar on a trailing 12-week subgraph | 5.4 |
| `regcn.py` | RE-GCN-style temporal GNN; validation-based early stopping; `--velocity-features` ablation arm | 5.3, 5.7 |
| `ablation_eval.py` | paired per-query bootstrap over ablation arms and retrains | 5.7, 5.9 |
| `text_embed.py` / `text_linkpred.py` | ChronoBERT entity embeddings + PIT-clean semantic scorer; cutoff pair | 5.4, 6 |
| `bootstrap_ci.py` | paired bootstrap CIs + the ChronoBERT cutoff (lookahead) comparison | 5.5, 6 |
| `resolve_check.py` | crude entity-resolution sensitivity bound (resolution itself is out of scope) | 3.8 |
| `figures.py` / `figures2.py` | regenerate every thesis figure (vector PDF) | all |

`dynamics/README.md` carries the full narrative of the analysis phase with
per-module results.

---

## Data access and licences

- **No article text is redistributed.** FNSPID is distributed under
  CC BY-NC 4.0 and the underlying articles remain publisher-copyrighted.
  Reproduction instead flows through `release/`: the exact FNSPID row
  identifiers of every sampled article (`subset_200k.ids.txt`,
  `subset_600k.ids.txt`), the identifiers dropped by the duplication audit
  (`dedup_drop_ids_600k.txt`), and the seeded sampling scripts. Both corpora
  reconstruct exactly from the public FNSPID distribution
  (`Zihan1004/FNSPID` on Hugging Face).
- **Extractor:** `victorlxh/ICKG-v4.2`, a LoRA adapter on
  `unsloth/Qwen2.5-14B-Instruct` (Apache-2.0 base; ICKG for research use).
- **Scope notes:** entity resolution is deliberately out of scope (a
  parallel project on this corpus); only a crude-merge sensitivity bound is
  reported. Country (GPE) relations are sparse in a US-centric corpus and
  documented as a limitation.

## Hardware

Extraction ran on 2× RTX Pro 6000 (~14 h for 200k, ~43 h for 600k articles).
**Every analysis reproduces on a single consumer GPU with 6 GB memory**;
the 14B audit judge additionally needs a ~16 GB GPU (vLLM); the 1.5B judge
fine-tune takes ~8 minutes on a 4070-class GPU.

## Citing

If you use this repository, cite the thesis, and for the upstream assets:
FinDKG/ICKG (Li & Sanna Passino, ICAIF 2024, arXiv:2407.10909) and FNSPID
(Dong, Fan & Peng, KDD 2024, arXiv:2402.06698).
