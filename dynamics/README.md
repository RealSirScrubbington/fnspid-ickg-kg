# Phase 2 — Temporal dynamics of the financial KG

Analysis of the temporal dynamics of financial relationships on the denoised **core KG**
(46,026 entities / 307,647 edge-instances / 236,416 distinct relationships / 365 weekly
timesteps, 2017–2023). Four method families: temporal EDA, edge-formation velocity, burst
detection, and temporal link prediction. All developed locally in `dynamics/` (Python 3.12,
pandas/numpy/scipy/torch); the RE-GCN was trained on the UCL goldbug GPU.

**Headline finding:** the dynamics are **descriptively rich but forecasting-hard** — velocity and
burst detection recover real market events *to the correct week*, yet no model (static or temporal)
can forecast *novel* relationships beyond a low ceiling. The past is legible; the future is
genuinely surprising.

Two conventions used throughout: a **formation** is the first week a distinct `(subj, rel, obj)`
appears; templated-media boilerplate (Zacks, Motley Fool, …, 422 entities / 8.3% of edges) is
removed via `load_core(drop_noise=True)`.

---

## 1. Temporal EDA (`eda.py`)
- Mean **648 formations/week**, flat across the span (H1 641 ≈ H2 655) — the even monthly
  stratification means edge formation reflects *structure, not sampling density*.
- Continuous series (only the two boundary weeks are empty); high weekly variance with a clear
  2020 elevation and a late-2023 spike.
- Outputs: `data/dynamics/eda/{weekly_stats.csv, temporal_overview.png}`.

## 2. Edge-formation velocity (`velocity.py`)
Velocity = rate of new-relationship formation, on a **centered** 8-week window (so peaks align with
the event, not the window edge), at global / per-relation / per-entity levels.

- Global velocity peaks **2020-10-19** (COVID era). Impact-type relations rose most H2-vs-H1
  (`Negative_Impact_On` ×1.25, `Positive_Impact_On` ×1.22).
- **Absolute** leaderboard (peak velocity) is dominated by market hubs; onset dates are exact:
  Pfizer **2020-11-09** (vaccine Phase-3 readout), Coronavirus 2020-03-16, GameStop **2021-01-25**
  (the squeeze), Gold 2023-10 (Israel–Hamas).
- **Relative** (z-score) leaderboard surfaces specific, datable entities above the hubs.
  **Audit fix applied:** the baseline μ/σ now EXCLUDE the window around each entity's peak
  (including the burst contaminated its own baseline and deflated z), with an σ-floor of 1
  formation (debut-spike entities have a zero out-of-peak baseline). Fixed dedup-core leaderboard:
  SVB z=41 (collapse week), Ant Group z=35 (cancelled-IPO week), Robinhood z=29 (squeeze week),
  Semiconductors z=29 (CHIPS Act), Vaccine z=27 (readout week).
- Outputs: `data/dynamics/velocity/{velocity_entities.csv, velocity_overview.png}`.

## 3. Burst detection (`burst.py`) — three detectors
Three complementary detectors, run together as a robustness check:
1. **Kleinberg** 2-state model → discrete burst **intervals** (start, end, intensity) on global,
   per-relation, and per-entity series.
2. **CUSUM** control chart (Page 1954) → sustained upward regime shifts on the standardised series
   (`cusum_bursts`, reference value k=0.5, decision interval h=5).
3. **BOCD** (Adams & MacKay 2007, `bocd_bursts`) → exact online Gamma–Poisson run-length posterior
   (log-space predictive; changepoints back-dated to the inferred onset t − run_length — audit
   fixes). The posterior is online, but interval extraction (like Kleinberg's/CUSUM's baselines) is
   retrospective — all three are descriptive tools; the walk-forward system is themes.py.
   O(T²)/series, so run on the global + gold-event series.
   **Dedup-core event recovery, each detector vs its own chance level (dilated flagged-week
   coverage — audit fix: BOCD's regime-scale intervals would otherwise buy recall with width):
   Kleinberg 78% vs 16% chance (+62pp), CUSUM 83% vs 24% (+59pp), BOCD 94% vs 29% (+66pp);
   union 18/18 = 100%** (back-dated BOCD catches Magnificent Seven, missed by all before; its one
   miss, the IRA deal, is caught by CUSUM). Recovery is demonstrably not algorithm-dependent.
   Global: CUSUM-vs-BOCD Jaccard 0.81 (both see the same macro-regimes).
- Global (Kleinberg): spring-2020 crash (Apr 6 → May 11) and the fall-2020 peak (Sep 14 → Nov 9, intensity 3667).
- Entity burst catalog (3,286 bursts / 2,003 entities) recovers correctly-timed event windows:
  Pfizer (vaccine), Inflation (2021-11→2023-01), Boeing (737-MAX), Aurora Cannabis (cannabis
  boom), Disney (Disney+ / COVID).
- **Cross-method agreement:** Kleinberg segments discrete events while CUSUM reads sustained regimes;
  on the global series **93% of Kleinberg burst-weeks nest inside CUSUM's intervals** (Jaccard 0.19),
  so detected epochs are *not* an artifact of one algorithm. (The 2-state Kleinberg merges sustained
  regimes into long bursts — exactly what CUSUM captures explicitly; hierarchical Kleinberg would
  separate spikes from sustained elevation.)
- Outputs: `data/dynamics/burst/{burst_catalog.csv, global_bursts.png}` (figure overlays both detectors).

## 4. Temporal link prediction (`linkpred*.py`, `complex_kge.py`, `regcn.py`)
**Protocol (strict PIT):** forecasting / extrapolation — for each test quad `(s,r,o,t)`, rank all
candidate entities using only history `< t`; raw (unfiltered) ranking; MRR & Hits@k averaged over
object- and subject-prediction. Chronological split: train wks 0–254 / valid 255–309 / test 310–364.

**Benchmark (test):**

| method | MRR all | MRR recurring (28%) | MRR **novel (72%)** | H@10 |
|---|---:|---:|---:|---:|
| popularity | 0.036 | 0.076 | 0.020 | 0.082 |
| recent CN / AA (12-wk) | 0.019–0.021 | 0.051–0.056 | 0.007–0.008 | 0.043–0.045 |
| RE-GCN (temporal GNN, valid-selected) | 0.073 | 0.214 | 0.019 | 0.119 |
| ComplEx (static embedding) | 0.082 | 0.231 | 0.025 | 0.155 |
| ChronoBERT-2022 (PIT-clean text) | 0.082 | 0.200 | 0.037 | 0.151 |
| recurrence | 0.135 | 0.487 | 0.000 | 0.205 |
| backoff(recurrence→ComplEx) | 0.152 | 0.487 | 0.025 | 0.242 |
| **backoff(recurrence→ChronoBERT)** | **0.162** | 0.487 | **0.037** | **0.258** |

**Findings:**
- The signal is **bimodal**: recurring relationships are highly predictable (MRR 0.49); novel ones
  hit a low ceiling — best **0.037** (PIT-clean ChronoBERT semantic embeddings) vs ~0.025 for the structural models.
- **72% of test edges are novel.** Recurrence is blind to them; ComplEx is the best single predictor
  on novel edges; cheap structural heuristics (common-neighbor / Adamic-Adar) are far weaker — so
  triadic closure / homophily does *not* transfer to this financial co-mention KG.
- **The temporal GNN adds no forecasting advantage.** RE-GCN (RE-GCN-style: per-week relational
  message pass + GRU evolving entity embeddings, DistMult decoder; valid-based early stopping at
  epoch 21) reaches test MRR 0.073 / novel 0.019 — *below* static ComplEx. A larger/longer run
  overfit badly (train loss ↓ to 4.6, test MRR ↓ to 0.073). Time-evolving embeddings carry no extra
  forecasting signal over static structure here. The best system is `backoff(recurrence→ChronoBERT)`
  (MRR 0.162, H@10 0.258, PIT-clean), just above `backoff(rec→ComplEx)` 0.152 — semantic priors help novel edges.
- Outputs: `data/dynamics/linkpred/{unified_results.csv, temporal_heur_results.csv, regcn_results.csv}`.

---

## Robustness of the link-prediction conclusion (four checks)
- **Apples-to-apples.** RE-GCN retrained on wks 0–309 (ComplEx's exact data, fixed 21 epochs) →
  MRR 0.090 / novel 0.025, indistinguishable from ComplEx; the backoff hybrids tie at 0.152. The
  temporal architecture buys nothing — the earlier 0–254 holdout was not the cause.
- **Time-shuffle placebo.** Randomizing the order of the history snapshots leaves RE-GCN essentially
  unchanged (chronological 0.0898 → shuffled 0.0914). A temporal model invariant to time-order uses
  *no* temporal signal → the null result is **intrinsic to the data**, not a tuning artifact.
- **Blinded extraction (leakage probe, 300 articles, `blinded_check.py`).** Re-extracting with entity
  names + dates masked **retains the relational structure** (valid triples/article 16.8 → 21.8;
  relation-mix TV distance 0.10; `Operate_In` share 0.317 → 0.305) → extraction is text-grounded, not
  entity-recognition-driven. **Caveat:** causal *impact* relations drop ~30% under blinding (impact
  share 27.5% → 21.1%, reallocated to generic `Relate_To`), so causal attribution carries some
  entity-dependence — *hindsight vs naming-specificity is not separable by this test*. This flags the
  impact-relation velocity/burst analyses (not link prediction, which rests on blinding-robust
  structural relations); burst onset *timing* remains text-anchored (sharp, correctly-dated peaks).
- **ChronoBERT cutoff test (lookahead probe, `text_embed.py`/`text_linkpred.py`).** Entity embeddings
  from two versions of the *same* model differing only in cutoff — `chrono-bert-2022` (blind to 2023)
  vs `chrono-bert-2024` (sees it) — scored on link prediction: novel MRR 0.0374 vs 0.0391 (overall &
  H@10 marginally *lower* for 2024). Paired bootstrap (`bootstrap_ci.py`): the novel-edge gap is
  **+0.0010 MRR [95% CI 0.000, 0.002]** — small and at the edge of significance, so lookahead is
  **bounded and ~13x below the semantic effect (+0.0133)**, not provably zero (the strongest,
  capacity-controlled check; He et al. 2025). Qwen-14B's higher novel MRR (0.0405) is model
  *capacity*, not lookahead — the same-size 2024 model with the same future knowledge barely moves.

**Method critique (`resolve_check.py`).** A crude entity resolution (corporate-suffix merge,
46k→38k entities) cuts test novelty only 72.4%→68.4% and lifts recurrence MRR ~9% — so surface-form
fragmentation inflates novelty but is *secondary*; "forecasting-hard" survives (a lower bound).

**Systematic event recovery (`event_validate.py`).** On 18 major 2017–2023 events (listed
independently of the results), burst detection recovers **89% (16/18)** with **median 1-week timing**
(75% within 3 wk); the two misses (negative oil, FTX) are thin-coverage entities whose local peaks
still align but fall below the burst-significance threshold. Global velocity is *not* elevated at
event weeks (p=0.18) → events are entity-localised, not global spikes.

**Statistical rigour (`bootstrap_ci.py`).** 95% paired-bootstrap CIs + 3 RE-GCN seeds (all 0.092 ±
0.005). The semantic gain (+0.0133 [0.012, 0.015]) and the new-best backoff (+0.0096 [0.009, 0.011])
are clearly significant; the lookahead gap (+0.0010 [0.000, 0.002]) is small and marginal — bounded,
not zero. RE-GCN beats ComplEx overall but *ties* on novel edges (no temporal forecasting advantage).

## Other limitations
- **US-centricity & unresolved entities** carry over from the substrate (entity resolution is the
  parallel thesis; ~66% singleton entities in the full graph).
- **Data scale (now tested — see 600k section below).** Novel edges cap forecasting accuracy; the
  lever is converting novel→recurring via denser sampling. The 600k rebuild confirms this directly.

## Reproduce
`python -m dynamics.<module>` for `eda`, `velocity`, `burst`, `linkpred`, `linkpred_eval`
(baselines + hybrids), `temporal_heur`, `complex_kge`, `regcn` (`--epochs/--dim/--hist/--patience`),
`text_embed`/`text_linkpred` (ChronoBERT/Qwen), `bootstrap_ci` (95% CIs), `event_validate`,
`resolve_check`. `loader.py` provides `load_core(drop_noise=True)`; set `KG_CORE_PATH=data/kg_600k_core`
to point any module at the 600k core. RE-GCN runs on goldbug GPU via the `.venv-blackwell` environment.

## 600k scale-up — results
The full pipeline was re-run on a 3× larger sample (604,800 articles → 11.5M triplets → denser core
**41,888 entities / 493,238 edges**, 11.8 vs 6.7 edges/entity). Consolidated in `../results_600k/`
(`RESULTS_600k.md`, link-pred/burst/regcn CSVs, `global_bursts_600k.png`, `reanalysis600.log`).

| metric | 200k core | 600k core |
|---|---:|---:|
| test novelty | 72% | **64%** |
| recurrence MRR | 0.135 | 0.167 |
| ComplEx / RE-GCN / ChronoBERT (novel) | 0.025 / 0.025 / 0.037 | 0.028 / **0.034** / 0.039 |
| best backoff(rec→ChronoBERT) | 0.162 / H@10 0.258 | **0.192 / 0.312** |

- **Density lowers the ceiling:** novelty 72→64%, best forecasting MRR +18% — the "forecasting-hard"
  ceiling is *partly* a data-density artifact.
- **Temporal edge emerges with scale:** RE-GCN beats ComplEx on novel (0.034±.001 over 3 seeds vs
  0.028 [CI .028–.029]) — absent at 200k. (Seed-range separation is indicative; a paired per-query
  test is the fully rigorous version.)
- **Lookahead — RETRACTED as "significant", now a bound:** the +0.0017 cutoff gap initially tested
  SIG under the query bootstrap, but scorer-seed replication shows training noise is the same size
  (3-seed novel ranges: chrono-2022 .0382–.0386, chrono-2024 .0374–.0385 — fully overlapping).
  Correct claim: **lookahead bias < ~0.002 MRR, below measurement resolution.** The semantic effect
  (~+0.011) is an order of magnitude above both and robust.
- **Descriptive recovery improved:** event recall 89→94% (17/18); on 600k, 97% of Kleinberg
  burst-weeks nest inside CUSUM's regimes (200k: 93%).
- **Duplication audit:** FNSPID syndicates one article across ticker rows — 24.3% of the 600k subset
  (10.5% of 200k) are exact duplicate bodies; dedup halves the core (41,888→21,216 entities,
  `data/kg_600k_dedup_core`). Cross-core MRRs are NOT comparable (candidate-set size); the defensible
  cross-corpus claims are the novelty trajectory (72→64→61.6%) and within-core method rankings.
- **CANONICAL BENCHMARK (dedup core, complete — see `../results_600k/RESULTS_600k.md` for the full
  table):** best = backoff(rec→ChronoBERT) **0.218 [.215,.221] / H@10 0.342**; novel-edge ordering
  ChronoBERT 0.0354 > RE-GCN 0.0308 (early-stopped on this core, epoch 22) > ComplEx 0.0245 — the
  SAME ranking as the 200k and duplicated-600k cores (replication). **Lookahead on the canonical core:
  +0.0005 [−0.0009, +0.0019] n.s.** — confirms the "< ~0.002 MRR bound" conclusion empirically.
  CN/AA (fixed): 0.025/0.029 all, ~0.008 novel — negative result replicates.
- **VELOCITY-FEATURE ABLATION (dedup core, `regcn.py --velocity-features` + `ablation_eval.py`):**
  the theme detector's three trailing per-entity statistics (log1p obs, log1p exp, clipped z;
  exact 4wk/26wk machinery) linearly projected into RE-GCN's weekly evolution pre-activation;
  2 arms × 3 seeds, identical hyperparameters + early stopping (baseline arm bit-identical to the
  frozen model; 3-seed mean 0.120 all / 0.031 novel reproduces the benchmark entry). Paired
  per-query bootstrap on seed-averaged reciprocal ranks (B=1000, 49,620 test directions):
  **velocity features significantly DEGRADE forecasting** — all −0.015 [−.016,−.014], recurring
  −0.025 [−.027,−.022], novel −0.009 [−.010,−.008]; seed spread ≤ 0.008 (gap ≫ spread). Burst
  statistics say THAT an entity is active, not WHICH link forms: **velocity is a detection signal,
  not a forecasting feature** (one integration design tested — input-side linear injection).
  Outputs: `data/dynamics/linkpred/regcn_{results,ranks}_abl_*.{csv,npz}`.

## Emerging-theme detection (`themes.py`, `themes_eval.py`) — the headline system
Walk-forward detector of anomalously accelerating themes (the thesis brief's core deliverable), run
on the deduplicated 600k core. Per week t (trailing windows only): Poisson-surprise acceleration per
entity (4wk recent vs 26wk baseline; new entities score high by design) → Louvain clustering of
accelerating entities on the trailing 8wk co-occurrence subgraph → themes ranked by excess
formations, tracked into lifelines (Jaccard ≥ 0.3) with **birth week = live flag date**. Noise
controls, both data-driven: template-entity exclusion (title+body slot-fingerprints,
`themes/template_entities.txt`) and habituation freshness (EMERGING vs recurring). Two configs:
lenient / strict (`--strict` adds walk-forward ubiquity exclusion; dissolves star-shaped screener
rituals, costs ~3wk of COVID lead).
**Gold-list evaluation** (frozen 10 events, word-boundary matching, birth window −13..+8wk):
**8/10 detected, median lead −1wk** — anticipatable themes early (COVID **−3wk** born 2020-02-03,
oil crash −2wk, vaccine race −8wk, vaccine efficacy −6wk), shock events at the news-flow floor
(+1wk: Ukraine, SVB, UAW), GameStop +0. Misses: FTX & ChatGPT (under-covered in a Nasdaq-equity
corpus). Lead time reflects the event's information structure. Bonus discoveries: cannabis boom
born 2019-03-18, EV/SPAC wave 2020-06-22, stay-at-home complex 2020-04-06.
**Placebo control:** random event weeks under the same criterion score ~2/10 (per-theme chance
6–36%); observed 8/10 → **p ≈ 1e-4** (lenient) / **2e-5** (strict). Not an abundance artifact.
**Overdispersion robustness (`--nb`):** weekly formation counts are massively super-Poisson
(NB dispersion α = 3.75, method of moments on train weeks only). Under the negative-binomial
variance the detector becomes high-precision: 102 lifelines (vs 401), **7/10 detected, median +0wk,
chance rate 0.9/10, p ≈ 0**. (α is estimated once on train weeks and applied throughout —
conservative for pre-2022 events; the template blacklist is the one other non-trailing input —
both disclosed.) Poisson-floor = high-recall config, NB = high-precision config; the
gold-list conclusion holds under both.
**Window sensitivity (`--recent/--base/--tag`):** every cell of the 3×3 grid (recent ∈ {2,4,8} ×
base ∈ {13,26,52}) detects 7–8/10 with placebo p < 4e-4; shorter windows detect slightly later,
longer slightly earlier (r8/b26: 8/10, median −2wk). Detection is not a window artifact.
Outputs: `data/dynamics/themes/{theme_weeks,theme_lifelines}_<tag>.csv`.

## Lifeline-level precision audit (`theme_audit.py`, `theme_cohesion.py`, `theme_judge.py`)
The placebo establishes week-level precision; this audit measures **lifeline-level precision**:
are the system's EMERGING discoveries real storylines, and are the co-occurrence edges inside
each theme substantive rather than listicle glue? Protocol (frozen before any judging):
- **Population:** all 282 EMERGING lifelines of the lenient config. Full membership recovered via
  `themes.py --dump-members` (adds `theme_members_lenient.csv`; the rerun reproduced
  `theme_weeks/theme_lifelines_lenient.csv` **byte-identically**, so frozen outputs are untouched
  — the display `members` column is a top-6/28-char label only).
- **Audit packs** (`theme_audit.py` → `audit_packs.jsonl`): per theme, the top-12 members by
  accumulated excess, birth month, and up to 10 source articles from the window [birth − 4wk,
  birth + 1wk] mentioning ≥ 1 member, ranked by distinct members hit; top-3 articles carry a
  450-char body lede. Provenance is deterministic (dedup triple rows → article ids → corpus
  parquet); judges see NO detector scores (blind to z/freshness/rank).
- **Rubric (frozen):** REAL = coherent, externally documentable storyline with genuine coverage;
  TEMPLATE = screener/boilerplate constructs dominate; INCOHERENT = incidental co-occurrence.
- **Judges:** (i) locally-hosted LLM (`theme_judge.py`, vLLM, greedy, fixed prompt); (ii) human
  hand-labels on a seed-42 sample of 15 (`audit_sample_human.csv`); (iii) known controls —
  Coronavirus/GameStop lifelines must judge REAL, the Earnings-ESP ritual TEMPLATE — excluded
  from the estimate; a judge that fails controls or disagrees badly with the human subset is not
  quoted. Reported: share REAL with Wilson 95% CI (all EMERGING + top-K by peak score).
- **Edge-level cohesion** (`theme_cohesion.py` → `cohesion_stats.csv`, `cohesion_packs.jsonl`):
  per theme, binding articles (≥ 2 members), multi3_share (≥ 3 members), pair_coverage, and the
  top member-pairs each with the actual joint article (headline + body lede);
  `theme_judge.py --mode pairs` classifies each binding edge SUBSTANTIVE vs COINCIDENTAL.
  FINDING (controls): raw structural cohesion does NOT certify quality — the boilerplate control
  binds denser (multi3 0.45) than COVID (0.04) because screener articles list many concepts at
  once; the semantic (judged) layer is the primary instrument, cohesion stats are descriptive.
- **Scope:** read-only over frozen artifacts; no entity merging; article text stays local
  (CC BY-NC); truthfulness of the news itself is out of scope — the audit certifies genuine,
  related sourcing, not events.
- **RESULTS (judged 2026-07-12, Qwen2.5-14B-AWQ on huffer; judge validated 11/11 controls +
  13/14 = 93% binary human agreement):** lenient **37.9%** REAL [Wilson 32.5, 43.7] (107/282;
  top-25 by peak score only 28% — rituals rank high); strict 42.9% [36.7, 49.2]; NB 37.1%
  [28.2, 47.0] with the HIGHEST template share (44%) — event-level specificity ≠ stream purity;
  **template-filtered arm 44.7% [38.6, 50.9]** (109/244 — keeps every real discovery AND the
  full gold-list result 8/10 / −1wk / p=5e-5). Post-hoc structural filter ceiling ~50%
  (pair_coverage ≥ 0.5: 47.4% at 68.2% REAL retention) — the residue is semantic.
- **Template-filtered arm (`template_flags.py` v2 + filtered rebuild):** v2 article fingerprints
  (25.1% of corpus; v1 script lost, re-implementation documented in the module docstring; v1
  entity list stays canonical) removed from the dedup triples (−16.9% instances) →
  `data/kg_600k_dedup_tf_core` (same w≥3/d≥3) → identical detector settings (`--tag tf`).
