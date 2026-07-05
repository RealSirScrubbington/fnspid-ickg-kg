# FNSPID → ICKG → FinDKG temporal knowledge graph

A time-indexed financial knowledge graph built by running the open-source **ICKG-v4.2**
extractor over a bounded, time-stratified subset of the **FNSPID** news corpus, output in
**FinDKG-compatible quadruple format**. This is the *input substrate* for an MSc thesis (UCL
CS&ML) on the temporal dynamics of financial relationships (velocity, burst/theme detection,
link prediction). Correctness, point-in-time (PIT) integrity, and reproducibility were
prioritised over scale. **Entity resolution is deliberately out of scope** (a parallel thesis);
only light surface-form normalisation is applied.

## Result (KG summary)
| | |
|---|---|
| entities | **628,765** (unresolved surface forms — see limitations) |
| relations | **15** (FinDKG schema) |
| timesteps | **365** weekly buckets, 2017-01-02 → 2023-12-18 |
| unique edges | **2,379,438** (deduped per `(subj,rel,obj,time)`; mention-count kept as weight) |
| split (chronological) | train **1,640,947** (wks 0–254) · valid **372,732** (255–309) · test **365,759** (310–364) |
| validation | round-trip OK — counts match, no time leakage across splits |

Output files in `data/kg/`: `stat.txt`, `entity2id.txt` (`name⇥id⇥type⇥type_id`),
`relation2id.txt`, `train.txt`/`valid.txt`/`test.txt` (`subj⇥rel⇥obj⇥time⇥index`),
`time2date.txt` (`time_id⇥bucket_start_date`), and `edges_weighted.tsv` (per-edge mention count,
for the velocity signal).

## Corpus subset (reproducible)
- Source: `Zihan1004/FNSPID`, `Stock_news/nasdaq_exteral_data.csv` (full-text `Article` column).
- Window **2017–2023**; body **≥200 words**; **time-stratified** to **2,400 articles/month**
  (even monthly coverage so graph density reflects structure, not sampling skew).
- **201,600 articles**, fixed **seed 42**, one streaming pass (`build_subset.py`); selected
  FNSPID row indices saved (`subset_200k.ids.txt`) + a unique processing id per article.
- Availability ranged from ~7.3k/month (early 2020) to **366,780** (Dec 2023) — all normalised
  to 2,400, so the recency skew does not masquerade as real activity.

## Extractor
- **`victorlxh/ICKG-v4.2`** = a LoRA adapter on **`unsloth/Qwen2.5-14B-Instruct`** (apache-2.0).
  A *local, fixed* model (never a frontier API) so the knowledge cutoff is controllable.
- The **verbatim FinDKG KG-construction prompt** (12 entity types, 15 relation verbs), wrapped
  in Qwen ChatML. Greedy decoding, `max_new_tokens=1280`, bf16.
- **Note:** ICKG-v4.2 emits triplets as Python paren-tuples *and* JSON arrays-of-arrays
  interchangeably; the parser handles both, and filters numeric/percentage "entities" the prompt
  forbids. (Output FORMAT is the first thing to check if a future ICKG version extracts poorly.)

## Extraction quality (200k build)
- **malformed-output rate: 0.00%** · mean **18.5 valid triplets/article** · valid-tuple fraction **97.5%**
- **precision spot-check: 83%** of 300 sampled triplets have *both* entities verbatim in the source
  text (a *lower bound* — ICKG simplifies surface forms, so true precision is higher).
- **GPE (country) edge density: 6.70%** (159,368 edges, 15,407 distinct geopolitical entities).

**Entity-type counts:** CONCEPT 157,291 · PRODUCT 104,557 · SECTOR 79,778 · COMP 63,980 ·
ECON_INDICATOR 58,680 · FIN_INSTRUMENT 44,104 · PERSON 36,312 · EVENT 33,577 · ORG 28,479 ·
GPE 15,407 · ORG/GOV 5,226 · ORG/REG 1,374.

**Relation distribution (unique edges):** Operate_In 678,556 · Relate_To 491,214 ·
Positive_Impact_On 317,766 · Has 156,123 · Raise 125,377 · Negative_Impact_On 107,619 ·
Control 97,988 · Invests_In 89,618 · Produce 88,219 · Announce 59,518 · Is_Member_Of 51,297 ·
Impact 45,206 · Introduce 36,460 · Participates_In 18,258 · Decrease 16,219.

## Point-in-time integrity
- Every triplet is dated by the **article publication date**, never by any event the article
  describes. Time-bucket resolution is a **config flag** (`--time-resolution week|fortnight|month`;
  weekly here). Train/valid/test are split strictly by time-step (never random) and validated for
  no cross-split leakage.
- **Blinded-vs-unblinded diagnostic** (`pit_sample.py`): a 100-article sample extracted with the
  publication date + ticker masked vs unmasked, to *measure* how much the extractor leans on
  recognisable entities/world knowledge. _Caveat:_ Qwen2.5's pretraining post-dates the entire
  2017–2023 corpus, so residual hindsight is plausible — hence this is measured, not assumed.
  **Result (100-article sample):** masking the publication date + ticker leaves the triplet *count*
  essentially unchanged (unmasked **13.8** vs masked **13.7** valid edges/article) but **only 55% of the
  exact edges survive masking** — the extractor is count-stable yet moderately sensitive to the masked
  cues. This is a basic hook (masks dates+ticker, not full entity names); the rigorous hindsight study
  (more complete masking) is the downstream dynamics phase. Per-article data: `data/pit_sample_compare.csv`.

## Pipeline / reproduce
`build_subset.py` (stratified subset) → `extract.py` (vLLM ICKG, sharded data-parallel,
CSV-checkpointed/resumable; `run_build.sh` launches one instance per GPU) → `assemble.py`
(→ FinDKG files) → this report. Built on **2× RTX PRO 6000 Blackwell** (96 GB), ~14 h wall-time,
~2 art/s/GPU.

## Noise characterisation & denoising
Noise is **measured and controlled-for, not destructively scrubbed** — the raw substrate is preserved
so the (out-of-scope) entity-resolution thesis and the noise study itself stay valid.

**Measured noise (full graph):**
- **Long tail (dominant):** 66% of entities are singletons (degree 1); 83% of edges are single-article
  (weight 1), only 6.9% supported by ≥3 articles.
- **Fragmentation (resolution headroom):** 628,765 surface forms → 564,922 crude-normalised keys
  (≈**1.11× lower bound**), concentrated in head companies (Alibaba 20 forms, PayPal 18, MarketAxess 17…).
- **Type skew:** CONCEPT = 25% of entities (often vague themes).
- **Metadata heterogeneity:** the `Publisher` field is empty for **95.7%** of articles (a few have
  Russian-language category labels) — so templated boilerplate is identifiable by *content*, not publisher.
- Plus the **blinded-PIT** extractor-hindsight measurement (above).

**Denoising lever — a non-destructive "core" graph** (`build_core.py`, same temporal axis):
thresholding to edge weight ≥2 + entity degree ≥2 collapses the long tail:

| | full | core (w≥2, deg≥2) |
|---|---|---|
| entities | 628,765 | **46,026** (7.3%) |
| edges | 2,379,438 | **307,647** (12.9%) |
| train/valid/test | 1.64M / 373k / 366k | 215k / 43k / 49k |

The core is training-ready (COMP-dominated; CONCEPT share 25%→14%). Thresholds are config flags so the
dynamics analysis can tune them per-experiment; edge weights, entity degree, and per-triplet
source-grounding are the available controls. Full entity resolution remains the parallel thesis.

## Known limitations (measured, not hidden)
- **Entity resolution OUT OF SCOPE** → **628k unresolved surface forms** (e.g. "Apple"/"Apple Inc."
  separate). The downstream resolution thesis collapses these; counts here are raw surface forms.
- **US-centricity** → thin country/regulator relations (ORG/REG 1,374; GPE 6.7% of edges). Measured
  and documented; *not* corrected by curating a non-random "global" subset.
- **Templated/boilerplate noise** — repetitive options/dividend/Zacks-style articles inflate some
  entities/edges (the `Publisher` field is mostly empty, so this is content- not publisher-identified);
  removed/downweighted by the edge-weight column and the `kg_core` thresholds (see Noise section).
- **Extraction noise** — ~17% of valid tuples were filtered (numeric/unmappable); residual
  per-article relation skew (e.g. over-use of `Operate_In`) is inherent extractor noise.
- **Licensing** — ICKG + FNSPID are non-commercial/research; source article text is **not**
  redistributed (only ids + extracted triplets).
