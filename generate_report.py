"""Generate the project report PDF (substrate + dynamics) with reportlab."""
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                Image, PageBreak, HRFlowable)

OUT = "FNSPID_ICKG_project_report.pdf"
NAVY = colors.HexColor("#1a3c5e")
LIGHT = colors.HexColor("#eef3f8")

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=15, textColor=NAVY, spaceBefore=14, spaceAfter=6)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=11.5, textColor=colors.HexColor("#2e5a82"), spaceBefore=9, spaceAfter=4)
BODY = ParagraphStyle("BODY", parent=ss["BodyText"], fontSize=9.5, leading=13.5, alignment=TA_JUSTIFY, spaceAfter=5)
CAP = ParagraphStyle("CAP", parent=ss["BodyText"], fontSize=8, textColor=colors.grey, alignment=TA_CENTER, spaceAfter=8)
BULLET = ParagraphStyle("BULLET", parent=BODY, leftIndent=14, bulletIndent=4, spaceAfter=3)


def P(t): return Paragraph(t, BODY)
def B(t): return Paragraph(t, BULLET, bulletText="•")


def table(data, widths):
    t = Table(data, colWidths=widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 8.3),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cfd8e3")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5)]))
    return t


def figure(path, cap, width=6.2 * inch):
    img = Image(path)
    img.drawHeight = width * img.imageHeight / img.imageWidth
    img.drawWidth = width
    return [img, Paragraph(cap, CAP)]


s = []
title = ParagraphStyle("T", parent=ss["Title"], fontSize=18, textColor=NAVY, spaceAfter=2)
sub = ParagraphStyle("S", parent=ss["Normal"], fontSize=10.5, textColor=colors.HexColor("#555"), alignment=TA_CENTER, spaceAfter=2)
s.append(Paragraph("Temporal Financial Knowledge Graph", title))
s.append(Paragraph("Construction (FNSPID to ICKG to FinDKG) and Dynamics Analysis", sub))
s.append(Paragraph("MSc Computational Statistics &amp; Machine Learning, UCL &nbsp;|&nbsp; Project report (200k + 600k) &nbsp;|&nbsp; 9 July 2026", sub))
s.append(Spacer(1, 6)); s.append(HRFlowable(width="100%", thickness=1, color=NAVY)); s.append(Spacer(1, 8))

s.append(Paragraph("Executive summary", H1))
s.append(P("This project builds a time-indexed financial knowledge graph (KG) from financial news and analyses the "
           "temporal dynamics of the relationships it encodes. <b>Phase 1</b> extracts a FinDKG-format quadruple KG "
           "from 201,600 FNSPID news articles (2017&ndash;2023) using the open ICKG-v4.2 extractor, yielding a "
           "628,765-entity graph and a denoised 46,026-entity core. <b>Phase 2</b> measures edge-formation velocity, "
           "detects bursts, and benchmarks temporal link prediction under strict point-in-time (PIT) discipline."))
s.append(P("<b>Headline finding:</b> the dynamics are <b>descriptively rich but forecasting-hard</b>. Velocity and "
           "burst detection recover real market events to the correct week (the COVID crash, the Pfizer vaccine "
           "readout, the GameStop squeeze, the 2023 UAW strike), yet no model &mdash; static embedding or temporal "
           "GNN &mdash; can forecast <i>novel</i> relationships beyond a low ceiling. The past is legible; the future "
           "is genuinely surprising. This conclusion survives four independent robustness checks, including a direct "
           "probe of the knowledge-cutoff-leakage concern. A velocity-feature ablation completes the picture: "
           "injecting the theme detector's burst statistics into the temporal model significantly <i>degrades</i> "
           "forecasting &mdash; velocity is a detection signal, not a forecasting feature."))
s.append(P("<b>Scale-up (Section 6):</b> a 3x larger 600k-article rebuild confirms this ceiling is partly a "
           "data-density artifact &mdash; denser sampling cuts test novelty 72% to 64% (61.6% after deduplication), "
           "and surfaces a small temporal-model advantage that was absent at 200k."))
s.append(P("<b>Headline system (Section 7):</b> a walk-forward <b>emerging-theme detector</b> built on link-formation "
           "velocity &mdash; the project's central deliverable. Scored against a frozen 10-event gold list it detects "
           "8/10 major 2020&ndash;2023 themes, flagging anticipatable ones weeks early (COVID &minus;3 wk, the vaccine "
           "race &minus;8 wk) and shock events at the news-flow floor (+1 wk). A blind, validated lifeline-level "
           "audit puts the high-recall stream's precision at 37.9%, and an audit-driven template-filtered "
           "configuration raises it to 44.7% with recall unchanged."))

s.append(Paragraph("1.&nbsp;&nbsp;Objective &amp; scope", H1))
s.append(P("Goal: produce a reproducible, PIT-clean temporal financial KG as the substrate for a thesis on the "
           "temporal dynamics of financial relationships, and characterise those dynamics. Correctness, point-in-time "
           "integrity, and reproducibility were prioritised over scale. Entity resolution is deliberately out of scope "
           "(a parallel thesis); only light surface-form normalisation is applied."))

s.append(Paragraph("2.&nbsp;&nbsp;Phase 1 &mdash; Substrate construction", H1))
s.append(Paragraph("2.1&nbsp;&nbsp;Corpus and sampling", H2))
s.append(P("Source: FNSPID <font face='Courier'>nasdaq_exteral_data.csv</font> (full-text articles). A single streaming "
           "pass selects a time-stratified subset &mdash; 2,400 articles per month over 2017&ndash;2023, bodies "
           ">= 200 words, fixed seed &mdash; giving <b>201,600 articles</b> with even monthly coverage so graph density "
           "reflects structure, not sampling skew."))
s.append(Paragraph("2.2&nbsp;&nbsp;Extraction", H2))
s.append(P("Extractor: <font face='Courier'>victorlxh/ICKG-v4.2</font>, a LoRA adapter on Qwen2.5-14B-Instruct &mdash; a "
           "local, fixed model (never a frontier API) so the knowledge cutoff is controllable. The verbatim FinDKG "
           "prompt (12 entity types, 15 relations) was run via vLLM across 2x RTX PRO 6000 GPUs (~14 h)."))
s.append(table([
    ["Extraction quality", "Value"],
    ["Malformed-output rate", "0.00%"],
    ["Mean valid triples / article", "18.5"],
    ["Precision (both entities verbatim in source)", "83% (lower bound)"],
    ["GPE (country) edge density", "6.70%"]], [3.6 * inch, 2.4 * inch]))
s.append(Spacer(1, 6))
s.append(Paragraph("2.3&nbsp;&nbsp;Assembly and denoising", H2))
s.append(P("Triplets are assembled into FinDKG flat files with weekly time-bucketing and chronological train/valid/test "
           "splits (no leakage). A non-destructive denoised <b>core</b> graph is produced by thresholding edge weight "
           ">= 2 and entity degree >= 2, collapsing a long tail (66% of full-graph entities are singletons)."))
s.append(table([
    ["", "Full graph", "Denoised core"],
    ["Entities", "628,765", "46,026"],
    ["Relations", "15", "15"],
    ["Weekly timesteps", "365", "365"],
    ["Edges", "2,379,438", "307,647"],
    ["Train / valid / test edges", "1.64M / 373k / 366k", "215k / 43k / 49k"]],
    [2.4 * inch, 1.9 * inch, 1.9 * inch]))
s.append(Spacer(1, 4))
s.append(P("Noise was measured, not hidden: ~1.11x entity fragmentation (lower bound), and 8.3% of edges came from "
           "templated financial-media boilerplate (Zacks, Motley Fool, ...), removed via an opt-in filter for the analysis."))

s.append(PageBreak())
s.append(Paragraph("3.&nbsp;&nbsp;Phase 2 &mdash; Temporal dynamics", H1))
s.append(P("All analyses run on the 46k core under a strict PIT protocol (history &lt; t only). A 'formation' is the "
           "first week a distinct (subject, relation, object) triple appears."))
s.append(Paragraph("3.1&nbsp;&nbsp;Temporal EDA", H2))
s.append(P("Mean <b>648 formations/week</b>, flat across the span (first half 641 ~ second half 655) &mdash; the even "
           "stratification means edge formation reflects structure, not sampling density. A clear 2020 elevation and a "
           "late-2023 spike are visible."))
s += figure("data/dynamics/eda/temporal_overview.png",
            "Fig 1. Core KG temporal signal: weekly activity, cumulative growth, active entities, and relation mix.")
s.append(Paragraph("3.2&nbsp;&nbsp;Edge-formation velocity", H2))
s.append(P("Velocity = rate of new-relationship formation on a centered 8-week window. Onset dates pin real events to "
           "the week:"))
s.append(table([
    ["Entity", "Onset", "Corresponds to"],
    ["Pfizer", "2020-11-09", "vaccine Phase-3 readout"],
    ["Coronavirus", "2020-03-16", "US lockdowns"],
    ["GameStop", "2021-01-25", "the short squeeze"],
    ["UAW", "2023-09-25", "auto-workers' strike"],
    ["West Virginia", "2022-07-25", "Manchin / IRA deal"],
    ["Magnificent Seven", "2023-11", "term coined that year"]], [1.7 * inch, 1.3 * inch, 3.0 * inch]))
s.append(Spacer(1, 6))
s += figure("data/dynamics/velocity/velocity_overview.png",
            "Fig 2. Global edge-formation velocity & acceleration (left); sharpest-burst entity trajectories (right).")
s.append(Paragraph("3.3&nbsp;&nbsp;Burst detection (three detectors)", H2))
s.append(P("Two complementary detectors are run. A <b>Kleinberg</b> 2-state model extracts discrete burst intervals: "
           "globally the spring-2020 COVID crash and the fall-2020 peak; the entity catalogue (3,286 bursts over 2,003 "
           "entities) recovers correctly-timed windows &mdash; Pfizer (vaccine), Inflation (2021-11 to 2023-01), Boeing "
           "(737-MAX), Aurora Cannabis, Disney+. A <b>CUSUM</b> control chart (Page 1954) instead flags sustained regime "
           "shifts. The two agree on <i>where</i> activity is elevated &mdash; 93% of Kleinberg burst-weeks nest inside "
           "CUSUM's intervals (Jaccard 0.19) &mdash; so detected epochs are not an artifact of one algorithm; they "
           "differ only in granularity (discrete events vs sustained regimes). A third, fully Bayesian detector was "
           "added on the canonical core: <b>Bayesian online changepoint detection</b> (Adams &amp; MacKay 2007, "
           "Gamma-Poisson, changepoints back-dated to the inferred onset). Because the three detectors flag intervals "
           "of very different widths, recall is reported against each detector's <i>chance level</i> (its dilated "
           "flagged-week coverage &mdash; the recall it would achieve on random event weeks): on the 18-event gold "
           "list, Kleinberg 78% vs 16% chance (+62pp), CUSUM 83% vs 24% (+59pp), BOCD 94% vs 29% (+66pp); "
           "<b>three-detector union 18/18 = 100%</b>. All three detect far above chance with comparable excesses, "
           "and recovery is demonstrably not algorithm-dependent."))
s += figure("data/dynamics/burst/global_bursts.png",
            "Fig 3. Global formation series: Kleinberg burst intervals (shaded) vs CUSUM regimes (hatched).")

s.append(Paragraph("3.4&nbsp;&nbsp;Temporal link prediction", H2))
s.append(P("Forecasting protocol: for each test quad, rank all candidate entities using only history &lt; t (raw, "
           "PIT-strict). Metrics are MRR and Hits@10, averaged over object- and subject-prediction. <b>72% of test "
           "edges are novel</b> (never seen before their prediction week)."))
s.append(table([
    ["Method", "MRR all", "MRR recurring", "MRR novel", "Hits@10"],
    ["popularity", "0.036", "0.076", "0.020", "0.082"],
    ["common-neighbour / Adamic-Adar", "0.019-0.021", "0.051-0.056", "0.007-0.008", "0.043-0.045"],
    ["ComplEx (static embedding)", "0.082", "0.231", "0.025", "0.155"],
    ["RE-GCN (temporal GNN)", "0.090", "0.260", "0.025", "0.151"],
    ["ChronoBERT-2022 (PIT-clean text)", "0.082", "0.200", "0.037", "0.151"],
    ["recurrence", "0.135", "0.487", "0.000", "0.205"],
    ["backoff (recurrence -> ComplEx)", "0.152", "0.487", "0.025", "0.242"],
    ["backoff (recurrence -> ChronoBERT)", "0.162", "0.487", "0.037", "0.258"]],
    [2.3 * inch, 0.85 * inch, 1.0 * inch, 0.85 * inch, 0.8 * inch]))
s.append(Spacer(1, 4))
s.append(P("The signal is <b>bimodal</b>: recurring relationships are highly predictable (MRR 0.49), novel ones are hard "
           "for every method (best 0.037). The temporal GNN adds <b>no</b> advantage over static structure. The best "
           "system is a recurrence backoff whose novel-edge fallback is the PIT-clean ChronoBERT embedding (MRR 0.162, "
           "Hits@10 0.258) &mdash; semantic priors lift novel-edge ranking (0.037 vs 0.025) without any lookahead."))

s.append(Paragraph("4.&nbsp;&nbsp;Key findings", H1))
s.append(B("<b>Descriptively rich (systematically validated).</b> On a curated list of 18 major 2017-2023 events "
           "(compiled independently of the results), burst detection recovers <b>89%</b> (16/18) with <b>median "
           "1-week timing</b> (75% within 3 weeks) &mdash; not cherry-picked anecdotes (the chance-controlled "
           "per-detector account on the canonical core is in Section 3.3). The two misses (negative oil "
           "prices, FTX) are thin-coverage entities whose local peaks still align but fall below the burst-significance "
           "threshold. Events are entity-localised: global velocity is not significantly elevated at event weeks (p=0.18)."))
s.append(B("<b>Forecasting-hard.</b> 72% of relationships are novel and near-unforecastable; recurrence dominates "
           "whatever is predictable."))
s.append(B("<b>No temporal-model advantage.</b> A RE-GCN temporal GNN only matches a static ComplEx embedding; "
           "time-evolving representations carry no extra forecasting signal on this KG."))

s.append(Paragraph("5.&nbsp;&nbsp;Robustness &amp; the point-in-time / leakage investigation", H1))
s.append(P("Because the extractor's knowledge cutoff postdates the corpus, hindsight contamination was investigated "
           "directly. Four independent checks all support the conclusion that leakage is negligible:"))
s.append(table([
    ["Check", "Result", "Rules out"],
    ["Apples-to-apples retrain", "RE-GCN 0.090 ~= ComplEx 0.082", "an unfair data split"],
    ["Time-shuffle placebo", "0.0898 -> 0.0914 (no change)", "any temporal-order signal"],
    ["Blinded re-extraction", "structure retained (TV dist 0.10)", "leakage driving KG structure"],
    ["ChronoBERT cutoff test", "novel MRR 0.0374 (blind) ~= 0.0391 (sees 2023)", "lookahead in entity knowledge"]],
    [1.7 * inch, 2.9 * inch, 1.7 * inch]))
s.append(Spacer(1, 4))
s.append(P("The time-shuffle placebo is decisive: a temporal model whose score is <i>invariant</i> to randomising the "
           "order of history is, by definition, using no temporal information &mdash; so the null result is intrinsic "
           "to the data, not a tuning artifact. The blinded re-extraction (300 articles, entity names + dates masked) "
           "retained the relational structure (valid triples/article 16.8 -> 21.8; relation-mix total-variation "
           "distance 0.10), confirming text-grounded extraction. <b>Caveat:</b> causal 'impact' relations fell ~30% "
           "under blinding (share 27.5% -> 21.1%), so causal attributions carry some entity-dependence &mdash; hindsight "
           "versus naming-specificity is not separable by this test. This flags the impact-relation analyses, not link "
           "prediction (which rests on blinding-robust structural relations)."))
s.append(P("<b>Lookahead bias (ChronoBERT cutoff test).</b> Following He et al. (2025), entity-name embeddings were "
           "taken from two versions of the same model differing only in training cutoff &mdash; chrono-bert-2022 (blind "
           "to the 2023 test year) and chrono-bert-2024 (which has ingested it) &mdash; and scored on the same link-"
           "prediction task. The apparent novel-edge gap (+0.0010 MRR at 200k; +0.0017 at 600k) initially tested as "
           "significant under a paired query bootstrap &mdash; but a scorer-seed replication showed that <b>training "
           "noise is the same size as the gap</b> (3-seed novel-MRR ranges: 2022-model 0.0382&ndash;0.0386, 2024-model "
           "0.0374&ndash;0.0385 &mdash; fully overlapping, the 2024 model not even consistently higher). The query "
           "bootstrap conditions on a single training run and cannot see this variance. The honest conclusion is a "
           "<b>bound, not a detection: lookahead bias is below measurement resolution (&lt; ~0.002 MRR)</b> &mdash; and "
           "the deduplicated-core benchmark confirms this empirically: there the gap is not significant even under the "
           "query bootstrap alone (+0.0005 [&minus;0.0009, +0.0019], the 2024 model <i>worse</i> on overall MRR). A larger "
           "Qwen-14B embedding scored higher (novel 0.0405), but that gap is model capacity, not lookahead. The "
           "cutoff-controlled design remains the strongest of the four checks. <b>Bonus (robust to seeds):</b> the "
           "PIT-clean ChronoBERT embeddings beat structural ComplEx on novel edges by ~+0.011&ndash;0.013 MRR &mdash; "
           "an order of magnitude above seed noise and roughly five times the lookahead bound &mdash; and are the "
           "study's best novel-edge predictor; the recurrence->ChronoBERT backoff is the best overall system."))
s.append(P("<b>Structural-only re-run.</b> Re-running velocity and burst with the 5 causal 'impact' relations removed "
           "leaves the core event recovery intact (GameStop 2021-01-25, Boeing 2020-07-20, and the global velocity peak "
           "all identical; COVID within a week) &mdash; confirming the temporal signal is text-anchored, not carried by "
           "the entity-dependent impact relations."))

s.append(Paragraph("6.&nbsp;&nbsp;The 600k scale-up &mdash; denser data lifts the ceiling", H1))
s.append(P("To test whether the forecasting ceiling is intrinsic or an artifact of data density, the entire pipeline "
           "was re-run on a 3x larger sample: <b>604,800 articles</b> (7,200/month, same extractor and protocol), "
           "giving 11.5M triplets and a denser denoised core (<b>41,888 entities / 493,238 edges</b>; 11.8 vs 6.7 edges "
           "per entity). The full Phase-2 analysis was repeated on this core."))
s.append(table([
    ["Metric", "200k core", "600k core"],
    ["Test novelty", "72%", "64%"],
    ["recurrence MRR", "0.135", "0.167"],
    ["ComplEx (novel-edge MRR)", "0.025", "0.028"],
    ["RE-GCN (novel-edge MRR)", "0.025", "0.034"],
    ["ChronoBERT (novel-edge MRR)", "0.037", "0.039"],
    ["best backoff(rec -> ChronoBERT)", "0.162 / H@10 0.258", "0.192 / H@10 0.312"]],
    [2.6 * inch, 1.7 * inch, 1.7 * inch]))
s.append(Spacer(1, 5))
s.append(B("<b>The ceiling is partly a data-density artifact.</b> 3x denser sampling converts novel relationships into "
           "recurring ones, cutting test novelty 72% to 64%, with every method's within-core ranking preserved &mdash; "
           "recurrence dominates and the semantic ChronoBERT prior remains the best novel-edge predictor. <b>Caveat:</b> "
           "absolute MRRs are not comparable across cores (raw ranking over 21k&ndash;46k candidates differs "
           "mechanically), so the defensible cross-corpus claims are the novelty trajectory and the method rankings, "
           "not MRR deltas."))
s.append(B("<b>Duplication audit &rarr; the canonical core.</b> FNSPID stores one row per article-ticker pair, so "
           "syndicated articles recur verbatim: 24.3% of the 600k subset (10.5% of 200k) are exact duplicate bodies, "
           "and per-mention edge weights count them multiply &mdash; deduplication halves the 600k core (41,888 to "
           "21,216 entities), making it the only core whose thresholds mean 'supported by N independent articles'. "
           "The <b>deduplicated core is therefore canonical</b> (themes, Section 7, and the completed benchmark): "
           "novelty <b>61.6%</b>; best system backoff(rec->ChronoBERT) <b>MRR 0.218 [0.215, 0.221], Hits@10 0.342</b>; "
           "novel-edge ordering ChronoBERT 0.035 &gt; RE-GCN 0.031 (early-stopped on this core's own validation weeks, "
           "epoch 22) &gt; ComplEx 0.025 &mdash; the <b>same method ranking as the 200k and duplicated-600k cores</b>, "
           "so the findings replicate across corpus scale, thresholds, and deduplication."))
s.append(B("<b>A temporal-model advantage emerges with scale.</b> On the denser graph RE-GCN beats static ComplEx on "
           "novel edges (0.034 +/- 0.001 over 3 seeds vs 0.028 [CI 0.028, 0.029]) &mdash; an edge <i>absent</i> at "
           "200k, where they tied at 0.025. (A paired per-query test is the fully rigorous comparison; the seed-range "
           "separation is indicative.) Event recovery on the anecdote-free gold list also improved with density; "
           "the chance-controlled three-detector account on the canonical core (Section 3.3) reaches a union recall "
           "of 18/18. A crude resolution pass cuts 600k novelty further to 60%."))
s.append(B("<b>Velocity features do not transfer to forecasting (ablation, canonical core).</b> The theme detector's "
           "three trailing per-entity statistics (log1p observed, log1p expected, clipped surprise z; its exact "
           "4-week/26-week machinery) were injected into RE-GCN's weekly evolution step through a learned projection "
           "&mdash; 2 arms x 3 seeds, identical hyperparameters and early stopping, the baseline arm bit-identical to "
           "the benchmark model (3-seed mean 0.120 all / 0.031 novel reproduces its entry). Paired per-query bootstrap "
           "(49,620 test directions): the features <b>significantly degrade</b> forecasting &mdash; MRR &minus;0.015 "
           "[&minus;0.016, &minus;0.014] overall, &minus;0.009 [&minus;0.010, &minus;0.008] on novel edges, gaps far "
           "above the seed spread (&lt;= 0.008). Burst statistics announce <i>that</i> an entity is active, not "
           "<i>which</i> link will form: <b>velocity is a detection signal, not a forecasting feature</b> (one "
           "integration design tested)."))
s += figure("results_600k/global_bursts_600k.png",
            "Fig 4. 600k global bursts: Kleinberg discrete events (shaded) vs two CUSUM macro-regimes (hatched) &mdash; "
            "the 2019-2022 COVID-to-inflation era and the 2023 AI surge.")

s.append(PageBreak())
s.append(Paragraph("7.&nbsp;&nbsp;Emerging-theme detection &mdash; the headline system", H1))
s.append(P("The project's central deliverable: a <b>walk-forward detector of anomalously accelerating themes</b>, "
           "built on link-formation velocity (per the thesis brief) and run on the deduplicated 600k core. At each "
           "week t, using only history &lt;= t: (1) every entity's formation acceleration is scored by a Poisson "
           "surprise on a trailing 4-week window against its own trailing 26-week baseline (new entities &mdash; "
           "'Coronavirus' in January 2020 &mdash; score high by design); (2) accelerating entities (z >= 3) are "
           "clustered on the trailing 8-week co-occurrence subgraph (Louvain), so a <b>theme</b> is a connected group "
           "of co-accelerating entities, not a lone spike; (3) themes are ranked by excess formations and tracked "
           "week-over-week into <b>lifelines</b> whose birth week is the date the system would have flagged them live. "
           "Two data-driven noise controls: entities whose mentions come mostly from template articles (measured by "
           "title/body slot-fingerprinting) are excluded, and a <b>habituation</b> score (excess-weighted share of "
           "members without prior acceleration episodes) separates EMERGING themes from recurring rituals such as "
           "earnings-season screens. Two configurations are reported &mdash; lenient, and strict (which additionally "
           "excludes walk-forward-ubiquitous background entities, dissolving star-shaped screener clusters at the cost "
           "of ~3 weeks of COVID lead time)."))
s.append(P("<b>Validation</b> is against a frozen 10-event gold list with externally documented dates, word-boundary "
           "entity matching, and an event-anchored birth window (&minus;13 to +8 weeks) so pre-existing topical "
           "lifelines cannot masquerade as detections:"))
s.append(table([
    ["Gold theme", "Event week", "Detected (birth)", "Lead"],
    ["COVID market crisis", "2020-02-24", "2020-02-03", "-3 wk"],
    ["Oil crash (negative WTI)", "2020-04-20", "2020-04-06", "-2 wk"],
    ["Vaccine race (Warp Speed)", "2020-05-15", "2020-03-16", "-8 wk"],
    ["Vaccine efficacy readout", "2020-11-09", "2020-09-28", "-6 wk"],
    ["GameStop squeeze", "2021-01-25", "2021-01-25", "+0 wk"],
    ["Russia-Ukraine war", "2022-02-21", "2022-02-28", "+1 wk"],
    ["SVB banking crisis", "2023-03-06", "2023-03-13", "+1 wk"],
    ["UAW strike", "2023-09-11", "2023-09-18", "+1 wk"],
    ["FTX collapse / ChatGPT wave", "2022-11", "MISS x2", "--"]],
    [2.5 * inch, 1.3 * inch, 1.5 * inch, 0.9 * inch]))
s.append(Spacer(1, 5))
s.append(P("<b>8/10 detected, median lead &minus;1 week</b> &mdash; and the structure of the leads is itself a "
           "finding: <b>anticipatable themes are detected early</b> (the COVID build-up, the vaccine race, the oil "
           "glut appear in the graph weeks before the market event), while <b>shock events land at the news-flow "
           "floor</b> (+1 week for an invasion, a bank run, a strike &mdash; nothing in prior public news predicts "
           "them). The system's lead time reflects the information structure of the event itself. The two misses are "
           "coverage-honest: FTX and OpenAI are barely present in a US-equity news corpus. Discovered themes "
           "beyond the gold list include the 2019 cannabis boom (HEXO/Aurora/Tilray, born 2019-03-18), the EV/SPAC "
           "wave (Nikola/Nio, born 2020-06-22), and the stay-at-home complex (Zoom/Kroger/Virgin Galactic, born "
           "2020-04-06)."))
s.append(P("<b>Placebo control.</b> Applying the same detection criterion to <i>random</i> event weeks (100,000 "
           "Monte-Carlo draws over the per-theme chance rates, which run 6&ndash;36% because broad topics have "
           "several lifelines across seven years) yields an expected <b>~2/10 by chance; the observed 8/10 has "
           "p &asymp; 10<super>-4</super></b> in both configurations. The gold-list performance is not an artifact "
           "of theme abundance."))
s.append(P("<b>Statistical robustness.</b> Two further checks harden the detector. <b>Overdispersion:</b> weekly "
           "formation counts are strongly super-Poisson (negative-binomial dispersion &alpha; = 3.75, method of "
           "moments on training weeks only), so a NB-variance variant was run: it is the high-precision "
           "configuration (102 lifelines vs 401; 7/10 detected at median +0 weeks; chance rate 0.9/10, p &asymp; 0), "
           "with the Poisson-floor default as the high-recall configuration &mdash; the gold-list conclusion holds "
           "under both. <b>Window sensitivity:</b> every cell of a 3&times;3 grid (recent 2/4/8 weeks &times; "
           "baseline 13/26/52 weeks) detects 7&ndash;8/10 with placebo p &lt; 4&times;10<super>-4</super>; shorter "
           "windows detect slightly later, longer slightly earlier. Detection is not a window artifact."))
s.append(P("<b>Lifeline-level precision audit (supervisor-requested).</b> The placebo establishes week-level "
           "precision; a blind audit measures the discovery stream itself. Every EMERGING lifeline was packaged with "
           "its members, birth month and provenance headlines (top articles with opening text; no detector scores) "
           "and classified REAL / TEMPLATE / INCOHERENT by a locally hosted Qwen2.5-14B judge, validated by 11/11 "
           "known controls and 93% binary agreement with a 14-pack human-labelled subset. Results: <b>lenient 37.9% "
           "REAL</b> [Wilson 95% 32.5&ndash;43.7], strict 42.9%, NB 37.1% (with the <i>highest</i> template share, "
           "44% &mdash; event-level specificity and stream purity are distinct properties), and a new "
           "<b>template-filtered arm at 44.7%</b> [38.6&ndash;50.9]: removing template-fingerprinted <i>articles</i> "
           "(25.1% of corpus) from the detector substrate keeps every real discovery (109 vs 107) and the full "
           "gold-list result (8/10, &minus;1 wk, p = 5&times;10<super>-5</super>) while cutting template lifelines "
           "from 86 to 54. Structural cohesion alone cannot certify quality &mdash; boilerplate binds <i>denser</i> "
           "than real news &mdash; so the semantic (judged) layer is the primary instrument; post-hoc structural "
           "filters saturate near 50% precision."))

s.append(Paragraph("8.&nbsp;&nbsp;Limitations", H1))
s.append(B("<b>Extractor cutoff.</b> Hindsight is excluded for link prediction (above) but causal-impact relations "
           "carry measurable entity-dependence."))
s.append(B("<b>US-centricity.</b> Thin country/regulator relations (GPE 6.7% of edges)."))
s.append(B("<b>Fragmentation inflates novelty (quantified).</b> Surface-form variants (Apple vs Apple Inc) make some "
           "recurring relationships look novel. A crude resolution (corporate-suffix merge, 46k to 38k entities) cuts "
           "test novelty 72.4% to 68.4% and lifts recurrence MRR ~9% &mdash; so fragmentation is real but secondary, and "
           "forecasting-difficulty survives (a lower bound; full resolution would shift it further)."))
s.append(B("<b>Validation scope.</b> Link-prediction MRRs carry 95% paired-bootstrap CIs and RE-GCN is averaged over "
           "3 seeds; raw (unfiltered) ranking still deflates absolute MRRs relative to the filtered TKG convention, "
           "and Hits@K under the mean-rank tie convention is quantized for tie blocks straddling K (MRR unaffected). "
           "The query bootstrap treats test directions as independent &mdash; direction pairs and week clustering make "
           "the CIs somewhat narrow, which is why small gaps are additionally checked against training-seed "
           "replications (Section 5). Event recovery is scored systematically (18-event gold list, 94% burst-recall) "
           "rather than by anecdote, with the theme-level evaluation of Section 7 as the stricter, "
           "birth-window-anchored test."))
s.append(B("<b>Corpus duplication (measured).</b> Syndicated duplicate bodies (10.5% of 200k, 24.3% of 600k) inflate "
           "per-mention edge weights and core membership; all headline conclusions were re-verified on a deduplicated "
           "rebuild (Section 6). Exact-hash deduplication is a lower bound &mdash; near-duplicates are not caught."))
s.append(B("<b>Scale (now addressed, Section 6).</b> The 600k rebuild confirms denser sampling is the lever &mdash; it "
           "cut novelty 72% to 64% and lifted forecasting ~18%; residual novel-edge difficulty is the remaining ceiling."))

s.append(Paragraph("9.&nbsp;&nbsp;Next steps", H1))
s.append(B("Per-theme false-discovery audit: hand-verify a sample of detected EMERGING lifelines that match no "
           "gold event (the week-level placebo control of Section 7 is done; this is the lifeline-level complement)."))
s.append(B("Entity-resolution pass to quantify its lift on link prediction &mdash; the next density lever now that the "
           "600k rebuild (Section 6) is complete."))
s.append(B("Broaden the static-embedding baselines (e.g. RotatE / TuckER via a KGE library) to confirm ComplEx is "
           "representative, keeping the strict chronological PIT evaluation rather than a transductive split."))

s.append(Paragraph("10.&nbsp;&nbsp;Deliverables &amp; reproducibility", H1))
s.append(P("All code is fixed-seed and resumable. Substrate: <font face='Courier'>build_subset / extract / assemble / "
           "build_core</font>. Dynamics (<font face='Courier'>dynamics/</font>): <font face='Courier'>loader, eda, "
           "velocity, burst (Kleinberg + CUSUM), linkpred, complex_kge, regcn, text_embed, bootstrap_ci, "
           "event_validate, themes, themes_eval</font>; a <font face='Courier'>KG_CORE_PATH</font> env switch points "
           "the same code at any core (200k / 600k / 600k-deduplicated). The 200k and 600k full KGs, denoised and "
           "deduplicated cores, raw triplet stores, subsets, and entity embeddings are archived locally; the 600k "
           "results are consolidated in <font face='Courier'>results_600k/</font> and the methods + literature review "
           "in an accompanying document."))

doc = SimpleDocTemplate(OUT, pagesize=letter, topMargin=0.7 * inch, bottomMargin=0.7 * inch,
                        leftMargin=0.8 * inch, rightMargin=0.8 * inch, title="FNSPID-ICKG project report")
doc.build(s)
print("wrote", OUT)
