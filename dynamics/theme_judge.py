"""Lifeline-level precision audit, part 2: LLM judge over the audit packs (runs on a GPU host).

Each EMERGING lifeline's pack (members + birth month + source headlines; NO detector scores) is
classified against the frozen rubric of dynamics/theme_audit.py:
  REAL / TEMPLATE / INCOHERENT
Greedy decoding, one pass, fixed prompt; the judge is validated against a human-labelled subset
and known-real / known-noise controls before its aggregate is quoted.

Run (prowl): python -m dynamics.theme_judge --packs audit_packs.jsonl --out theme_judge_labels.csv
"""
import argparse
import json
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RUBRIC = """You are auditing the output of a financial news theme-detection system. A "theme" is
a group of entities that started forming new relationships unusually fast in the same weeks,
plus sample headlines from the news articles that drove it. Classify the theme:

REAL       - the members form a coherent, externally documentable financial storyline (a market
             event, sector wave, or macro development), and the headlines read like genuine news
             coverage of that storyline.
TEMPLATE   - the members and/or headlines are dominated by stock-screener or boilerplate
             constructs: financial-metric names (e.g. "Earnings ESP", "200 day moving average"),
             ranking-service products, or formulaic mail-merge headlines repeated across tickers.
INCOHERENT - no discernible common storyline; the entities' co-occurrence looks incidental.

Answer with strict JSON only: {"label": "REAL"|"TEMPLATE"|"INCOHERENT", "rationale": "<one sentence>"}"""


PAIR_RUBRIC = """You are auditing one detected theme from a financial news theme-detection system.
For each MEMBER PAIR below you are given the actual article (headline + opening text) that most
strongly bound the two entities together. Decide for each pair:

SUBSTANTIVE   - the article genuinely relates the two entities within one storyline (they play
                roles in the same event, sector development, or macro narrative).
COINCIDENTAL  - the two entities merely co-appear (roundup/listicle/screener article, ticker
                lists, formulaic content); the text does not actually relate them.

Answer with strict JSON only:
{"pairs": [{"pair": ["A","B"], "verdict": "SUBSTANTIVE"|"COINCIDENTAL"}, ...]}"""


def render(p):
    mem = ", ".join(m["name"] for m in p["members"])
    lines = []
    for h in p["headlines"]:
        lines.append(f'- [{h["date"]}] {h["title"]}')
        if h.get("body_lede"):
            lines.append(f'    opening: {h["body_lede"]}')
    heads = "\n".join(lines) or "- (none retrieved)"
    return (f"Theme first detected: {p['birth_month']}\n"
            f"Member entities ({p['n_members_total']} total, top shown): {mem}\n"
            f"Sample source articles:\n{heads}")


def render_pairs(p):
    lines = []
    for pr in p["pairs"]:
        lines.append(f'PAIR: {pr["pair"][0]}  <->  {pr["pair"][1]}  '
                     f'({pr["joint_articles"]} joint articles)')
        lines.append(f'  binding article [{pr["date"]}]: {pr["sample_title"]}')
        if pr.get("body_lede"):
            lines.append(f'  opening: {pr["body_lede"]}')
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packs", default="audit_packs.jsonl")
    ap.add_argument("--out", default="theme_judge_labels.csv")
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct-AWQ")
    ap.add_argument("--mode", choices=["themes", "pairs"], default="themes",
                    help="themes: REAL/TEMPLATE/INCOHERENT per lifeline. "
                         "pairs: SUBSTANTIVE/COINCIDENTAL per binding member-pair")
    ap.add_argument("--merge-system", action="store_true",
                    help="for chat templates without a system role (Mistral): "
                         "prepend the rubric to the user turn instead")
    a = ap.parse_args()

    packs = [json.loads(l) for l in open(a.packs, encoding="utf-8")]
    print(f"{len(packs)} packs, mode={a.mode}")

    from vllm import LLM, SamplingParams
    llm = LLM(model=a.model, max_model_len=8192, gpu_memory_utilization=0.90)
    sp = SamplingParams(temperature=0.0, max_tokens=400 if a.mode == "pairs" else 120)
    sysmsg = PAIR_RUBRIC if a.mode == "pairs" else RUBRIC
    rend = render_pairs if a.mode == "pairs" else render
    if a.merge_system:
        msgs = [[{"role": "user", "content": sysmsg + "\n\n" + rend(p)}] for p in packs]
    else:
        msgs = [[{"role": "system", "content": sysmsg}, {"role": "user", "content": rend(p)}]
                for p in packs]
    outs = llm.chat(msgs, sp)

    from collections import Counter
    with open(a.out, "w", encoding="utf-8") as f:
        if a.mode == "themes":
            f.write("theme,label,rationale\n")
            counts = Counter()
            for p, o in zip(packs, outs):
                txt = o.outputs[0].text
                m = re.search(r'"label"\s*:\s*"(REAL|TEMPLATE|INCOHERENT)"', txt)
                r = re.search(r'"rationale"\s*:\s*"([^"]*)"', txt)
                lab = m.group(1) if m else "PARSE_FAIL"
                rat = (r.group(1) if r else txt.replace("\n", " "))[:300]
                f.write(f'{p["theme"]},{lab},"{rat.replace(chr(34), chr(39))}"\n')
                counts[lab] += 1
        else:
            f.write("theme,pair_a,pair_b,joint_articles,verdict\n")
            counts = Counter()
            for p, o in zip(packs, outs):
                txt = o.outputs[0].text
                verdicts = re.findall(r'"pair"\s*:\s*\[\s*"([^"]*)"\s*,\s*"([^"]*)"\s*\]\s*,\s*'
                                      r'"verdict"\s*:\s*"(SUBSTANTIVE|COINCIDENTAL)"', txt)
                vmap = {(x, y): v for x, y, v in verdicts}
                for pr in p["pairs"]:
                    v = vmap.get(tuple(pr["pair"]), "PARSE_FAIL")
                    f.write(f'{p["theme"]},"{pr["pair"][0]}","{pr["pair"][1]}",'
                            f'{pr["joint_articles"]},{v}\n')
                    counts[v] += 1
    print(counts)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
