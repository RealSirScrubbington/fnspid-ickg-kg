"""ICKG-v3.2 extractor: load Mistral-7B-Instruct-v0.2 + the ICKG LoRA adapter,
run the *verbatim* fine-tuning prompt, and parse the emitted triplets.

The prompt below is copied EXACTLY from the victorlxh/ICKG-v3.2 model card
("Prompt Template" -> Generative Knowledge Graph Construction). Do not paraphrase
it: ICKG was fine-tuned on this exact wording and deviating degrades extraction.
The article text replaces the trailing `<input_text>` placeholder, and the whole
thing is wrapped as a single Mistral `[INST] ... [/INST]` user turn (the base
model's chat template).
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from .schema import normalize_entity_type, normalize_relation

# ---------------------------------------------------------------------------
# VERBATIM ICKG-v3.2 KG-construction prompt (model card). `<input_text>` is the
# only placeholder we substitute.
# ---------------------------------------------------------------------------
ICKG_PROMPT = """From the provided document labeled as INPUT_TEXT, your task is to extract structured information from it in the form of triplet for constructing a knowledge graph. Each tuple should be in the form of ('h', 'type',  'r', 'o', 'type'), where 'h' stands for the head entity, 'r' for the relationship, and 'o' for the tail entity. The 'type' denotes the category of the corresponding entity. Do NOT include redundant triplets, NOT include triplets with relationship that occurs in the past.

Note that the entities should not be generic, numerical or temporal (like dates or percentages).  Entities must be classified into the following categories:
ORG: Organizations other than government or regulatory bodies
ORG/GOV: Government bodies (e.g., "United States Government")
ORG/REG: Regulatory bodies (e.g., "Federal Reserve")
PERSON: Individuals (e.g., "Elon Musk")
GPE: Geopolitical entities such as countries, cities, etc. (e.g., "Germany")
COMP: Companies (e.g., "Google")
PRODUCT: Products or services (e.g., "iPhone")
EVENT: Specific and Material Events (e.g., "Olympic Games", "Covid-19")
SECTOR: Company sectors or industries (e.g., "Technology sector")
ECON_INDICATOR: Economic indicators (e.g., "Inflation rate"), numerical value like "10%" is not a ECON_INDICATOR;
FIN_INSTRUMENT: Financial and market instruments (e.g., "Stocks", "Global Markets")
CONCEPT: Abstract ideas or notions or themes (e.g., "Inflation", "AI", "Climate Change")

The relationships 'r' between these entities must be represented by one of the following relation verbs set: Has, Announce, Operate_In, Introduce, Produce, Control, Participates_In, Impact, Positive_Impact_On, Negative_Impact_On, Relate_To, Is_Member_Of, Invests_In, Raise, Decrease.

Remember to conduct entity disambiguation, consolidating different phrases or acronyms that refer to the same entity (for instance,  "UK Central Bank", "BOE" and "Bank of England" should be unified as "Bank of England"). Simplify each entity of the triplet to be less than four words.

Your output should strictly be in a list format of triplets in the JSON list format of ('h', 'type', 'r', 'o', 'type'), where the relationship 'r' must be in the given relation verbs set above. Only output the list.
===========================================================
As an Example, consider the following news excerpt:
'Apple Inc. is set to introduce the new iPhone 14 in the technology sector this month. The product's release is likely to positively impact Apple's stock value.'

From this text, your output should be:
[('Apple Inc.', 'COMP', 'Introduce', 'iPhone 14', 'PRODUCT'),
 ('Apple Inc.', 'COMP', 'Operate_In', 'Technology Sector', 'SECTOR'),
 ('iPhone 14', 'PRODUCT', 'Positive_Impact_On', 'Apple's Stock Value', 'FIN_INSTRUMENT')]

INPUT_TEXT:
<input_text>"""

PROMPT_VERSION = "ickg-v3.2-modelcard-verbatim"


# ---------------------------------------------------------------------------
# Triplet parsing
# ---------------------------------------------------------------------------
def _iter_tuples(raw: str):
    """Yield the interior of each top-level (...) tuple, tolerating *balanced*
    parentheses inside entity names (e.g. "AlphaDEX ETF (FXR)", "U.S. Steel (X)").
    A naive `\\(...\\)` regex drops every tuple whose entity contains a paren --
    which is most financial-news tuples (tickers in parens). We instead track
    paren depth: depth 0->1 opens a tuple, the matching )->0 closes it, and inner
    parens are kept as content."""
    depth = 0
    buf: list[str] = []
    for ch in raw:
        if ch == "(":
            if depth > 0:
                buf.append(ch)
            else:
                buf = []
            depth += 1
        elif ch == ")":
            if depth > 1:
                buf.append(ch)
                depth -= 1
            elif depth == 1:
                depth = 0
                yield "".join(buf)
            # depth == 0: stray close, ignore
        elif depth > 0:
            buf.append(ch)


def _split_fields(inner: str) -> list[str]:
    """Split one tuple's interior into fields, robust to apostrophes inside
    entities (e.g. "Apple's Stock Value"). We split only on a quote-comma-quote
    boundary, then strip the outer quotes -- inner apostrophes survive."""
    inner = inner.strip()
    for q in ("'", '"'):
        if inner.startswith(q):
            parts = re.split(rf"{q}\s*,\s*{q}", inner)
            if len(parts) == 5:
                parts[0] = parts[0][len(q):] if parts[0].startswith(q) else parts[0]
                parts[-1] = parts[-1][:-len(q)] if parts[-1].endswith(q) else parts[-1]
                return [p.strip() for p in parts]
    # last resort: naive comma split (entities with commas will be rejected upstream)
    return [p.strip().strip("'\"").strip() for p in inner.split(",")]


@dataclass
class Triplet:
    h: str
    h_type: str
    r: str
    o: str
    o_type: str
    valid: bool          # types + relation all map to the FinDKG ontology

    def as_row(self) -> dict:
        return {"h": self.h, "h_type": self.h_type, "r": self.r,
                "o": self.o, "o_type": self.o_type, "valid": self.valid}


# ICKG-v4.2 (Qwen2.5-14B) inconsistently emits triplets either as Python paren
# tuples  [('h','type','r','o','type'), ...]  OR as a JSON array-of-arrays
# (often inside a ```json fence, multi-line):  [ ["h","type","r","o","type"], ...].
# This double-quoted square-bracket quintuple matcher handles the JSON variant
# (newline-tolerant, escape-aware, and robust to truncation -- an incomplete
# trailing array simply doesn't match).
_JSON_QUINT = re.compile(
    r'\[\s*"((?:[^"\\]|\\.)*)"\s*,\s*"((?:[^"\\]|\\.)*)"\s*,\s*'
    r'"((?:[^"\\]|\\.)*)"\s*,\s*"((?:[^"\\]|\\.)*)"\s*,\s*"((?:[^"\\]|\\.)*)"\s*\]')

# A "numeric/temporal" surface form the prompt forbids as an entity (e.g. 18.69%,
# 45.72, $125.00, 130) -- flagged invalid so it never enters the KG.
_NUMERIC = re.compile(r'^[\s$£€¥]*[-+]?\d[\d.,]*\s*%?\s*$')


def _unescape(s: str) -> str:
    return s.replace('\\"', '"').replace("\\n", " ").replace("\\\\", "\\").strip()


def _make_triplet(h: str, ht: str, r: str, o: str, ot: str) -> "Triplet | None":
    h, o = h.strip(), o.strip()
    if not h or not o:
        return None
    nht, nr, not_ = normalize_entity_type(ht), normalize_relation(r), normalize_entity_type(ot)
    numeric = bool(_NUMERIC.match(h)) or bool(_NUMERIC.match(o))
    valid = nht is not None and nr is not None and not_ is not None and not numeric
    return Triplet(h=h, h_type=nht or ht, r=nr or r, o=o, o_type=not_ or ot, valid=valid)


def parse_triplets(raw: str) -> list[Triplet]:
    out: list[Triplet] = []
    # Format A: JSON arrays-of-arrays (double-quoted, square brackets)
    for m in _JSON_QUINT.finditer(raw):
        t = _make_triplet(*(_unescape(g) for g in m.groups()))
        if t is not None:
            out.append(t)
    if out:
        return out
    # Format B: Python paren-tuples (single-quoted, parentheses)
    for inner in _iter_tuples(raw):
        fields = _split_fields(inner)
        if len(fields) == 5:
            t = _make_triplet(*fields)
            if t is not None:
                out.append(t)
    return out


@dataclass
class ExtractionResult:
    article_id: str
    raw: str
    triplets: list[Triplet]
    latency_s: float
    n_parsed: int = field(init=False)
    n_valid: int = field(init=False)
    malformed: bool = field(init=False)   # produced text but zero parseable tuples

    def __post_init__(self):
        self.n_parsed = len(self.triplets)
        self.n_valid = sum(t.valid for t in self.triplets)
        self.malformed = self.n_parsed == 0


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class IckgExtractor:
    def __init__(self, cfg):
        import torch
        import transformers
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel

        self.cfg = cfg
        self._torch = torch
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.adapter_repo)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # left-pad for decoder-only batched generation
        self.tokenizer.padding_side = "left"

        quant = None
        dtype = torch.float16
        if cfg.precision == "4bit":
            quant = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        elif cfg.precision == "8bit":
            quant = BitsAndBytesConfig(load_in_8bit=True)
        elif cfg.precision == "bf16":
            dtype = torch.bfloat16

        # transformers 5.x renamed `torch_dtype` -> `dtype`
        dtype_kw = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
        load_kwargs = {dtype_kw: dtype, "device_map": "auto"}
        if quant is not None:
            load_kwargs["quantization_config"] = quant
        base = AutoModelForCausalLM.from_pretrained(cfg.base_model, **load_kwargs)
        self.model = PeftModel.from_pretrained(base, cfg.adapter_repo)
        self.model.eval()
        self.device = self.model.device

        # token budget for the article = cap - (everything else in the prompt)
        base_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": ICKG_PROMPT.replace("<input_text>", "")}],
            tokenize=True)
        self._base_tokens = len(base_ids)
        self.load_s = time.time() - t0

    def _truncate_article(self, article: str) -> str:
        article = article[: self.cfg.max_body_chars]
        budget = max(64, self.cfg.max_input_tokens - self._base_tokens - 8)
        ids = self.tokenizer(article, add_special_tokens=False)["input_ids"]
        if len(ids) > budget:
            article = self.tokenizer.decode(ids[:budget])
        return article

    def build_prompt_text(self, article: str) -> str:
        content = ICKG_PROMPT.replace("<input_text>", self._truncate_article(article))
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False)

    def extract(self, article_id: str, article: str) -> ExtractionResult:
        torch = self._torch
        content = ICKG_PROMPT.replace("<input_text>", self._truncate_article(article))
        enc = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            return_tensors="pt", return_dict=True).to(self.device)
        input_len = enc["input_ids"].shape[1]
        t0 = time.time()
        with torch.no_grad():
            gen = self.model.generate(
                **enc, max_new_tokens=self.cfg.max_new_tokens,
                do_sample=self.cfg.do_sample, num_beams=1,
                pad_token_id=self.tokenizer.eos_token_id,
                stop_strings="]", tokenizer=self.tokenizer)
        new = gen[0, input_len:]
        raw = self.tokenizer.decode(new, skip_special_tokens=True)
        return ExtractionResult(article_id, raw, parse_triplets(raw), time.time() - t0)

    def extract_batch(self, items: list[tuple[str, str]]) -> list[ExtractionResult]:
        """Batched extraction (left-padded). Portable across GPUs; the per-item
        latency reported is the batch wall-time divided by batch size, so
        throughput accounting stays honest."""
        if len(items) == 1:
            return [self.extract(items[0][0], items[0][1])]
        torch = self._torch
        convs = [[{"role": "user", "content":
                   ICKG_PROMPT.replace("<input_text>", self._truncate_article(text))}]
                 for _id, text in items]
        enc = self.tokenizer.apply_chat_template(
            convs, return_tensors="pt", return_dict=True, padding=True).to(self.device)
        input_len = enc["input_ids"].shape[1]
        t0 = time.time()
        try:
            with torch.no_grad():
                gen = self.model.generate(
                    **enc, max_new_tokens=self.cfg.max_new_tokens,
                    do_sample=self.cfg.do_sample, num_beams=1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    stop_strings="]", tokenizer=self.tokenizer)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" not in str(e).lower() or len(items) == 1:
                raise
            # transient VRAM spike (shared GPU): clear and split the batch
            del enc
            torch.cuda.empty_cache()
            mid = len(items) // 2
            return self.extract_batch(items[:mid]) + self.extract_batch(items[mid:])
        per = (time.time() - t0) / len(items)
        out = []
        for i, (_id, _text) in enumerate(items):
            raw = self.tokenizer.decode(gen[i, input_len:], skip_special_tokens=True)
            out.append(ExtractionResult(_id, raw, parse_triplets(raw), per))
        return out
