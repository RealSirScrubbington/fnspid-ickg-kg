"""Central configuration + CONFIG FLAGS.

Every knob the brief asks to be configurable lives here and is exposed on the
command line by `add_common_args` / `Config.from_args`:

  * model-load precision   (--precision 4bit|8bit|fp16|bf16)
  * batch size             (--batch-size)
  * subset size            (--n)
  * random seed            (--seed)
  * input / output tokens  (--max-input-tokens, --max-new-tokens)
  * decoding               (--do-sample / greedy by default)
  * time-bucket resolution (--time-resolution week|day|month)  [used in Stage 3]
  * edge weighting         (--edge-weight / --no-edge-weight)   [used in Stage 3]
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

# --- Repos / data locations (verified against the live HF Hub at build time) ---
ICKG_ADAPTER_REPO = "victorlxh/ICKG-v3.2"               # LoRA adapter (13 MB)
ICKG_BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"  # base (ungated)
FNSPID_REPO = "Zihan1004/FNSPID"
# The canonical full news corpus: ~23.2 GB, ~15.7M rows, 1999-2023, grouped by
# symbol & newest-first (=> must be offset-sampled, never read head-first).
FNSPID_NEWS_PATH = "Stock_news/nasdaq_exteral_data.csv"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_DIR = DATA_DIR / "sample"
TRIPLET_DIR = DATA_DIR / "triplets"
KG_DIR = DATA_DIR / "kg"
REPORT_DIR = PROJECT_ROOT / "reports"


@dataclass
class Config:
    # sampling
    n: int = 500
    seed: int = 42
    fnspid_repo: str = FNSPID_REPO
    news_path: str = FNSPID_NEWS_PATH
    min_body_chars: int = 200          # cheap pre-filter: drop near-empty rows
    min_body_words: int = 200          # quality floor: keep articles >= N words
    max_body_chars: int = 12000        # hard cap before tokenisation

    # model / extraction
    base_model: str = ICKG_BASE_MODEL
    adapter_repo: str = ICKG_ADAPTER_REPO
    precision: str = "4bit"            # 4bit|8bit|fp16|bf16
    batch_size: int = 1
    max_input_tokens: int = 1536
    max_new_tokens: int = 512
    do_sample: bool = False            # greedy by default (reproducible)

    # assembly (Stage 3) -- carried here so all flags live in one place
    time_resolution: str = "week"      # week|day|month
    edge_weight: bool = True

    # io
    data_dir: Path = field(default=DATA_DIR)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Config":
        kwargs = {}
        for f in cls.__dataclass_fields__:
            if hasattr(args, f) and getattr(args, f) is not None:
                kwargs[f] = getattr(args, f)
        return cls(**kwargs)


def add_common_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    g = p.add_argument_group("config flags")
    g.add_argument("--n", type=int, help="number of articles to sample")
    g.add_argument("--seed", type=int, help="random seed (reproducible sampling)")
    g.add_argument("--precision", choices=["4bit", "8bit", "fp16", "bf16"],
                   help="model load precision")
    g.add_argument("--batch-size", dest="batch_size", type=int)
    g.add_argument("--max-input-tokens", dest="max_input_tokens", type=int)
    g.add_argument("--max-new-tokens", dest="max_new_tokens", type=int)
    g.add_argument("--do-sample", dest="do_sample", action="store_true", default=None,
                   help="enable sampling (default: greedy)")
    g.add_argument("--time-resolution", dest="time_resolution",
                   choices=["week", "day", "month"])
    g.add_argument("--news-path", dest="news_path",
                   help="path within the FNSPID repo to the news CSV")
    g.add_argument("--min-body-words", dest="min_body_words", type=int,
                   help="quality floor: only keep articles with >= N words")
    return p


def ensure_dirs() -> None:
    for d in (DATA_DIR, SAMPLE_DIR, TRIPLET_DIR, KG_DIR, REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)
