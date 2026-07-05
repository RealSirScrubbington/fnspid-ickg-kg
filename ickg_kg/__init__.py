"""ICKG -> FinDKG-format time-indexed knowledge-graph build.

A bounded, point-in-time-aware pipeline that runs the open-source ICKG-v3.2
extractor (Mistral-7B + LoRA) over a random subset of the FNSPID financial-news
corpus and emits a temporal KG in the FinDKG quadruple format.

Stages (see the project brief / README):
  0  setup + verification        (done in scripts / notebooks)
  1  trial.py            -- go/no-go gate over ~500 random articles
  2  build.py            -- bounded checkpointed extraction
  3  assemble.py         -- triplets -> FinDKG-format KG
  4-5 pit_noise_checks.py -- point-in-time + noise diagnostics
"""

__version__ = "0.1.0"
