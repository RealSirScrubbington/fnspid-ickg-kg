"""Templated financial-media boilerplate entities (Zacks, Motley Fool, Stock Options Channel,
YieldBoost, Validea, ...). These are attribution / product noise, not economic actors. Because
the FNSPID Publisher field is empty for 95.7% of articles, they can only be caught by surface
form. Filtering is opt-in (`load_core(drop_noise=True)`) and reported, so it stays reversible
and documented for the thesis.
"""
import re

NOISE_PATTERN = re.compile(
    r"\b(zacks|yieldboost|stock\s*options\s*channel|motley\s*fool|validea|simply\s*wall|"
    r"insider\s*monkey|tipranks|smarteranalyst|kiplinger|benzinga|seeking\s*alpha|"
    r"investorplace|marketbeat|gurufocus)\b", re.I)


def noise_entity_ids(id2name: dict) -> set:
    """Ids whose surface form matches a known templated-media brand/product."""
    return {i for i, n in id2name.items() if NOISE_PATTERN.search(str(n))}
