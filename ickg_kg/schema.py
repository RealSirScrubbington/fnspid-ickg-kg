"""FinDKG / ICKG ontology -- the exact 12 entity types and 15 relation verbs the
ICKG-v3.2 extractor is fine-tuned to emit, plus canonical id maps that match the
released FinDKG_dataset files so our output is drop-in compatible.

Verified at build time:
  * Entity categories + relation verb set: victorlxh/ICKG-v3.2 model-card prompt.
  * relation -> id ordering: FinDKG_dataset/FinDKG/relation2id.txt (verbatim).
  * entity type -> type_id: FinDKG_dataset/FinDKG/entity2id.txt (col 4).
"""
from __future__ import annotations

# Relation -> id, copied verbatim from FinDKG relation2id.txt (canonical order).
RELATION2ID: dict[str, int] = {
    "Relate_To": 0,
    "Control": 1,
    "Operate_In": 2,
    "Impact": 3,
    "Has": 4,
    "Negative_Impact_On": 5,
    "Raise": 6,
    "Is_Member_Of": 7,
    "Participates_In": 8,
    "Announce": 9,
    "Invests_In": 10,
    "Produce": 11,
    "Introduce": 12,
    "Decrease": 13,
    "Positive_Impact_On": 14,
}
RELATIONS: list[str] = list(RELATION2ID)
assert len(RELATIONS) == 15, "FinDKG defines exactly 15 relations"

# The 12 entity categories exactly as named in the ICKG prompt. type_id values
# follow FinDKG's entity2id.txt where directly observed (PERSON=0, GPE=1, COMP=2,
# EVENT=3, ORG/GOV=6, ORG/REG=7); the rest are assigned deterministically here
# and reconciled against the full FinDKG file in assemble.py (Stage 3).
ENTITY_TYPE2ID: dict[str, int] = {
    "PERSON": 0,
    "GPE": 1,
    "COMP": 2,
    "EVENT": 3,
    "PRODUCT": 4,
    "SECTOR": 5,
    "ORG/GOV": 6,
    "ORG/REG": 7,
    "ORG": 8,
    "ECON_INDICATOR": 9,
    "FIN_INSTRUMENT": 10,
    "CONCEPT": 11,
}
ENTITY_TYPES: list[str] = list(ENTITY_TYPE2ID)
assert len(ENTITY_TYPES) == 12, "FinDKG defines exactly 12 entity types"

# Tolerant aliases -> canonical, for normalising slightly-off model output.
_ENTITY_TYPE_ALIASES: dict[str, str] = {
    "ECON_IND": "ECON_INDICATOR",
    "ECON_INDICATORS": "ECON_INDICATOR",
    "ECONOMIC_INDICATOR": "ECON_INDICATOR",
    "FIN_INST": "FIN_INSTRUMENT",
    "FIN_INSTRUMENTS": "FIN_INSTRUMENT",
    "FINANCIAL_INSTRUMENT": "FIN_INSTRUMENT",
    "ORG/GOVERNMENT": "ORG/GOV",
    "GOV": "ORG/GOV",
    "ORG/REGULATOR": "ORG/REG",
    "REG": "ORG/REG",
    "COMPANY": "COMP",
    "COMPANIES": "COMP",
    "ORGANIZATION": "ORG",
    "ORGANISATION": "ORG",
    "PER": "PERSON",
    "LOCATION": "GPE",
    "LOC": "GPE",
    "PROD": "PRODUCT",
}

_RELATION_ALIASES: dict[str, str] = {
    "POSITIVE_IMPACT": "Positive_Impact_On",
    "NEGATIVE_IMPACT": "Negative_Impact_On",
    "POSITIVELY_IMPACT": "Positive_Impact_On",
    "NEGATIVELY_IMPACT": "Negative_Impact_On",
    "IMPACTS": "Impact",
    "RELATED_TO": "Relate_To",
    "RELATES_TO": "Relate_To",
    "MEMBER_OF": "Is_Member_Of",
    "PARTICIPATE_IN": "Participates_In",
    "INVEST_IN": "Invests_In",
    "OPERATES_IN": "Operate_In",
    "PRODUCES": "Produce",
    "INTRODUCES": "Introduce",
    "ANNOUNCES": "Announce",
    "CONTROLS": "Control",
    "RAISES": "Raise",
    "DECREASES": "Decrease",
}

_REL_CANON = {r.upper(): r for r in RELATIONS}
_ENT_CANON = {e.upper(): e for e in ENTITY_TYPES}


def normalize_entity_type(s: str) -> str | None:
    """Map a model-emitted entity type to its canonical form, or None if invalid."""
    if not s:
        return None
    key = s.strip().upper().replace(" ", "_")
    if key in _ENT_CANON:
        return _ENT_CANON[key]
    if key in _ENTITY_TYPE_ALIASES:
        return _ENTITY_TYPE_ALIASES[key]
    # tolerate ORG-GOV / ORG_GOV spellings
    key2 = key.replace("-", "/").replace("_GOV", "/GOV").replace("_REG", "/REG")
    return _ENT_CANON.get(key2)


def normalize_relation(s: str) -> str | None:
    """Map a model-emitted relation verb to its canonical form, or None if invalid."""
    if not s:
        return None
    key = s.strip().upper().replace(" ", "_")
    if key in _REL_CANON:
        return _REL_CANON[key]
    return _RELATION_ALIASES.get(key)
