"""Canonical AdaptNER column contract.

The Codabench scorer reads the 21 tag columns positionally in this fixed
(alphabetical) order — a wrong order scores silently wrong, so every module
that touches the 22-field format imports the constants from here and nowhere
else.
"""

ENTITY_TYPES: tuple[str, ...] = (
    "CARDINAL",
    "CURR",
    "DATE",
    "EVENT",
    "FAC",
    "GPE",
    "LANGUAGE",
    "LAW",
    "LOC",
    "MONEY",
    "NORP",
    "OCC",
    "ORDINAL",
    "ORG",
    "PERCENT",
    "PERS",
    "PRODUCT",
    "QUANTITY",
    "TIME",
    "UNIT",
    "WEBSITE",
)

NUM_TYPES = len(ENTITY_TYPES)
TYPE_INDEX: dict[str, int] = {t: i for i, t in enumerate(ENTITY_TYPES)}

assert NUM_TYPES == 21
assert list(ENTITY_TYPES) == sorted(ENTITY_TYPES), "contract order is alphabetical"

# The 10 Konooz domains; submissions concatenate domain files in sorted() order.
# Validators derive the actual order from the released test directory listing and
# cross-check it against this fallback.
KONOOZ_DOMAINS: tuple[str, ...] = (
    "Agriculture",
    "Art",
    "Economics",
    "Finance",
    "Health",
    "History",
    "Law",
    "Politics",
    "Science",
    "Sport",
)


def valid_column_tags(type_name: str) -> frozenset[str]:
    """The only strings admissible in the column of `type_name`."""
    return frozenset(("O", f"B-{type_name}", f"I-{type_name}"))
