"""Soft domain/range constraints for RE inference (design §4.3).

Admissible (subject_type, object_type) pairs per label = the A02 Table 10
spec (verbatim from the design doc) ∪ pairs observed in train.jsonl — A04
documents real annotations deviating from the spec, so observation always
extends, never narrows. At inference a fixed penalty is SUBTRACTED from
inadmissible labels' logits (soft, never a hard mask); pairs involving UNK
types are fully permissive. Penalty magnitude is tuned on the local dev split
(design suggests 2–4 logits; 0 disables).

Labels absent from the spec (e.g. Affiliation.employee_of) have an empty spec
side and rely on observed pairs alone.
"""

from __future__ import annotations

from .re_typing import UNK_TYPE

# label → (admissible subject types, admissible object types); the admissible
# spec set is the cartesian product of the two.
SPEC: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    label: (frozenset(subj), frozenset(obj))
    for label, subj, obj in [
        ("Family.has_parent", {"PERS"}, {"PERS"}),
        ("Family.has_spouse", {"PERS"}, {"PERS"}),
        ("Family.has_sibling", {"PERS"}, {"PERS"}),
        ("Family.has_relative", {"PERS"}, {"PERS"}),
        ("Personal.birth_date", {"PERS"}, {"DATE"}),
        ("Personal.death_date", {"PERS"}, {"DATE"}),
        ("Personal.birth_place", {"PERS"}, {"GPE", "LOC"}),
        ("Personal.has_occupation", {"PERS"}, {"OCC"}),
        ("Business.has_conflict_with", {"ORG", "NORP", "GPE", "PERS"}, {"ORG", "NORP", "GPE", "PERS"}),
        ("Business.has_competitor", {"PERS", "ORG"}, {"PERS", "ORG"}),
        ("Business.has_partner_with", {"ORG"}, {"ORG"}),
        ("Administration.manager_of", {"PERS"}, {"ORG", "FAC"}),
        ("Administration.president_of", {"PERS"}, {"ORG", "GPE"}),
        ("Administration.leader_of", {"PERS"}, {"ORG"}),
        ("PartOf.geopolitical_division", {"GPE", "LOC"}, {"GPE", "LOC"}),
        ("PartOf.subsidiary", {"ORG"}, {"ORG"}),
        ("Affiliation.member_of", {"PERS", "GPE"}, {"ORG", "NORP"}),
        ("Affiliation.student_at", {"PERS"}, {"ORG"}),
        ("Affiliation.owner_of", {"PERS"}, {"ORG", "FAC"}),
        ("Productivity.inventor_of", {"PERS"}, {"PRODUCT"}),
        ("Productivity.manufacturer_of", {"ORG"}, {"PRODUCT"}),
        ("Productivity.builder_of", {"PERS", "NORP", "ORG"}, {"FAC", "ORG"}),
        ("Productivity.founder_of", {"PERS"}, {"ORG"}),
        ("Location.lives_in", {"PERS", "NORP"}, {"GPE", "LOC"}),
        ("Location.located_in", {"FAC", "ORG"}, {"GPE", "LOC"}),
        ("Location.headquartered_in", {"ORG"}, {"LOC", "GPE"}),
        ("Location.has_border_with", {"LOC", "GPE"}, {"LOC", "GPE"}),
        ("Location.nearby", {"GPE", "LOC", "FAC"}, {"GPE", "LOC", "FAC"}),
        ("Organization.has_propoerty", {"ORG"}, {"PRODUCT"}),
        ("Organization.branch_count", {"ORG"}, {"CARDINAL"}),
        ("Organization.employs", {"ORG"}, {"CARDINAL"}),
        ("GPE.has_population", {"GPE"}, {"CARDINAL"}),
        ("Organization.has_revenue", {"ORG"}, {"MONEY"}),
        ("Organization.found_on", {"ORG"}, {"DATE", "TIME"}),
        ("Organization.has_alternate_name", {"ORG", "FAC"}, {"ORG", "FAC"}),
        ("GPE.has_area", {"GPE", "LOC"}, {"QUANTITY"}),
        ("GPE.official_language", {"GPE", "LOC"}, {"LANGUAGE"}),
        ("GPE.has_currency", {"GPE", "LOC"}, {"CURR"}),
        ("GPE.capital_of", {"GPE"}, {"GPE"}),
    ]
}


def observed_pairs(
    records: list[dict], types: list[tuple[str, str]]
) -> dict[str, set[tuple[str, str]]]:
    """label → (subj_type, obj_type) pairs seen in training data (UNK excluded)."""
    out: dict[str, set[tuple[str, str]]] = {}
    for rec, (s_type, o_type) in zip(records, types, strict=True):
        if UNK_TYPE in (s_type, o_type):
            continue
        out.setdefault(rec["relation"], set()).add((s_type, o_type))
    return out


class ConstraintTable:
    """Admissibility oracle: spec product ∪ observed pairs, permissive on UNK."""

    def __init__(self, observed: dict[str, set[tuple[str, str]]]):
        self._admissible: dict[str, set[tuple[str, str]]] = {}
        for label, pairs in observed.items():
            self._admissible.setdefault(label, set()).update(pairs)
        for label, (subj_types, obj_types) in SPEC.items():
            self._admissible.setdefault(label, set()).update(
                (s, o) for s in subj_types for o in obj_types
            )

    def admissible(self, label: str, subj_type: str, obj_type: str) -> bool:
        if UNK_TYPE in (subj_type, obj_type):
            return True
        pairs = self._admissible.get(label)
        if not pairs:  # label with no spec row and never observed: stay permissive
            return True
        return (subj_type, obj_type) in pairs

    def penalty_mask(
        self, label_vocab: list[str], subj_type: str, obj_type: str
    ) -> list[bool]:
        """True where the label is INadmissible for this type pair."""
        return [
            not self.admissible(label, subj_type, obj_type) for label in label_vocab
        ]
