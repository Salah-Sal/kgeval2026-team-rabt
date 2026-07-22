"""Typed entity markers for RE (design §4.1, variant B — Zhou & Chen punct).

Subject mention wrapped as `@ * type * mention @`, object as `# ^ type ^
mention #`; the classifier reads the encoder states at the two start markers,
so their character offsets in the marked text are returned for char_to_token
lookup.

Occurrence ambiguity (5.9% of triples): the non-overlapping occurrence pair
with minimal character distance wins, ties leftmost. Containment pairs — ~20%
of WojoodRelations has the object embedded in the subject mention (e.g.
located_in(نقابة العاملين في جامعة بيرزيت, بيرزيت)) — get nested insertion:
the inner segment goes inside the outer mention, keeping both markers in
context. Degraded cases never raise (blind test data must always encode);
they are labeled in `fallback` and counted by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Marked:
    text: str
    subj_start: int  # char offset of the subject's '@'
    obj_start: int  # char offset of the object's '#'
    fallback: str | None = None  # None, 'overlap', 'subject_missing', 'object_missing', 'both_missing'


def occurrences(sentence: str, mention: str) -> list[int]:
    out: list[int] = []
    if not mention:
        return out
    start = 0
    while (i := sentence.find(mention, start)) != -1:
        out.append(i)
        start = i + 1
    return out


def _subj_prefix(typ: str) -> str:
    return f"@ * {typ.lower()} * "


def _obj_prefix(typ: str) -> str:
    return f"# ^ {typ.lower()} ^ "


def _subj_segment(mention: str, typ: str) -> str:
    return _subj_prefix(typ) + mention + " @"


def _obj_segment(mention: str, typ: str) -> str:
    return _obj_prefix(typ) + mention + " #"


def choose_pair(
    sentence: str, subject: str, obj: str
) -> tuple[str, int, int] | None:
    """→ (kind, subj_start, obj_start), kind ∈ {'disjoint', 'contains'}.

    Disjoint pairs win by minimal mid-to-mid distance; if none exists, a
    containment pair (one mention's span inside the other's) is returned.
    None if either mention is absent or only partial crossings exist.
    """
    best_disjoint: tuple[int, int, int] | None = None  # (distance, si, oi)
    best_contains: tuple[int, int] | None = None  # (si, oi)
    for si in occurrences(sentence, subject):
        se = si + len(subject)
        for oi in occurrences(sentence, obj):
            oe = oi + len(obj)
            if se <= oi or oe <= si:
                mid_s = si + len(subject) // 2
                mid_o = oi + len(obj) // 2
                key = (abs(mid_s - mid_o), si, oi)
                if best_disjoint is None or key < best_disjoint:
                    best_disjoint = key
            elif (si <= oi and oe <= se) or (oi <= si and se <= oe):
                if best_contains is None or (si, oi) < best_contains:
                    best_contains = (si, oi)
    if best_disjoint is not None:
        return ("disjoint", best_disjoint[1], best_disjoint[2])
    if best_contains is not None:
        return ("contains", best_contains[0], best_contains[1])
    return None


def _insert_disjoint(
    sentence: str, si: int, subj_seg: str, oi: int, obj_seg: str,
    subject: str, obj: str,
) -> Marked:
    spans = sorted([(si, si + len(subject), subj_seg), (oi, oi + len(obj), obj_seg)])
    (a_start, a_end, a_seg), (b_start, b_end, b_seg) = spans
    text = (
        sentence[:a_start] + a_seg
        + sentence[a_end:b_start] + b_seg
        + sentence[b_end:]
    )
    a_pos = a_start
    b_pos = a_start + len(a_seg) + (b_start - a_end)
    if a_seg is subj_seg:
        return Marked(text, subj_start=a_pos, obj_start=b_pos)
    return Marked(text, subj_start=b_pos, obj_start=a_pos)


def _insert_nested(
    sentence: str, si: int, oi: int,
    subject: str, subj_type: str, obj: str, obj_type: str,
) -> Marked:
    subj_contains = si <= oi and oi + len(obj) <= si + len(subject)
    if subj_contains:
        outer_start, outer, outer_prefix = si, subject, _subj_prefix(subj_type)
        inner_start, inner, inner_seg = oi, obj, _obj_segment(obj, obj_type)
        outer_close = " @"
    else:
        outer_start, outer, outer_prefix = oi, obj, _obj_prefix(obj_type)
        inner_start, inner, inner_seg = si, subject, _subj_segment(subject, subj_type)
        outer_close = " #"
    rel = inner_start - outer_start
    marked_mention = outer[:rel] + inner_seg + outer[rel + len(inner):]
    text = (
        sentence[:outer_start]
        + outer_prefix + marked_mention + outer_close
        + sentence[outer_start + len(outer):]
    )
    inner_pos = outer_start + len(outer_prefix) + rel
    if subj_contains:
        return Marked(text, subj_start=outer_start, obj_start=inner_pos)
    return Marked(text, subj_start=inner_pos, obj_start=outer_start)


def insert_typed_markers(
    sentence: str, subject: str, subj_type: str, obj: str, obj_type: str
) -> Marked:
    subj_seg = _subj_segment(subject, subj_type)
    obj_seg = _obj_segment(obj, obj_type)

    pair = choose_pair(sentence, subject, obj)
    if pair is not None:
        kind, si, oi = pair
        if kind == "disjoint":
            return _insert_disjoint(sentence, si, subj_seg, oi, obj_seg, subject, obj)
        return _insert_nested(sentence, si, oi, subject, subj_type, obj, obj_type)

    # degraded paths: mark what is present in place, append what is not
    s_occ = occurrences(sentence, subject)
    o_occ = occurrences(sentence, obj)
    if s_occ and o_occ:  # partial crossing (0 cases in train; possible in blind test)
        si = s_occ[0]
        text = sentence[:si] + subj_seg + sentence[si + len(subject):] + " " + obj_seg
        return Marked(text, subj_start=si, obj_start=len(text) - len(obj_seg),
                      fallback="overlap")
    if s_occ:
        si = s_occ[0]
        text = sentence[:si] + subj_seg + sentence[si + len(subject):] + " " + obj_seg
        return Marked(text, subj_start=si, obj_start=len(text) - len(obj_seg),
                      fallback="object_missing")
    if o_occ:
        oi = o_occ[0]
        text = sentence[:oi] + obj_seg + sentence[oi + len(obj):] + " " + subj_seg
        return Marked(text, subj_start=len(text) - len(subj_seg), obj_start=oi,
                      fallback="subject_missing")
    text = sentence + " " + subj_seg + " " + obj_seg
    return Marked(text, subj_start=len(sentence) + 1,
                  obj_start=len(sentence) + 1 + len(subj_seg) + 1,
                  fallback="both_missing")
