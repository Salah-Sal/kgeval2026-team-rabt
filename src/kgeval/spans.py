"""Per-column BIO decoding with IOB repair, and span-set extraction.

Spans are (start, end_exclusive, type) over token positions within one
sentence. An `I-x` with no open span of type x (after O or at sentence start)
starts a new span and is counted as a repair — the standard IOB2 repair, also
applied to model output before submission writing.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .columns import ENTITY_TYPES, TYPE_INDEX

Span = tuple[int, int, str]


@dataclass
class DecodeResult:
    spans: set[Span]
    n_repairs: int = 0


def decode_bio_column(tags: list[str], type_name: str) -> DecodeResult:
    """Decode one type column of one sentence."""
    spans: set[Span] = set()
    repairs = 0
    start: int | None = None
    for i, tag in enumerate(tags):
        if tag == "O":
            if start is not None:
                spans.add((start, i, type_name))
                start = None
        elif tag == f"B-{type_name}":
            if start is not None:
                spans.add((start, i, type_name))
            start = i
        elif tag == f"I-{type_name}":
            if start is None:  # orphan I- → repaired to a span start
                start = i
                repairs += 1
        else:
            raise ValueError(f"tag {tag!r} not admissible in column {type_name}")
    if start is not None:
        spans.add((start, len(tags), type_name))
    return DecodeResult(spans=spans, n_repairs=repairs)


def spans_from_rows(tag_rows: list[list[str]]) -> DecodeResult:
    """All typed spans of one sentence given per-token 21-tag rows."""
    all_spans: set[Span] = set()
    repairs = 0
    for ci, type_name in enumerate(ENTITY_TYPES):
        column = [row[ci] for row in tag_rows]
        res = decode_bio_column(column, type_name)
        all_spans |= res.spans
        repairs += res.n_repairs
    return DecodeResult(spans=all_spans, n_repairs=repairs)


def encode_spans_to_rows(spans: set[Span] | list[Span], n_tokens: int) -> list[list[str]]:
    """Per-token 21-tag rows from a span set (inverse of spans_from_rows).

    Same-type spans must be non-overlapping — a per-type BIO column cannot
    encode same-type overlap; resolve conflicts first (see union_tag_rows).
    Adjacent same-type spans round-trip: the second one starts with B-.
    """
    rows = [["O"] * len(ENTITY_TYPES) for _ in range(n_tokens)]
    by_type: dict[str, list[Span]] = {}
    for span in spans:
        start, end, typ = span
        if not (0 <= start < end <= n_tokens):
            raise ValueError(f"span {span} out of bounds for {n_tokens} tokens")
        by_type.setdefault(typ, []).append(span)
    for typ, tspans in by_type.items():
        ci = TYPE_INDEX[typ]
        prev_end = -1
        for start, end, _ in sorted(tspans):
            if start < prev_end:
                raise ValueError(f"overlapping {typ} spans cannot share a BIO column")
            rows[start][ci] = f"B-{typ}"
            for i in range(start + 1, end):
                rows[i][ci] = f"I-{typ}"
            prev_end = end
    return rows


@dataclass
class EnsembleResult:
    rows: list[list[str]]
    spans: set[Span]
    n_dropped: int  # candidates discarded by same-type overlap resolution


def union_tag_rows(
    rows_per_model: list[list[list[str]]], min_votes: int = 1
) -> EnsembleResult:
    """Span-level vote ensemble of one sentence tagged by several models.

    Decodes each model's spans (IOB repair applied), keeps spans reaching
    min_votes (1 = pure union — the recall lever), then resolves same-type
    overlaps greedily by (votes desc, length desc, start asc) since a BIO
    column cannot encode same-type overlap. Deterministic given input order.
    """
    if not rows_per_model:
        raise ValueError("need at least one model's rows")
    n_tokens = len(rows_per_model[0])
    if any(len(rows) != n_tokens for rows in rows_per_model):
        raise ValueError("models disagree on sentence length")
    votes: Counter[Span] = Counter()
    for rows in rows_per_model:
        votes.update(spans_from_rows(rows).spans)
    candidates = sorted(
        (s for s, v in votes.items() if v >= min_votes),
        key=lambda s: (-votes[s], -(s[1] - s[0]), s[0], s[2]),
    )
    accepted: list[Span] = []
    occupied: dict[str, list[tuple[int, int]]] = {}
    dropped = 0
    for start, end, typ in candidates:
        if any(start < e and b < end for b, e in occupied.get(typ, [])):
            dropped += 1
            continue
        accepted.append((start, end, typ))
        occupied.setdefault(typ, []).append((start, end))
    return EnsembleResult(
        rows=encode_spans_to_rows(accepted, n_tokens),
        spans=set(accepted),
        n_dropped=dropped,
    )
