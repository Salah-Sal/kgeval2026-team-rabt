"""Per-column BIO decoding with IOB repair, and span-set extraction.

Spans are (start, end_exclusive, type) over token positions within one
sentence. An `I-x` with no open span of type x (after O or at sentence start)
starts a new span and is counted as a repair — the standard IOB2 repair, also
applied to model output before submission writing.
"""

from __future__ import annotations

from dataclasses import dataclass

from .columns import ENTITY_TYPES

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
