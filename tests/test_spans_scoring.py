import pytest

from kgeval.ner_scoring import score_span_sets
from kgeval.spans import (
    decode_bio_column,
    encode_spans_to_rows,
    spans_from_rows,
    union_tag_rows,
)


def test_decode_with_iob_repair():
    res = decode_bio_column(["O", "I-GPE", "I-GPE", "O", "B-GPE"], "GPE")
    assert res.spans == {(1, 3, "GPE"), (4, 5, "GPE")}
    assert res.n_repairs == 1


def test_adjacent_b_tags_are_two_spans():
    res = decode_bio_column(["B-GPE", "B-GPE"], "GPE")
    assert res.spans == {(0, 1, "GPE"), (1, 2, "GPE")}
    assert res.n_repairs == 0


def test_foreign_tag_in_column_raises():
    with pytest.raises(ValueError):
        decode_bio_column(["B-ORG"], "GPE")


def test_spans_from_rows_collects_all_columns():
    from kgeval.columns import ENTITY_TYPES, TYPE_INDEX

    def row(**overrides):
        out = ["O"] * len(ENTITY_TYPES)
        for typ, tag in overrides.items():
            out[TYPE_INDEX[typ]] = tag
        return out

    rows = [row(MONEY="B-MONEY"), row(MONEY="I-MONEY", CURR="B-CURR")]
    res = spans_from_rows(rows)
    assert res.spans == {(0, 2, "MONEY"), (1, 2, "CURR")}


def test_encode_spans_roundtrip():
    spans = {(0, 2, "MONEY"), (1, 2, "CURR"), (2, 3, "MONEY")}  # adjacent MONEY
    rows = encode_spans_to_rows(spans, 3)
    assert spans_from_rows(rows).spans == spans


def test_encode_rejects_same_type_overlap_and_bad_bounds():
    with pytest.raises(ValueError):
        encode_spans_to_rows({(0, 2, "GPE"), (1, 3, "GPE")}, 3)
    with pytest.raises(ValueError):
        encode_spans_to_rows({(0, 4, "GPE")}, 3)


def _rows_of(spans, n_tokens):
    return encode_spans_to_rows(spans, n_tokens)


def test_union_is_span_union_when_no_conflict():
    a = _rows_of({(0, 1, "GPE")}, 4)
    b = _rows_of({(2, 4, "ORG")}, 4)
    res = union_tag_rows([a, b])
    assert res.spans == {(0, 1, "GPE"), (2, 4, "ORG")}
    assert res.n_dropped == 0
    assert spans_from_rows(res.rows).spans == res.spans


def test_union_same_type_conflict_resolved_by_votes_then_length():
    # (0,2,GPE) has 2 votes, (1,3,GPE) one vote and overlaps → dropped;
    # (1,3,ORG) overlaps nothing in its own column → kept
    a = _rows_of({(0, 2, "GPE")}, 3)
    b = _rows_of({(0, 2, "GPE"), (1, 3, "ORG")}, 3)
    c = _rows_of({(1, 3, "GPE")}, 3)
    res = union_tag_rows([a, b, c])
    assert res.spans == {(0, 2, "GPE"), (1, 3, "ORG")}
    assert res.n_dropped == 1


def test_union_min_votes_majority():
    a = _rows_of({(0, 1, "GPE"), (2, 3, "ORG")}, 3)
    b = _rows_of({(0, 1, "GPE")}, 3)
    c = _rows_of(set(), 3)
    res = union_tag_rows([a, b, c], min_votes=2)
    assert res.spans == {(0, 1, "GPE")}


def test_union_rejects_length_mismatch():
    with pytest.raises(ValueError):
        union_tag_rows([_rows_of(set(), 3), _rows_of(set(), 4)])


def test_micro_f1_math():
    gold = [{(0, 1, "GPE"), (2, 3, "ORG")}]
    pred = [{(0, 1, "GPE")}]
    s = score_span_sets(gold, pred)
    p, r, f1 = s.micro
    assert (s.tp, s.fp, s.fn) == (1, 0, 1)
    assert p == 1.0 and r == 0.5 and abs(f1 - 2 / 3) < 1e-9
    assert s.per_type["ORG"] == (0, 0, 1)
