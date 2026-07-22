import pytest

from kgeval.ner_scoring import score_span_sets
from kgeval.spans import decode_bio_column, spans_from_rows


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


def test_micro_f1_math():
    gold = [{(0, 1, "GPE"), (2, 3, "ORG")}]
    pred = [{(0, 1, "GPE")}]
    s = score_span_sets(gold, pred)
    p, r, f1 = s.micro
    assert (s.tp, s.fp, s.fn) == (1, 0, 1)
    assert p == 1.0 and r == 0.5 and abs(f1 - 2 / 3) < 1e-9
    assert s.per_type["ORG"] == (0, 0, 1)
