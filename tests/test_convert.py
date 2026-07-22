from kgeval.columns import ENTITY_TYPES, TYPE_INDEX
from kgeval.convert import ConvertStats, nested_tags_to_columns, sentence_to_rows


def cols(**overrides: str) -> list[str]:
    out = ["O"] * len(ENTITY_TYPES)
    for typ, tag in overrides.items():
        out[TYPE_INDEX[typ]] = tag
    return out


def test_flat_tag_lands_in_its_column():
    assert nested_tags_to_columns(["B-MONEY"]) == cols(MONEY="B-MONEY")
    assert nested_tags_to_columns(["O"]) == cols()


def test_nested_tags_fill_two_columns():
    # the canonical Wojood example: ملز inside MONEY, opening nested CURR
    assert nested_tags_to_columns(["I-MONEY", "B-CURR"]) == cols(
        MONEY="I-MONEY", CURR="B-CURR"
    )


def test_same_type_collision_keeps_outer_tag():
    stats = ConvertStats()
    out = nested_tags_to_columns(["I-ORG", "B-ORG"], stats)
    assert out == cols(ORG="I-ORG")  # outer (first-listed) wins
    assert stats.n_same_type_collisions == 1
    assert stats.collisions_by_type["ORG"] == 1


def test_sentence_stats_and_nesting_pairs():
    sentence = [
        ("فئة", ["O"]),
        ("500", ["B-MONEY"]),
        ("ملز", ["I-MONEY", "B-CURR"]),
    ]
    stats = ConvertStats()
    rows = sentence_to_rows(sentence, stats)
    assert [r[0] for r in rows] == ["فئة", "500", "ملز"]
    assert len(rows[0]) == 1 + len(ENTITY_TYPES)
    assert stats.n_tokens == 3
    assert stats.n_multi_tag_tokens == 1
    assert stats.nesting_pairs[("MONEY", "CURR")] == 1
