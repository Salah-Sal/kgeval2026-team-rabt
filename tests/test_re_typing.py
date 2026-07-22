from collections import Counter

from kgeval.re_typing import (
    UNK_TYPE,
    assign_types,
    build_lexicon,
    entity_types,
    type_mention,
)

FLAT = [("Amr", ["B-PERS"]), ("visited", ["O"]), ("Cairo", ["B-GPE"])]
NESTED = [("Bank", ["B-ORG"]), ("of", ["I-ORG"]), ("Cairo", ["I-ORG", "B-GPE"])]


def test_entity_types_flat():
    assert entity_types(FLAT) == {"Amr": Counter({"PERS": 1}), "Cairo": Counter({"GPE": 1})}


def test_entity_types_nested():
    ents = entity_types(NESTED)
    assert ents["Bank of Cairo"] == Counter({"ORG": 1})
    assert ents["Cairo"] == Counter({"GPE": 1})


def test_type_mention_priority():
    lexicon = build_lexicon({"1": FLAT, "2": NESTED})
    ents = entity_types(FLAT)
    assert type_mention("Amr", ents, lexicon) == ("PERS", "sentence")
    # not in this sentence, but in the corpus lexicon
    assert type_mention("Bank of Cairo", ents, lexicon) == ("ORG", "lexicon")
    assert type_mention("Atlantis", ents, lexicon) == (UNK_TYPE, "unk")
    assert type_mention("Amr", None, lexicon) == ("PERS", "lexicon")


def test_assign_types_and_stats():
    sentences = {"10": FLAT, "20": NESTED}
    lexicon = build_lexicon(sentences)
    records = [
        {"sentence_id": "10", "subject": "Amr", "object": "Cairo", "relation": "lives_in"},
        {"sentence_id": "20", "subject": "Bank of Cairo", "object": "Cairo", "relation": "headquartered_in"},
        {"sentence_id": "manual_1", "subject": "Amr", "object": "Atlantis", "relation": "lives_in"},
    ]
    types, stats = assign_types(records, sentences, lexicon)
    assert types == [("PERS", "GPE"), ("ORG", "GPE"), ("PERS", UNK_TYPE)]
    assert stats["joined"] == 2 and stats["unjoined"] == 1
    assert stats["subj_sentence"] == 2 and stats["subj_lexicon"] == 1
    assert stats["obj_unk"] == 1
