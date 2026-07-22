from kgeval.re_markers import choose_pair, insert_typed_markers, occurrences


def test_occurrences_overlapping():
    assert occurrences("aaaaa", "aaa") == [0, 1, 2]
    assert occurrences("abc", "z") == []
    assert occurrences("abc", "") == []


def test_clean_insertion_subject_first():
    m = insert_typed_markers("X went to Y", "X", "PERS", "Y", "GPE")
    assert m.text == "@ * pers * X @ went to # ^ gpe ^ Y #"
    assert m.text[m.subj_start] == "@"
    assert m.text[m.obj_start] == "#"
    assert m.fallback is None


def test_clean_insertion_object_first():
    m = insert_typed_markers("Y hosts X", "X", "PERS", "Y", "GPE")
    assert m.text == "# ^ gpe ^ Y # hosts @ * pers * X @"
    assert m.text[m.subj_start] == "@"
    assert m.text[m.obj_start] == "#"


def test_min_distance_pair_chosen():
    # subject "aa" occurs at 0 and 13; object at 16 → the occurrence at 13 wins
    sentence = "aa bb cc bb  aa Y"
    assert choose_pair(sentence, "aa", "Y") == ("disjoint", 13, 16)


def test_nested_subject_contains_object():
    # the WojoodRelations pattern: object GPE embedded in the subject ORG name
    m = insert_typed_markers("aa bb cc", "aa bb", "ORG", "bb", "GPE")
    assert m.fallback is None
    assert m.text == "@ * org * aa # ^ gpe ^ bb # @ cc"
    assert m.text[m.subj_start] == "@"
    assert m.text[m.obj_start] == "#"


def test_nested_object_contains_subject():
    m = insert_typed_markers("bb cc dd", "cc", "PERS", "bb cc", "ORG")
    assert m.fallback is None
    assert m.text == "# ^ org ^ bb @ * pers * cc @ # dd"
    assert m.text[m.subj_start] == "@"
    assert m.text[m.obj_start] == "#"


def test_identical_mentions_single_occurrence():
    m = insert_typed_markers("aa cc", "aa", "PERS", "aa", "GPE")
    assert m.fallback is None
    assert m.text == "@ * pers * # ^ gpe ^ aa # @ cc"
    assert m.text[m.subj_start] == "@"
    assert m.text[m.obj_start] == "#"


def test_partial_crossing_falls_back():
    # neither span contains the other and no disjoint pair exists
    m = insert_typed_markers("aa bb cc", "aa bb", "ORG", "bb cc", "GPE")
    assert m.fallback == "overlap"
    assert m.text[m.subj_start] == "@"
    assert m.text[m.obj_start] == "#"
    assert m.text.endswith("# ^ gpe ^ bb cc #")


def test_missing_fallbacks():
    m = insert_typed_markers("aa bb", "aa", "PERS", "zz", "GPE")
    assert m.fallback == "object_missing"
    assert m.text[m.subj_start] == "@" and m.text[m.obj_start] == "#"
    m = insert_typed_markers("aa bb", "zz", "PERS", "bb", "GPE")
    assert m.fallback == "subject_missing"
    assert m.text[m.subj_start] == "@" and m.text[m.obj_start] == "#"
    m = insert_typed_markers("aa bb", "zz", "PERS", "qq", "GPE")
    assert m.fallback == "both_missing"
    assert m.text[m.subj_start] == "@" and m.text[m.obj_start] == "#"


def test_arabic_sentence():
    m = insert_typed_markers(
        "ولد أحمد في صنعاء القديمة", "أحمد", "PERS", "صنعاء", "GPE"
    )
    assert m.fallback is None
    assert m.text[m.subj_start] == "@"
    assert m.text[m.obj_start] == "#"
    assert "* pers * أحمد @" in m.text
    assert "^ gpe ^ صنعاء #" in m.text
