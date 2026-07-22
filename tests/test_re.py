import zipfile

import pytest

from kgeval.re_data import detect_no_relation
from kgeval.re_scoring import score
from kgeval.re_submission import validate_predictions, write_predictions
from kgeval.splits import grouped_split

RECORDS = [
    {"triple_id": "1", "sentence_id": "s1"},
    {"triple_id": "2", "sentence_id": "s1"},
    {"triple_id": "3", "sentence_id": "s2"},
]
LABELS = ["Location.located_in", "Personal.has_occupation", "Location.located_in"]


def test_round_trip_passes(tmp_path):
    out = tmp_path / "sub.zip"
    write_predictions(RECORDS, LABELS, out, label_whitelist=set(LABELS))
    rep = validate_predictions(out, RECORDS, label_whitelist=set(LABELS))
    assert rep.ok, rep.pretty()
    assert rep.stats["lines"] == 3


def test_no_relation_is_refused_without_flag(tmp_path):
    with pytest.raises(ValueError, match="no_relation"):
        write_predictions(RECORDS, ["no_relation"] * 3, tmp_path / "s.zip")
    write_predictions(RECORDS, ["no_relation"] * 3, tmp_path / "s.zip", allow_no_relation=True)
    rep = validate_predictions(tmp_path / "s.zip", RECORDS, allow_no_relation=True)
    assert rep.ok and rep.warnings


def test_foreign_label_is_refused(tmp_path):
    with pytest.raises(ValueError, match="whitelist"):
        write_predictions(RECORDS, ["Made.up"] * 3, tmp_path / "s.zip", label_whitelist={"A.b"})


def test_validator_catches_wrong_member_and_order(tmp_path):
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("wrong_name.txt", "1\tA.b\n")
    assert not validate_predictions(bad, RECORDS).ok

    swapped = tmp_path / "swapped.zip"
    with zipfile.ZipFile(swapped, "w") as z:
        z.writestr("predictions.txt", "2\tA.b\n1\tA.b\n3\tA.b\n")
    rep = validate_predictions(swapped, RECORDS)
    assert not rep.ok and any("order" in e for e in rep.errors)


def test_scorer_positives_only():
    gold = [("1", "A.x"), ("2", "B.y"), ("3", "no_relation")]
    pred = {"1": "A.x", "2": "A.x"}
    s = score(gold, pred, positives_only=True)
    assert s.n == 2  # no_relation gold dropped
    assert s.correct == 1
    assert abs(s.micro_f1 - 0.5) < 1e-9
    assert s.confusions[0][:2] == ("B.y", "A.x")


def test_detect_no_relation_variants():
    assert detect_no_relation({"no_relation", "A.b"}) == "no_relation"
    assert detect_no_relation({"NO-RELATION"}) == "NO-RELATION"
    assert detect_no_relation({"A.b"}) is None


def test_grouped_split_never_splits_a_sentence():
    records = [{"triple_id": str(i), "sentence_id": f"s{i % 10}"} for i in range(100)]
    train, val, info = grouped_split(records, val_frac=0.2, seed=7)
    train_groups = {r["sentence_id"] for r in train}
    val_groups = {r["sentence_id"] for r in val}
    assert not (train_groups & val_groups)
    assert info["n_train"] + info["n_val"] == 100


def test_grouped_kfold_partitions_and_balances():
    from kgeval.splits import grouped_kfold

    # uneven group sizes: sentence s0 has 6 triples, others 1-3
    records = []
    sizes = {"s0": 6, "s1": 3, "s2": 3, "s3": 2, "s4": 2, "s5": 1, "s6": 1,
             "s7": 1, "s8": 1, "s9": 1}
    tid = 0
    for sid, n in sizes.items():
        for _ in range(n):
            records.append({"triple_id": str(tid), "sentence_id": sid})
            tid += 1
    folds = grouped_kfold(records, k=3, seed=7)
    assert len(folds) == 3
    all_val_ids = []
    for train, val in folds:
        train_groups = {r["sentence_id"] for r in train}
        val_groups = {r["sentence_id"] for r in val}
        assert not (train_groups & val_groups)  # group never split
        assert len(train) + len(val) == len(records)
        all_val_ids.extend(r["triple_id"] for r in val)
    # val sides partition the record set exactly
    assert sorted(all_val_ids) == sorted(r["triple_id"] for r in records)
    val_sizes = [len(val) for _, val in folds]
    assert max(val_sizes) - min(val_sizes) <= 6  # bounded by the largest group
    # deterministic
    folds2 = grouped_kfold(records, k=3, seed=7)
    assert [[r["triple_id"] for r in v] for _, v in folds] == \
           [[r["triple_id"] for r in v] for _, v in folds2]
