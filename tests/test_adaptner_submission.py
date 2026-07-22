"""Round-trip + tamper tests for the AdaptNER submission contract.

All fixtures are synthetic — licensed task data never enters the repo.
"""

import zipfile

import pytest

from kgeval.adaptner_submission import validate_submission, write_submission
from kgeval.columns import ENTITY_TYPES, NUM_TYPES, TYPE_INDEX


def o_row() -> list[str]:
    return ["O"] * NUM_TYPES


def tagged_row(**overrides: str) -> list[str]:
    row = o_row()
    for typ, tag in overrides.items():
        row[TYPE_INDEX[typ]] = tag
    return row


def o_line(token: str) -> str:
    return token + " " + " ".join(o_row())


@pytest.fixture
def reference_dir(tmp_path):
    ref = tmp_path / "ref"
    ref.mkdir()
    # Domain A: two sentences, file does NOT end with a blank line.
    a = "\r\n".join([o_line("tokA1"), o_line("tokA2"), "", o_line("tokA3")]) + "\r\n"
    (ref / "Alpha.txt").write_bytes(a.encode("utf-8"))
    # Domain B: one sentence.
    b = "\r\n".join([o_line("tokB1"), o_line("tokB2")]) + "\r\n"
    (ref / "Beta.txt").write_bytes(b.encode("utf-8"))
    return ref


@pytest.fixture
def predictions():
    return {
        "Alpha": [tagged_row(GPE="B-GPE"), o_row(), o_row()],
        "Beta": [o_row(), tagged_row(MONEY="B-MONEY")],
    }


def test_round_trip_passes(tmp_path, reference_dir, predictions):
    out = tmp_path / "sub.zip"
    info = write_submission(reference_dir, predictions, out)
    assert info["token_lines"] == 5
    assert info["inter_domain_blanks_inserted"] == 1  # Alpha didn't end blank
    rep = validate_submission(out, reference_dir)
    assert rep.ok, rep.pretty()
    assert rep.stats["token_lines_checked"] == 5
    assert rep.stats["entity_tags"] == 2


def test_member_name_needs_pred_marker(tmp_path, reference_dir, predictions):
    with pytest.raises(ValueError, match="_pred_"):
        write_submission(reference_dir, predictions, tmp_path / "s.zip", member_name="preds.txt")


def test_row_count_mismatch_raises(tmp_path, reference_dir, predictions):
    predictions["Beta"] = predictions["Beta"][:1]
    with pytest.raises(ValueError, match="fewer prediction rows"):
        write_submission(reference_dir, predictions, tmp_path / "s.zip")


def _tampered_zip(tmp_path, reference_dir, predictions, mutate):
    out = tmp_path / "good.zip"
    write_submission(reference_dir, predictions, out)
    with zipfile.ZipFile(out) as z:
        name = z.namelist()[0]
        text = z.read(name).decode("utf-8")
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr(name, mutate(text))
    return bad


def test_validator_catches_wrong_token(tmp_path, reference_dir, predictions):
    bad = _tampered_zip(tmp_path, reference_dir, predictions, lambda t: t.replace("tokA2", "tokAX"))
    rep = validate_submission(bad, reference_dir)
    assert not rep.ok and any("tokAX" in e for e in rep.errors)


def test_validator_catches_missing_column(tmp_path, reference_dir, predictions):
    def drop_last_field(text):
        lines = text.split("\r\n")
        lines[0] = " ".join(lines[0].split(" ")[:-1])
        return "\r\n".join(lines)

    bad = _tampered_zip(tmp_path, reference_dir, predictions, drop_last_field)
    rep = validate_submission(bad, reference_dir)
    assert not rep.ok and any("fields" in e for e in rep.errors)


def test_validator_catches_tag_in_wrong_column(tmp_path, reference_dir, predictions):
    # a GPE tag sitting in the CARDINAL column (position 1) is a contract breach
    def swap(text):
        lines = text.split("\r\n")
        fields = lines[0].split(" ")
        fields[1 + TYPE_INDEX["CARDINAL"]] = "B-GPE"
        lines[0] = " ".join(fields)
        return "\r\n".join(lines)

    bad = _tampered_zip(tmp_path, reference_dir, predictions, swap)
    rep = validate_submission(bad, reference_dir)
    assert not rep.ok and any("invalid in column" in e for e in rep.errors)


def test_validator_catches_missing_sentence_break(tmp_path, reference_dir, predictions):
    bad = _tampered_zip(
        tmp_path, reference_dir, predictions, lambda t: t.replace("\r\n\r\n", "\r\n", 1)
    )
    rep = validate_submission(bad, reference_dir)
    assert not rep.ok


def test_all_o_submission_is_valid_but_warned(tmp_path, reference_dir):
    all_o = {"Alpha": [o_row()] * 3, "Beta": [o_row()] * 2}
    out = tmp_path / "sub.zip"
    write_submission(reference_dir, all_o, out)
    rep = validate_submission(out, reference_dir)
    assert rep.ok
    assert any("all-O" in w for w in rep.warnings)


def test_column_order_is_the_contract_order():
    assert ENTITY_TYPES[0] == "CARDINAL" and ENTITY_TYPES[-1] == "WEBSITE"
    assert len(ENTITY_TYPES) == 21
