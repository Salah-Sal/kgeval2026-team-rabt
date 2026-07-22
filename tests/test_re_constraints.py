import pytest

from kgeval.re_constraints import SPEC, ConstraintTable, observed_pairs
from kgeval.re_typing import UNK_TYPE


def test_spec_covers_40_positive_labels_minus_employee_of():
    # every positive label except Affiliation.employee_of has a Table-10 row
    assert len(SPEC) == 39
    assert "Affiliation.employee_of" not in SPEC
    assert "Organization.has_propoerty" in SPEC  # the dataset's typo, verbatim


def test_spec_product_admissible():
    table = ConstraintTable(observed={})
    assert table.admissible("Location.located_in", "FAC", "GPE")
    assert table.admissible("Location.located_in", "ORG", "LOC")
    assert not table.admissible("Location.located_in", "PERS", "GPE")


def test_observed_extends_spec():
    records = [{"relation": "Location.located_in"}]
    observed = observed_pairs(records, [("PERS", "GPE")])
    table = ConstraintTable(observed)
    assert table.admissible("Location.located_in", "PERS", "GPE")  # observed
    assert table.admissible("Location.located_in", "FAC", "GPE")  # spec kept
    assert not table.admissible("Location.located_in", "PERS", "PERS")


def test_unk_and_unknown_labels_permissive():
    table = ConstraintTable(observed={})
    assert table.admissible("Location.located_in", UNK_TYPE, "GPE")
    assert table.admissible("Location.located_in", "FAC", UNK_TYPE)
    assert table.admissible("Affiliation.employee_of", "PERS", "ORG")  # no spec row
    assert table.admissible("no_relation", "PERS", "PERS")


def test_observed_pairs_skips_unk():
    records = [{"relation": "r"}, {"relation": "r"}]
    observed = observed_pairs(records, [("PERS", UNK_TYPE), ("PERS", "GPE")])
    assert observed == {"r": {("PERS", "GPE")}}


def test_penalty_flips_inadmissible_argmax():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from kgeval.re_train import REBatch, predict

    vocab = ["Location.located_in", "Personal.birth_place"]

    class Stub(torch.nn.Module):
        def forward(self, input_ids, attention_mask, subj_pos, obj_pos):
            # located_in slightly ahead of birth_place
            return torch.tensor([[2.0, 1.5]])

    batch = REBatch(
        enc={"input_ids": torch.zeros(1, 4, dtype=torch.long),
             "attention_mask": torch.ones(1, 4, dtype=torch.long)},
        subj_pos=torch.tensor([1]),
        obj_pos=torch.tensor([2]),
        label_ids=torch.tensor([0]),
        triple_ids=["t1"],
        type_pairs=[("PERS", "GPE")],  # PERS subject: inadmissible for located_in
        n_truncated_markers=0,
    )
    table = ConstraintTable(observed={})
    unconstrained = predict(Stub(), [batch], vocab, "cpu", masked_label=None)
    constrained = predict(Stub(), [batch], vocab, "cpu", masked_label=None,
                          constraints=table, penalty=3.0)
    assert unconstrained == {"t1": "Location.located_in"}
    assert constrained == {"t1": "Personal.birth_place"}
