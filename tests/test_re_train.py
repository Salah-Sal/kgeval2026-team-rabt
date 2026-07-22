"""RE train-stack tests: batch positions, no_relation masking, tiny end-to-end run."""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from kgeval.re_markers import insert_typed_markers
from kgeval.re_model import MarkerClassifier
from kgeval.re_train import (
    REExample,
    RETrainConfig,
    build_examples,
    make_batches,
    predict,
    run_re_training,
)

LABELS = ["no_relation", "rel_a", "rel_b"]


def make_examples(n: int, offset: int = 0) -> list[REExample]:
    examples = []
    for i in range(n):
        # rel_a sentences contain cc between the arguments, rel_b do not —
        # a learnable signal for the tiny encoder
        label = LABELS[1 + i % 2]
        middle = "cc" if label == "rel_a" else "bb"
        sentence = f"aa {middle} x"
        marked = insert_typed_markers(sentence, "aa", "PERS", "x", "GPE")
        examples.append(REExample(f"t{offset + i}", marked, label))
    return examples


def test_build_examples_counts_fallbacks():
    records = [
        {"triple_id": 1, "sentence": "aa bb x", "subject": "aa", "object": "x",
         "relation": "rel_a", "sentence_id": "s1"},
        {"triple_id": 2, "sentence": "aa bb", "subject": "aa", "object": "zz",
         "relation": "rel_b", "sentence_id": "s1"},
    ]
    examples, fallbacks = build_examples(records, [("PERS", "GPE"), ("PERS", "GPE")])
    assert [ex.triple_id for ex in examples] == ["1", "2"]
    assert fallbacks == {"object_missing": 1}


def test_batch_marker_positions(tiny_tokenizer):
    examples = make_examples(4)
    label_index = {l: i for i, l in enumerate(LABELS)}
    batches = make_batches(examples, tiny_tokenizer, label_index, batch_size=2)
    for b in batches:
        assert b.n_truncated_markers == 0
        for bi in range(len(b.triple_ids)):
            ids = b.enc["input_ids"][bi]
            assert tiny_tokenizer.convert_ids_to_tokens([ids[b.subj_pos[bi]]]) == ["@"]
            assert tiny_tokenizer.convert_ids_to_tokens([ids[b.obj_pos[bi]]]) == ["#"]


def test_truncated_marker_falls_back_to_cls(tiny_tokenizer):
    long_sentence = "bb " * 60 + "x"
    marked = insert_typed_markers(long_sentence, "bb", "PERS", "x", "GPE")
    examples = [REExample("t0", marked, "rel_a")]
    batches = make_batches(examples, tiny_tokenizer, {"rel_a": 0}, batch_size=1, max_len=16)
    assert batches[0].n_truncated_markers >= 1
    assert int(batches[0].obj_pos[0]) == 0  # CLS fallback


def test_predict_never_emits_masked_label(tiny_tokenizer, tiny_encoder):
    examples = make_examples(8)
    label_index = {l: i for i, l in enumerate(LABELS)}
    batches = make_batches(examples, tiny_tokenizer, label_index, batch_size=4)
    torch.manual_seed(5)
    model = MarkerClassifier(tiny_encoder, len(LABELS))
    # bias the model heavily toward no_relation to prove masking works
    with torch.no_grad():
        model.ffnn[-1].bias[0] = 50.0
    preds = predict(model, batches, LABELS, "cpu", masked_label="no_relation")
    assert set(preds) == {ex.triple_id for ex in examples}
    assert all(p != "no_relation" for p in preds.values())


def test_run_re_training(tiny_tokenizer, tiny_encoder, tmp_path):
    cfg = RETrainConfig(max_epochs=2, patience=3, batch_size=4, max_len=32, seed=7)
    model, tok, result = run_re_training(
        make_examples(16), make_examples(6, offset=100), LABELS, cfg, tmp_path,
        encoder=tiny_encoder, tokenizer=tiny_tokenizer, log=lambda *_: None,
    )
    assert (tmp_path / "best_re_model.pt").exists()
    assert (tmp_path / "re_history.json").exists()
    assert len(result["history"]) == 2
    assert result["masked_label"] == "no_relation"
    assert 0.0 <= result["best_val_micro_f1"] <= 1.0
    assert result["label_vocab"] == LABELS
