"""Train-stack tests: loss math, label/assembly round-trip, tiny end-to-end run.

All synthetic (licensed data never enters the repo). Offline by construction:
the tokenizer is a BertTokenizerFast built from a temp vocab file and the
encoder a small randomly initialized BertModel.
"""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from kgeval import ner_data
from kgeval.columns import ENTITY_TYPES, TYPE_INDEX
from kgeval.ner_model import N_TYPES, MultiHeadTagger, unified_loss
from kgeval.ner_train import TrainConfig, evaluate, run_training

from conftest import VOCAB  # shared tiny vocab; fixtures come from conftest.py


def make_doc(words: list[str], spans: list[tuple[int, int, str]]) -> list[list[str]]:
    """Rows of [token, tag*21] with the given (start, end_exclusive, type) spans."""
    rows = [[w] + ["O"] * N_TYPES for w in words]
    for start, end, typ in spans:
        ci = TYPE_INDEX[typ]
        rows[start][1 + ci] = f"B-{typ}"
        for i in range(start + 1, end):
            rows[i][1 + ci] = f"I-{typ}"
    return rows


DOCS = [
    # "aaaa" tokenizes to 2 subtokens — first-subtoken labeling exercised
    make_doc(["aa", "aaaa", "bb", "x"], [(0, 2, "PERS"), (3, 4, "GPE")]),
    make_doc(["bb", "cc", "aa"], [(1, 3, "ORG"), (2, 3, "GPE")]),  # nested columns
    make_doc(["x", "aa"], []),
    make_doc(["cc"] * 12 + ["aaaa"] * 6, [(0, 2, "DATE"), (14, 16, "PERS")]),
]


def test_tag_id_roundtrip():
    for typ in ("PERS", "GPE"):
        for tag in ("O", f"B-{typ}", f"I-{typ}"):
            i = ner_data.tag_id(tag)
            assert ner_data.id_to_tag(i, typ) == tag


def test_unified_loss_finite_and_differentiable():
    torch.manual_seed(1)
    logits = torch.randn(2, 9, N_TYPES, 3, requires_grad=True)
    labels = torch.randint(0, 3, (2, 9, N_TYPES))
    labels[:, 5:] = -100  # padding tail
    loss, parts = unified_loss(logits, labels)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert set(parts) == {"ce", "dice", "tversky", "focal", "head_var"}


def test_unified_loss_prefers_correct_predictions():
    torch.manual_seed(2)
    labels = torch.randint(0, 3, (2, 9, N_TYPES))
    good = torch.full((2, 9, N_TYPES, 3), -8.0)
    good.scatter_(-1, labels.unsqueeze(-1), 8.0)
    bad_labels = (labels + 1) % 3
    bad = torch.full((2, 9, N_TYPES, 3), -8.0)
    bad.scatter_(-1, bad_labels.unsqueeze(-1), 8.0)
    good_loss, _ = unified_loss(good, labels)
    bad_loss, _ = unified_loss(bad, labels)
    assert good_loss < 0.1 < bad_loss


def test_unified_loss_variance_zero_when_heads_agree():
    torch.manual_seed(3)
    one_head = torch.randn(2, 9, 1, 3)
    logits = one_head.repeat(1, 1, N_TYPES, 1)
    labels = torch.randint(0, 3, (2, 9, 1)).repeat(1, 1, N_TYPES)
    _, parts = unified_loss(logits, labels)
    assert parts["head_var"] == pytest.approx(0.0, abs=1e-9)


def test_label_assembly_roundtrip(tiny_tokenizer):
    # max_len 16 forces the 18-word doc to chunk; labels pushed through the
    # batch tensors and reassembled must reproduce the gold spans exactly
    examples = ner_data.build_examples(DOCS, tiny_tokenizer, max_len=16)
    assert max(ex.n_subtokens for ex in examples) <= 14
    assert any(ex.word_start > 0 for ex in examples)
    batches = ner_data.make_batches(examples, tiny_tokenizer, batch_size=3, max_len=16)
    preds = [b.labels.clamp(min=0) for b in batches]
    docs = ner_data.assemble_predictions(preds, batches, [len(rows) for rows in DOCS])
    gold = [[row[1:] for row in rows] for rows in DOCS]
    assert docs == gold


def test_run_training_and_evaluate(tiny_tokenizer, tiny_encoder, tmp_path):
    cfg = TrainConfig(max_epochs=2, patience=5, batch_size=3, max_len=32, seed=7)
    model, tok, result = run_training(
        DOCS, DOCS[:2], cfg, tmp_path, encoder=tiny_encoder, tokenizer=tiny_tokenizer,
        log=lambda *_: None,
    )
    assert (tmp_path / "best_model.pt").exists()
    assert (tmp_path / "history.json").exists()
    assert len(result["history"]) == 2
    assert result["best_epoch"] >= 0
    assert all(torch.isfinite(torch.tensor(e["train_loss"])) for e in result["history"])
    # evaluate on the training docs runs the full predict→assemble→score path
    batches = ner_data.make_batches(
        ner_data.build_examples(DOCS, tok, cfg.max_len), tok, cfg.batch_size, cfg.max_len
    )
    score, pred_docs = evaluate(model, batches, DOCS, "cpu")
    assert len(pred_docs) == len(DOCS)
    assert all(len(p) == len(g) for p, g in zip(pred_docs, DOCS))
    p, r, f1 = score.micro
    assert 0.0 <= f1 <= 1.0


def test_run_training_select_last_keeps_final_weights(
    tiny_tokenizer, tiny_encoder, tmp_path
):
    import copy

    cfg = TrainConfig(max_epochs=3, patience=1, batch_size=3, max_len=32, seed=7,
                      select="last")
    model, _tok, result = run_training(
        DOCS, DOCS[:2], cfg, tmp_path,
        encoder=copy.deepcopy(tiny_encoder), tokenizer=tiny_tokenizer,
        log=lambda *_: None,
    )
    # patience=1 must NOT early-stop a select="last" run
    assert len(result["history"]) == 3
    assert result["stopped"] == "max_epochs"
    assert result["final_epoch"] == 2
    assert result["config"]["select"] == "last"
    # the saved checkpoint holds the in-memory final weights, not a reload of best
    saved = torch.load(tmp_path / "best_model.pt", weights_only=True)
    live = model.state_dict()
    assert all(torch.equal(saved[k], live[k].cpu()) for k in saved)


def test_multihead_tagger_shapes(tiny_encoder):
    model = MultiHeadTagger(tiny_encoder)
    ids = torch.randint(5, len(VOCAB), (2, 11))
    mask = torch.ones_like(ids)
    logits = model(ids, mask)
    assert logits.shape == (2, 11, N_TYPES, 3)
    assert len(ENTITY_TYPES) == N_TYPES == 21


def test_sigmoid_tagger_decode_semantics():
    from kgeval.ner_model import SigmoidTagger

    logits = torch.full((1, 2, 2 * N_TYPES + 1), -10.0)
    logits[0, 0, 0] = 10.0  # B bit of type 0 confidently on
    logits[0, 1, N_TYPES + 3] = 10.0  # I bit of type 3 confidently on

    class Stub(SigmoidTagger):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.threshold = 0.5

    tags = Stub().decode(logits)
    assert tags.shape == (1, 2, N_TYPES)
    assert tags[0, 0, 0] == 1 and tags[0, 0].sum() == 1  # lone B
    assert tags[0, 1, 3] == 2 and tags[0, 1].sum() == 2  # lone I
    all_off = Stub().decode(torch.full((1, 1, 2 * N_TYPES + 1), -10.0))
    assert all_off.sum() == 0  # below threshold everywhere -> all O


def test_sigmoid_loss_prefers_correct_bits(tiny_encoder):
    from kgeval.ner_model import SigmoidTagger

    model = SigmoidTagger(tiny_encoder)
    labels = torch.randint(0, 3, (2, 5, N_TYPES))
    good = torch.full((2, 5, 2 * N_TYPES + 1), -10.0)
    good[..., :N_TYPES][labels == 1] = 10.0
    good[..., N_TYPES : 2 * N_TYPES][labels == 2] = 10.0
    good[..., -1][(labels == 0).all(dim=-1)] = 10.0
    good_loss, parts = model.compute_loss(good, labels)
    bad_loss, _ = model.compute_loss(-good, labels)
    assert good_loss < 0.01 < bad_loss
    assert "focal_bce" in parts


def test_run_training_sigmoid_head(tiny_tokenizer, tiny_encoder, tmp_path):
    cfg = TrainConfig(head="sigmoid", lr=6e-5, max_epochs=2, patience=5,
                      batch_size=3, max_len=32, seed=7)
    model, _tok, result = run_training(
        DOCS, DOCS[:2], cfg, tmp_path, encoder=tiny_encoder, tokenizer=tiny_tokenizer,
        log=lambda *_: None,
    )
    assert type(model).__name__ == "SigmoidTagger"
    assert len(result["history"]) == 2
    assert result["config"]["head"] == "sigmoid"
