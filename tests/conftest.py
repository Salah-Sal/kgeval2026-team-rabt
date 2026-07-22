"""Shared fixtures: an offline tiny WordPiece tokenizer + random tiny BERT.

torch/transformers are imported lazily inside the fixtures so the pure-python
tests keep running in environments without the train group installed.
"""

import pytest

# includes the RE marker characters @ # * ^ so they tokenize to real ids
VOCAB = [
    "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
    "aa", "##aa", "bb", "##bb", "cc", "x",
    "@", "#", "*", "^",
]


@pytest.fixture(scope="session")
def tiny_tokenizer():
    # built via the tokenizers library: transformers 5 no longer loads a
    # BertTokenizerFast vocab from a bare vocab_file (only specials survive)
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    backend = tokenizers.Tokenizer(
        tokenizers.models.WordPiece(
            {t: i for i, t in enumerate(VOCAB)}, unk_token="[UNK]"
        )
    )
    backend.pre_tokenizer = tokenizers.pre_tokenizers.BertPreTokenizer()
    backend.post_processor = tokenizers.processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B [SEP]",
        special_tokens=[("[CLS]", VOCAB.index("[CLS]")), ("[SEP]", VOCAB.index("[SEP]"))],
    )
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )


@pytest.fixture
def tiny_encoder():
    """Function-scoped: training mutates weights, so each test gets a fresh one."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    config = transformers.BertConfig(
        vocab_size=len(VOCAB),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=37,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    return transformers.BertModel(config)
