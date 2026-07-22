"""Training loop for the RE typed-marker baseline (design §4.1, §4.5).

41-way training (negatives kept — downsampling costs ~10 F1 per F08); local
validation mimics the test condition: no_relation masked out, forced choice
over the 40 positives, scored positives-only via re_scoring. Marker char
offsets → token positions via char_to_token; a marker truncated away falls
back to CLS (position 0) and is counted, never fatal.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .re_constraints import ConstraintTable
from .re_data import detect_no_relation
from .re_markers import Marked, insert_typed_markers
from .re_typing import UNK_TYPE
from .re_model import MarkerClassifier
from .re_scoring import REScore, score


@dataclass
class REExample:
    triple_id: str
    marked: Marked
    label: str
    subj_type: str = UNK_TYPE
    obj_type: str = UNK_TYPE


@dataclass
class REBatch:
    enc: object
    subj_pos: torch.Tensor
    obj_pos: torch.Tensor
    label_ids: torch.Tensor
    triple_ids: list[str]
    type_pairs: list[tuple[str, str]]
    n_truncated_markers: int


@dataclass
class RETrainConfig:
    model_name: str = "aubmindlab/bert-base-arabertv02"
    lr: float = 2e-5
    weight_decay: float = 0.01
    batch_size: int = 32
    max_len: int = 256
    dropout: float = 0.1
    max_epochs: int = 10
    patience: int = 2
    seed: int = 13
    clip_grad_norm: float | None = 1.0
    wall_limit_s: float | None = None


def build_examples(
    records: list[dict], types: list[tuple[str, str]]
) -> tuple[list[REExample], dict]:
    examples = []
    fallbacks: dict[str, int] = {}
    for rec, (s_type, o_type) in zip(records, types, strict=True):
        marked = insert_typed_markers(
            rec["sentence"], rec["subject"], s_type, rec["object"], o_type
        )
        if marked.fallback:
            fallbacks[marked.fallback] = fallbacks.get(marked.fallback, 0) + 1
        examples.append(
            REExample(str(rec["triple_id"]), marked, rec["relation"], s_type, o_type)
        )
    return examples, fallbacks


def make_batches(
    examples: list[REExample],
    tokenizer,
    label_index: dict[str, int],
    batch_size: int,
    max_len: int = 256,
) -> list[REBatch]:
    batches: list[REBatch] = []
    ordered = sorted(examples, key=lambda ex: len(ex.marked.text))
    for i in range(0, len(ordered), batch_size):
        group = ordered[i : i + batch_size]
        enc = tokenizer(
            [ex.marked.text for ex in group],
            truncation=True,
            max_length=max_len,
            padding=True,
            return_tensors="pt",
        )
        subj_pos, obj_pos, truncated = [], [], 0
        for bi, ex in enumerate(group):
            positions = []
            for char in (ex.marked.subj_start, ex.marked.obj_start):
                pos = enc.char_to_token(bi, char)
                if pos is None:  # marker truncated away → CLS
                    truncated += 1
                    pos = 0
                positions.append(pos)
            subj_pos.append(positions[0])
            obj_pos.append(positions[1])
        batches.append(
            REBatch(
                enc=enc,
                subj_pos=torch.tensor(subj_pos),
                obj_pos=torch.tensor(obj_pos),
                label_ids=torch.tensor([label_index[ex.label] for ex in group]),
                triple_ids=[ex.triple_id for ex in group],
                type_pairs=[(ex.subj_type, ex.obj_type) for ex in group],
                n_truncated_markers=truncated,
            )
        )
    return batches


def predict(
    model: MarkerClassifier,
    batches: list[REBatch],
    label_vocab: list[str],
    device: str,
    masked_label: str | None,
    constraints: ConstraintTable | None = None,
    penalty: float = 0.0,
) -> dict[str, str]:
    """triple_id → label, with masked_label (no_relation) never emitted.

    With constraints + penalty > 0, inadmissible labels for each example's
    (subject_type, object_type) get `penalty` subtracted from their logit —
    soft masking per design §4.3, never hard."""
    masked_id = label_vocab.index(masked_label) if masked_label else None
    model.eval()
    out: dict[str, str] = {}
    with torch.no_grad():
        for b in batches:
            logits = model(
                b.enc["input_ids"].to(device),
                b.enc["attention_mask"].to(device),
                b.subj_pos.to(device),
                b.obj_pos.to(device),
            )
            if constraints is not None and penalty:
                inadmissible = torch.tensor(
                    [constraints.penalty_mask(label_vocab, s, o) for s, o in b.type_pairs],
                    device=logits.device,
                )
                logits = logits - penalty * inadmissible.float()
            if masked_id is not None:
                logits[:, masked_id] = float("-inf")
            for tid, li in zip(b.triple_ids, logits.argmax(dim=-1).tolist()):
                out[tid] = label_vocab[li]
    return out


def run_re_training(
    train_examples: list[REExample],
    val_examples: list[REExample],
    label_vocab: list[str],
    cfg: RETrainConfig,
    out_dir: str | Path,
    encoder=None,
    tokenizer=None,
    log=print,
) -> tuple[MarkerClassifier, object, dict]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if encoder is None:
        from transformers import AutoModel

        encoder = AutoModel.from_pretrained(cfg.model_name)
    model = MarkerClassifier(encoder, len(label_vocab), dropout=cfg.dropout).to(device)

    label_index = {l: i for i, l in enumerate(label_vocab)}
    neg = detect_no_relation(set(label_vocab))
    t0 = time.time()
    train_batches = make_batches(
        train_examples, tokenizer, label_index, cfg.batch_size, cfg.max_len
    )
    val_batches = make_batches(
        val_examples, tokenizer, label_index, cfg.batch_size, cfg.max_len
    )
    val_gold = [(ex.triple_id, ex.label) for ex in val_examples]
    n_truncated = sum(b.n_truncated_markers for b in train_batches + val_batches)
    log(f"[data] {len(train_batches)} train / {len(val_batches)} val batches, "
        f"{n_truncated} truncated markers, neg label {neg!r}")

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        (no_decay if name.endswith(".bias") or "LayerNorm" in name else decay).append(param)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
    )
    amp = device == "cuda"
    scaler = torch.amp.GradScaler(device, enabled=amp)
    ce = torch.nn.CrossEntropyLoss()

    rng = random.Random(cfg.seed)
    best_path = out / "best_re_model.pt"
    history: list[dict] = []
    best_f1, best_epoch, bad_epochs = -1.0, -1, 0
    stopped = "max_epochs"
    for epoch in range(cfg.max_epochs):
        model.train()
        order = list(range(len(train_batches)))
        rng.shuffle(order)
        epoch_loss = 0.0
        t_epoch = time.time()
        for i in order:
            b = train_batches[i]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device, enabled=amp):
                logits = model(
                    b.enc["input_ids"].to(device),
                    b.enc["attention_mask"].to(device),
                    b.subj_pos.to(device),
                    b.obj_pos.to(device),
                )
                loss = ce(logits.float(), b.label_ids.to(device))
            scaler.scale(loss).backward()
            if cfg.clip_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(loss.detach())

        preds = predict(model, val_batches, label_vocab, device, masked_label=neg)
        val_score = score(val_gold, preds, positives_only=True)
        entry = {
            "epoch": epoch,
            "train_loss": round(epoch_loss / len(train_batches), 5),
            "val_micro_f1": round(val_score.micro_f1, 5),
            "val_macro_f1": round(val_score.macro_f1, 5),
            "seconds": round(time.time() - t_epoch, 1),
        }
        history.append(entry)
        log(f"[epoch {epoch}] loss {entry['train_loss']:.4f} "
            f"val micro-F1 {val_score.micro_f1:.4f} macro-F1 {val_score.macro_f1:.4f} "
            f"{entry['seconds']:.0f}s")

        if val_score.micro_f1 > best_f1:
            best_f1, best_epoch, bad_epochs = val_score.micro_f1, epoch, 0
            torch.save(model.state_dict(), best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                stopped = "early_stopping"
                break
        if cfg.wall_limit_s is not None and time.time() - t0 > cfg.wall_limit_s:
            stopped = "wall_limit"
            break

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    best_preds = predict(model, val_batches, label_vocab, device, masked_label=neg)
    best_score: REScore = score(val_gold, best_preds, positives_only=True)
    result = {
        "config": asdict(cfg),
        "device": device,
        "label_vocab": label_vocab,
        "masked_label": neg,
        "n_truncated_markers": n_truncated,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_micro_f1": best_f1,
        "best_val_macro_f1": best_score.macro_f1,
        "best_confusions": best_score.confusions[:15],
        "stopped": stopped,
        "total_seconds": round(time.time() - t0, 1),
    }
    (out / "re_history.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return model, tokenizer, result
