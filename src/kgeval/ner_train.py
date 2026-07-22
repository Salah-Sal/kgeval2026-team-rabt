"""Training loop for the AdaptNER baseline (design §3.1–3.2).

Recipe: lr 1e-5 AdamW (weight decay 1e-3 except bias/LayerNorm, per A10),
batch 16 dynamic-padded, max len 512 with word-boundary chunking, exponential
LR decay per epoch, early stopping on Wojood-val span micro-F1, fp16 autocast
on CUDA. Checkpoints and history land in out_dir; the caller gets the model
back with best weights loaded.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from . import ner_data
from .ner_model import MultiHeadTagger, SigmoidTagger
from .ner_scoring import NERScore, score_tag_docs


@dataclass
class TrainConfig:
    model_name: str = "aubmindlab/bert-base-arabertv02"
    lr: float = 1e-5
    weight_decay: float = 1e-3
    lr_gamma: float = 0.95
    batch_size: int = 16
    max_len: int = 512
    dropout: float = 0.1
    max_epochs: int = 40
    patience: int = 5
    seed: int = 13
    head: str = "multi"  # "multi" (A10 unified loss) | "sigmoid" (B22, use lr 6e-5)
    # "best": keep the best-val-F1 checkpoint, early-stop on patience.
    # "last": fixed-epoch schedule — train to max_epochs, keep final weights;
    # val is monitoring only (train+val-combined runs have no held-out val).
    select: str = "best"
    clip_grad_norm: float | None = 1.0
    wall_limit_s: float | None = None
    loss_lambdas: tuple[float, float, float, float] = (0.4, 0.2, 0.2, 0.2)
    tversky_alpha: float = 0.5
    focal_gamma: float = 2.0
    var_penalty: float = 5.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def predict_docs(
    model,
    batches: list[ner_data.Batch],
    sentence_lengths: list[int],
    device: str,
) -> list[list[list[str]]]:
    """Per-sentence 21-tag rows for pre-collated batches (gold not needed)."""
    model.eval()
    preds = []
    with torch.no_grad():
        for b in batches:
            logits = model(
                b.enc["input_ids"].to(device), b.enc["attention_mask"].to(device)
            )
            preds.append(model.decode(logits).cpu())
    return ner_data.assemble_predictions(preds, batches, sentence_lengths)


def evaluate(
    model,
    batches: list[ner_data.Batch],
    gold_docs: list[list[list[str]]],
    device: str,
) -> tuple[NERScore, list[list[list[str]]]]:
    """Span micro-F1 of the model on pre-collated batches vs gold tag docs."""
    pred_docs = predict_docs(model, batches, [len(rows) for rows in gold_docs], device)
    gold_tags = [[row[1:] for row in rows] for rows in gold_docs]
    return score_tag_docs(gold_tags, pred_docs), pred_docs


def run_training(
    train_docs: list[list[list[str]]],
    val_docs: list[list[list[str]]],
    cfg: TrainConfig,
    out_dir: str | Path,
    encoder=None,
    tokenizer=None,
    log=print,
) -> tuple[torch.nn.Module, object, dict]:
    if cfg.select not in ("best", "last"):
        raise ValueError(f"unknown select {cfg.select!r}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if encoder is None:
        from transformers import AutoModel

        encoder = AutoModel.from_pretrained(cfg.model_name)
    if cfg.head == "multi":
        model = MultiHeadTagger(
            encoder, dropout=cfg.dropout, loss_lambdas=cfg.loss_lambdas,
            tversky_alpha=cfg.tversky_alpha, focal_gamma=cfg.focal_gamma,
            var_penalty=cfg.var_penalty,
        ).to(device)
    elif cfg.head == "sigmoid":
        model = SigmoidTagger(encoder, dropout=cfg.dropout).to(device)
    else:
        raise ValueError(f"unknown head {cfg.head!r}")

    t0 = time.time()
    train_batches = ner_data.make_batches(
        ner_data.build_examples(train_docs, tokenizer, cfg.max_len),
        tokenizer, cfg.batch_size, cfg.max_len,
    )
    val_batches = ner_data.make_batches(
        ner_data.build_examples(val_docs, tokenizer, cfg.max_len),
        tokenizer, cfg.batch_size, cfg.max_len,
    )
    log(f"[data] {len(train_batches)} train / {len(val_batches)} val batches "
        f"(size {cfg.batch_size}, tensorized in {time.time() - t0:.0f}s)")

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
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.lr_gamma)
    amp = device == "cuda"
    scaler = torch.amp.GradScaler(device, enabled=amp)

    rng = random.Random(cfg.seed)
    best_path = out / "best_model.pt"
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
                    b.enc["input_ids"].to(device), b.enc["attention_mask"].to(device)
                )
            loss, _parts = model.compute_loss(logits, b.labels.to(device))
            scaler.scale(loss).backward()
            if cfg.clip_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(loss.detach())
        scheduler.step()

        score, _ = evaluate(model, val_batches, val_docs, device)
        p, r, f1 = score.micro
        entry = {
            "epoch": epoch,
            "train_loss": round(epoch_loss / len(train_batches), 5),
            "val_p": round(p, 5),
            "val_r": round(r, 5),
            "val_f1": round(f1, 5),
            "val_repairs": score.n_pred_repairs,
            "lr": scheduler.get_last_lr()[0],
            "seconds": round(time.time() - t_epoch, 1),
        }
        history.append(entry)
        log(f"[epoch {epoch}] loss {entry['train_loss']:.4f} "
            f"val-F1 {f1:.4f} (P {p:.4f} R {r:.4f}) {entry['seconds']:.0f}s")

        if f1 > best_f1:
            best_f1, best_epoch, bad_epochs = f1, epoch, 0
            if cfg.select == "best":
                torch.save(model.state_dict(), best_path)
        else:
            bad_epochs += 1
            if cfg.select == "best" and bad_epochs >= cfg.patience:
                stopped = "early_stopping"
                break
        if cfg.wall_limit_s is not None and time.time() - t0 > cfg.wall_limit_s:
            stopped = "wall_limit"
            break

    if cfg.select == "last":
        # final weights under the checkpoint's usual name; no reload needed
        torch.save(model.state_dict(), best_path)
    else:
        model.load_state_dict(
            torch.load(best_path, map_location=device, weights_only=True)
        )
    result = {
        "config": asdict(cfg),
        "device": device,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_f1": best_f1,
        "final_epoch": history[-1]["epoch"] if history else -1,
        "stopped": stopped,
        "total_seconds": round(time.time() - t0, 1),
    }
    (out / "history.json").write_text(json.dumps(result, indent=2))
    return model, tokenizer, result
