"""Multi-head nested tagger + the A10 unified loss.

Architecture (design §3.1): encoder + one 3-way O/B/I softmax head per entity
type. The 21 heads are stored as a single Linear(hidden, 21*3) — each output
unit is independent, so this is the same math as 21 separate Linears in one
matmul.

Loss (A10, measured +1.18 nested on Wojood test):
    L = mean_head(LUL) + p · Var_head(LUL)
    LUL = 0.4·CE + 0.2·Dice + 0.2·Tversky(α=0.5) + 0.2·Focal(γ=2)
Dice/Tversky/Focal include the O class — A10 does not specify an exclusion.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .columns import ENTITY_TYPES

N_TYPES = len(ENTITY_TYPES)


class MultiHeadTagger(nn.Module):
    """Both taggers share the interface run_training relies on:
    forward → logits; compute_loss(logits, labels) → (loss, parts);
    decode(logits) → tag ids [B, T, 21] with 0=O 1=B 2=I."""

    def __init__(
        self,
        encoder,
        dropout: float = 0.1,
        loss_lambdas: tuple[float, float, float, float] = (0.4, 0.2, 0.2, 0.2),
        tversky_alpha: float = 0.5,
        focal_gamma: float = 2.0,
        var_penalty: float = 5.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.Linear(encoder.config.hidden_size, N_TYPES * 3)
        self.loss_lambdas = loss_lambdas
        self.tversky_alpha = tversky_alpha
        self.focal_gamma = focal_gamma
        self.var_penalty = var_penalty

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        logits = self.heads(self.dropout(hidden))
        return logits.view(logits.shape[0], logits.shape[1], N_TYPES, 3)

    def compute_loss(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        return unified_loss(
            logits, labels, self.loss_lambdas,
            self.tversky_alpha, self.focal_gamma, self.var_penalty,
        )

    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=-1)


class SigmoidTagger(nn.Module):
    """B22's single-head multilabel variant (design §3.1 A/B candidate).

    One Linear(hidden, 43): a B- and I- output per type plus one O output,
    each sigmoid-activated; focal BCE (α=0.75, γ=1.0 per B22). Decode per
    type column: max(B, I) over threshold wins, else O — the O output only
    participates in the loss. Scored 92.30 vs 88.4 for the multi-task
    original on nested Wojood test (B22)."""

    def __init__(
        self,
        encoder,
        dropout: float = 0.1,
        focal_alpha: float = 0.75,
        focal_gamma: float = 1.0,
        threshold: float = 0.5,
    ):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(encoder.config.hidden_size, 2 * N_TYPES + 1)
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.threshold = threshold

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        return self.head(self.dropout(hidden))  # [B, T, 43]

    def _targets(self, labels: torch.Tensor) -> torch.Tensor:
        """labels [N, 21] with 0/1/2 → multilabel bits [N, 43]."""
        b_bits = (labels == 1).float()
        i_bits = (labels == 2).float()
        o_bit = (labels == 0).all(dim=-1, keepdim=True).float()
        return torch.cat([b_bits, i_bits, o_bit], dim=-1)

    def compute_loss(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        mask = labels[..., 0] != -100
        lg = logits[mask].float()  # [N, 43]
        targets = self._targets(labels[mask])
        p = torch.sigmoid(lg)
        pt = torch.where(targets > 0, p, 1 - p).clamp_min(1e-7)
        alpha_t = torch.where(
            targets > 0,
            torch.full_like(p, self.focal_alpha),
            torch.full_like(p, 1 - self.focal_alpha),
        )
        loss = (alpha_t * (1 - pt) ** self.focal_gamma * -pt.log()).mean()
        return loss, {"focal_bce": float(loss.detach())}

    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits.float())
        b_probs = p[..., :N_TYPES]
        i_probs = p[..., N_TYPES : 2 * N_TYPES]
        best = torch.maximum(b_probs, i_probs)
        tag_ids = torch.where(b_probs >= i_probs, 1, 2)
        return torch.where(best > self.threshold, tag_ids, torch.zeros_like(tag_ids))


def unified_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    lambdas: tuple[float, float, float, float] = (0.4, 0.2, 0.2, 0.2),
    tversky_alpha: float = 0.5,
    focal_gamma: float = 2.0,
    var_penalty: float = 5.0,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    """logits [B, T, 21, 3]; labels [B, T, 21] with -100 on unlabeled positions.

    Labeled positions are identical across heads (first subtokens), so one mask
    serves all 21. Computed in fp32 even under autocast — the Dice/Tversky sums
    are batch-level reductions that lose precision in fp16.
    """
    mask = labels[..., 0] != -100
    lg = logits[mask].float()  # [N, 21, 3]
    y = labels[mask]  # [N, 21]
    if lg.shape[0] == 0:
        raise ValueError("no labeled positions in batch")
    logp = F.log_softmax(lg, dim=-1)
    p = logp.exp()
    y1 = F.one_hot(y, num_classes=3).float()  # [N, 21, 3]

    ce = -logp.gather(-1, y.unsqueeze(-1)).squeeze(-1).mean(dim=0)  # [21]
    inter = (p * y1).sum(dim=0)  # [21, 3]
    dice = 1 - ((2 * inter + eps) / ((p * p).sum(dim=0) + y1.sum(dim=0) + eps)).mean(dim=-1)
    fp = (p * (1 - y1)).sum(dim=0)
    fn = ((1 - p) * y1).sum(dim=0)
    tversky = 1 - (
        (inter + eps) / (inter + tversky_alpha * fp + (1 - tversky_alpha) * fn + eps)
    ).mean(dim=-1)
    pt = p.gather(-1, y.unsqueeze(-1)).squeeze(-1).clamp_min(eps)  # [N, 21]
    focal = ((1 - pt) ** focal_gamma * -pt.log()).mean(dim=0)  # [21]

    l_ce, l_di, l_tv, l_fl = lambdas
    per_head = l_ce * ce + l_di * dice + l_tv * tversky + l_fl * focal  # [21]
    variance = per_head.var(unbiased=False)
    total = per_head.mean() + var_penalty * variance
    parts = {
        "ce": float(ce.mean().detach()),
        "dice": float(dice.mean().detach()),
        "tversky": float(tversky.mean().detach()),
        "focal": float(focal.mean().detach()),
        "head_var": float(variance.detach()),
    }
    return total, parts
