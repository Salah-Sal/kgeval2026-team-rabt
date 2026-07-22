"""RE classifier: encoder + FFNN over the two start-marker hidden states
(design §4.1, F17/G09). Trained 41-way with no_relation; the negative class is
masked at inference (§4.5) — masking lives in re_train.predict, not here.
"""

from __future__ import annotations

import torch
from torch import nn


class MarkerClassifier(nn.Module):
    def __init__(self, encoder, n_labels: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        hidden = encoder.config.hidden_size
        self.ffnn = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_labels),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        subj_pos: torch.Tensor,
        obj_pos: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        feats = torch.cat([hidden[rows, subj_pos], hidden[rows, obj_pos]], dim=-1)
        return self.ffnn(feats)
