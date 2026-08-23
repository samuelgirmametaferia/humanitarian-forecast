from __future__ import annotations

import math
import torch
from torch import nn


class ProbabilisticLocationTransformer(nn.Module):
    """Predict a coarse next-event center and isotropic spatial uncertainty."""

    def __init__(
        self, feature_dim: int = 15, sequence_length: int = 16,
        d_model: int = 128, heads: int = 8, layers: int = 4,
        ff_dim: int = 384, dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.config = dict(
            feature_dim=feature_dim, sequence_length=sequence_length, d_model=d_model,
            heads=heads, layers=layers, ff_dim=ff_dim, dropout=dropout,
        )
        self.input_norm = nn.LayerNorm(feature_dim)
        self.projection = nn.Linear(feature_dim, d_model)
        self.position = nn.Parameter(torch.empty(1, sequence_length, d_model))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=ff_dim, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model, 3),
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        valid = features[:, :, 0] > 0.5
        x = self.projection(self.input_norm(features)) * math.sqrt(self.config["d_model"])
        x = x + self.position[:, : x.shape[1]]
        encoded = self.encoder(x, src_key_padding_mask=~valid)
        # Sequences are left-padded, so the newest valid event is always last.
        last = encoded[:, -1]
        output = self.head(last)
        center = output[:, :2]
        sigma = nn.functional.softplus(output[:, 2]) + 0.01
        return center, sigma


def gaussian_location_loss(
    center: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor,
) -> torch.Tensor:
    squared_error = ((center - target) ** 2).sum(dim=1)
    return (0.5 * squared_error / sigma.square() + 2.0 * sigma.log()).mean()
