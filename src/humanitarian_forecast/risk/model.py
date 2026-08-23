from __future__ import annotations

import math
import torch
from torch import nn


class TemporalRiskTransformer(nn.Module):
    """Entity-agnostic temporal encoder shared by countries and coarse areas."""

    def __init__(
        self,
        feature_dim: int,
        sequence_length: int,
        d_model: int = 128,
        heads: int = 8,
        layers: int = 4,
        ff_dim: int = 384,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.config = dict(
            feature_dim=feature_dim, sequence_length=sequence_length, d_model=d_model,
            heads=heads, layers=layers, ff_dim=ff_dim, dropout=dropout,
        )
        self.input_norm = nn.LayerNorm(feature_dim)
        self.input_projection = nn.Linear(feature_dim, d_model)
        self.position = nn.Parameter(torch.empty(1, sequence_length, d_model))
        nn.init.normal_(self.position, std=0.02)
        block = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=ff_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(self.input_norm(features))
        x = x * math.sqrt(self.config["d_model"]) + self.position[:, : x.shape[1]]
        return self.head(self.encoder(x)[:, -1])

