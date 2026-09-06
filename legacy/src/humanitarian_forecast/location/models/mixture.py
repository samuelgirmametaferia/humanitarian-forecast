from __future__ import annotations

import math
import torch
from torch import nn


class MixtureLocationTransformer(nn.Module):
    """Multimodal next-location density: K centers, probabilities, and radii."""

    def __init__(self, feature_dim=15, sequence_length=16, components=5,
                 d_model=160, heads=8, layers=5, ff_dim=512, dropout=0.15):
        super().__init__()
        self.config = dict(feature_dim=feature_dim, sequence_length=sequence_length,
                           components=components, d_model=d_model, heads=heads,
                           layers=layers, ff_dim=ff_dim, dropout=dropout)
        self.components = components
        self.norm = nn.LayerNorm(feature_dim)
        self.projection = nn.Linear(feature_dim, d_model)
        self.position = nn.Parameter(torch.randn(1, sequence_length, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, heads, ff_dim, dropout, "gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model),
                                  nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(d_model, components * 4))

    def forward(self, features):
        valid = features[:, :, 0] > 0.5
        x = self.projection(self.norm(features)) * math.sqrt(self.config["d_model"])
        x = self.encoder(x + self.position[:, :x.shape[1]], src_key_padding_mask=~valid)
        out = self.head(x[:, -1]).view(len(x), self.components, 4)
        logits = out[:, :, 0]
        centers = out[:, :, 1:3]
        sigmas = nn.functional.softplus(out[:, :, 3]) + 0.01
        return logits, centers, sigmas


def mixture_nll(logits, centers, sigmas, target):
    error2 = ((centers - target[:, None, :]) ** 2).sum(-1)
    log_component = -math.log(2 * math.pi) - 2 * sigmas.log() - error2 / (2 * sigmas.square())
    return -torch.logsumexp(torch.log_softmax(logits, -1) + log_component, -1).mean()

