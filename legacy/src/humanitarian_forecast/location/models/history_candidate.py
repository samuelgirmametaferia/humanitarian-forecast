from __future__ import annotations

import math
import torch
from torch import nn


class HistoryCandidateRanker(nn.Module):
    """Score each observed historical location as the next destination."""

    def __init__(self, feature_dim=19, sequence_length=16, d_model=96,
                 heads=4, layers=3, ff_dim=256, dropout=.10):
        super().__init__()
        self.config = dict(feature_dim=feature_dim, sequence_length=sequence_length,
                           d_model=d_model, heads=heads, layers=layers,
                           ff_dim=ff_dim, dropout=dropout)
        self.norm = nn.LayerNorm(feature_dim)
        self.projection = nn.Linear(feature_dim, d_model)
        self.position = nn.Parameter(torch.randn(1, sequence_length, d_model) * .02)
        layer = nn.TransformerEncoderLayer(
            d_model, heads, ff_dim, dropout, "gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.context = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU())
        self.candidate = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU())
        self.bias = nn.Sequential(nn.Linear(feature_dim, 48), nn.GELU(), nn.Linear(48, 1))

    def forward(self, features):
        valid = features[:, :, 0] > .5
        encoded = self.projection(self.norm(features)) * math.sqrt(self.config["d_model"])
        encoded = self.encoder(encoded + self.position, src_key_padding_mask=~valid)
        context = self.context(encoded[:, -1])
        candidate = self.candidate(encoded)
        logits = (candidate * context[:, None]).sum(-1) / math.sqrt(self.config["d_model"])
        logits = logits + self.bias(features).squeeze(-1)
        return logits.masked_fill(~valid, -1e9)


def candidate_distances(features, target):
    return torch.linalg.vector_norm(features[:, :, 1:3] - target[:, None, :], dim=-1)


def ranking_loss(logits, features, target, expected_distance_weight=.25):
    distance = candidate_distances(features, target)
    valid = features[:, :, 0] > .5
    nearest = distance.masked_fill(~valid, float("inf")).argmin(-1)
    cross_entropy = nn.functional.cross_entropy(logits, nearest)
    expected_distance = (logits.softmax(-1) * distance.masked_fill(~valid, 0)).sum(-1).mean()
    return cross_entropy + expected_distance_weight * expected_distance, cross_entropy, expected_distance
