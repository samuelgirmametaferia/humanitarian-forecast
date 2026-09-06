from __future__ import annotations

import math

import torch
from torch import nn


class CandidateCenterRegressor(nn.Module):
    """Rank spatial candidates and learn a bounded correction to their soft center."""

    def __init__(
        self,
        event_dim=25,
        candidate_dim=28,
        sequence_length=16,
        countries=121,
        conflicts=1014,
        d_model=128,
        heads=4,
        layers=3,
        ff_dim=384,
        dropout=0.1,
        maximum_residual=0.5,
    ):
        super().__init__()
        self.config = dict(
            event_dim=event_dim,
            candidate_dim=candidate_dim,
            sequence_length=sequence_length,
            countries=countries,
            conflicts=conflicts,
            d_model=d_model,
            heads=heads,
            layers=layers,
            ff_dim=ff_dim,
            dropout=dropout,
            maximum_residual=maximum_residual,
        )
        self.maximum_residual = maximum_residual
        self.event_norm = nn.LayerNorm(event_dim)
        self.event_projection = nn.Linear(event_dim, d_model)
        self.position = nn.Parameter(torch.randn(1, sequence_length, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model,
            heads,
            ff_dim,
            dropout,
            "gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.context = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU())
        self.country = nn.Embedding(countries, d_model)
        self.conflict = nn.Embedding(conflicts, d_model)
        self.candidate = nn.Sequential(
            nn.LayerNorm(candidate_dim),
            nn.Linear(candidate_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.bias = nn.Sequential(nn.Linear(candidate_dim, 64), nn.GELU(), nn.Linear(64, 1))
        self.residual = nn.Sequential(
            nn.LayerNorm(d_model * 2 + 4),
            nn.Linear(d_model * 2 + 4, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, events, candidates, coordinates, valid, country, conflict):
        event_mask = events[:, :, 0] > 0.5
        encoded = self.event_projection(self.event_norm(events)) * math.sqrt(self.config["d_model"])
        encoded = self.encoder(encoded + self.position, src_key_padding_mask=~event_mask)
        context = self.context(encoded[:, -1]) + self.country(country) + self.conflict(conflict)
        candidate = self.candidate(candidates)
        logits = (
            (candidate * context[:, None]).sum(-1) / math.sqrt(self.config["d_model"])
            + self.bias(candidates).squeeze(-1)
        ).masked_fill(~valid, -1e9)
        probability = logits.softmax(-1)
        base_center = (probability[:, :, None] * coordinates).sum(1)
        pooled_candidate = (probability[:, :, None] * candidate).sum(1)
        spread = torch.sqrt(
            (probability[:, :, None] * (coordinates - base_center[:, None]).square()).sum(1)
            + 1e-8
        )
        correction = self.maximum_residual * torch.tanh(
            self.residual(torch.cat((context, pooled_candidate, base_center, spread), dim=-1))
        )
        return logits, base_center + correction, correction


def center_loss(logits, center, correction, target, label, cross_entropy_weight, residual_weight):
    distance = torch.sqrt((center - target).square().sum(-1) + 1e-8).mean()
    cross_entropy = nn.functional.cross_entropy(logits, label)
    residual_penalty = correction.square().sum(-1).mean()
    loss = distance + cross_entropy_weight * cross_entropy + residual_weight * residual_penalty
    return loss, distance, cross_entropy, residual_penalty
