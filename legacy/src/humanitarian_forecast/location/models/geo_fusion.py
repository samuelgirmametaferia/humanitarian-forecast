from __future__ import annotations

import math

import torch
from torch import nn


class GeoFusionCandidateRanker(nn.Module):
    """Motion-aware candidate/history fusion model for broad-area forecasting.

    Compared with the promoted v9 dot-product ranker, every candidate is allowed
    to cross-attend to the complete pre-cutoff event history and then exchange
    information with the other candidates.  Static/dynamic candidate features
    and event histories use training-partition featurewise normalization so
    absolute magnitudes are retained.
    """

    def __init__(
        self,
        *,
        event_dim: int,
        candidate_dim: int,
        sequence_length: int,
        countries: int,
        conflicts: int,
        event_mean: list[float] | torch.Tensor,
        event_std: list[float] | torch.Tensor,
        candidate_mean: list[float] | torch.Tensor,
        candidate_std: list[float] | torch.Tensor,
        d_model: int = 192,
        heads: int = 6,
        event_layers: int = 4,
        candidate_layers: int = 2,
        ff_dim: int = 512,
        dropout: float = 0.12,
    ) -> None:
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")

        event_mean_t = torch.as_tensor(event_mean, dtype=torch.float32).reshape(event_dim)
        event_std_t = torch.as_tensor(event_std, dtype=torch.float32).reshape(event_dim).clamp_min(1e-4)
        candidate_mean_t = torch.as_tensor(candidate_mean, dtype=torch.float32).reshape(candidate_dim)
        candidate_std_t = torch.as_tensor(candidate_std, dtype=torch.float32).reshape(candidate_dim).clamp_min(1e-4)
        self.register_buffer("event_mean", event_mean_t)
        self.register_buffer("event_std", event_std_t)
        self.register_buffer("candidate_mean", candidate_mean_t)
        self.register_buffer("candidate_std", candidate_std_t)

        self.config = {
            "event_dim": event_dim,
            "candidate_dim": candidate_dim,
            "sequence_length": sequence_length,
            "countries": countries,
            "conflicts": conflicts,
            "event_mean": event_mean_t.tolist(),
            "event_std": event_std_t.tolist(),
            "candidate_mean": candidate_mean_t.tolist(),
            "candidate_std": candidate_std_t.tolist(),
            "d_model": d_model,
            "heads": heads,
            "event_layers": event_layers,
            "candidate_layers": candidate_layers,
            "ff_dim": ff_dim,
            "dropout": dropout,
        }
        self.d_model = d_model

        self.event_projection = nn.Linear(event_dim, d_model)
        self.position = nn.Parameter(torch.empty(1, sequence_length, d_model))
        nn.init.normal_(self.position, std=0.02)
        self.local_temporal = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=2, dilation=2),
            nn.GELU(),
        )
        event_block = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.event_encoder = nn.TransformerEncoder(event_block, num_layers=event_layers)
        self.event_out_norm = nn.LayerNorm(d_model)

        self.country = nn.Embedding(countries, d_model)
        self.conflict = nn.Embedding(conflicts, d_model)
        self.context_projection = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )

        self.candidate_encoder = nn.Sequential(
            nn.Linear(candidate_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )
        self.cross_attention = nn.MultiheadAttention(
            d_model, heads, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(d_model)

        candidate_block = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.candidate_encoder_context = nn.TransformerEncoder(
            candidate_block, num_layers=candidate_layers
        )

        self.score = nn.Sequential(
            nn.LayerNorm(3 * d_model),
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.displacement_head = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Softplus(),
        )

    @staticmethod
    def _masked_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        weights = valid.to(values.dtype).unsqueeze(-1)
        return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def forward(
        self,
        events: torch.Tensor,
        candidates: torch.Tensor,
        valid: torch.Tensor,
        country: torch.Tensor,
        conflict: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        event_valid = events[:, :, 0] > 0.5
        standardized_events = (events - self.event_mean) / self.event_std
        event_hidden = self.event_projection(standardized_events)
        local = self.local_temporal(event_hidden.transpose(1, 2)).transpose(1, 2)
        event_hidden = event_hidden + local + self.position[:, : events.shape[1]]
        event_hidden = self.event_encoder(event_hidden, src_key_padding_mask=~event_valid)
        event_hidden = self.event_out_norm(event_hidden)
        global_context = self._masked_mean(event_hidden, event_valid)
        identity = self.country(country) + self.conflict(conflict)
        global_context = self.context_projection(global_context + identity)

        standardized_candidates = (candidates - self.candidate_mean) / self.candidate_std
        candidate_hidden = self.candidate_encoder(standardized_candidates)
        candidate_hidden = candidate_hidden + global_context.unsqueeze(1)
        attended, _ = self.cross_attention(
            query=candidate_hidden,
            key=event_hidden,
            value=event_hidden,
            key_padding_mask=~event_valid,
            need_weights=False,
        )
        candidate_hidden = self.cross_norm(candidate_hidden + attended)
        candidate_hidden = self.candidate_encoder_context(
            candidate_hidden, src_key_padding_mask=~valid
        )

        context = global_context.unsqueeze(1).expand(-1, candidate_hidden.shape[1], -1)
        interaction = candidate_hidden * context
        logits = self.score(torch.cat([candidate_hidden, context, interaction], dim=-1)).squeeze(-1)
        logits = logits.masked_fill(~valid, -1e9)

        pooled_candidates = self._masked_mean(candidate_hidden, valid)
        displacement_thousand_km = self.displacement_head(
            torch.cat([global_context, pooled_candidates], dim=-1)
        ).squeeze(-1)
        return logits, displacement_thousand_km


def spatial_distribution_loss(
    logits: torch.Tensor,
    coordinates: torch.Tensor,
    target: torch.Tensor,
    hard_label: torch.Tensor,
    valid: torch.Tensor,
    *,
    hard_weight: float = 0.35,
    soft50_weight: float = 0.75,
    soft100_weight: float = 0.45,
    distance_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Distance-aware broad-area objective.

    Coordinates are expressed in thousands of kilometres by the existing data
    contract.  Soft targets therefore use 0.05/0.10 native-unit temperatures.
    """
    distances = torch.linalg.vector_norm(coordinates - target[:, None], dim=-1)
    masked_distances = distances.masked_fill(~valid, float("inf"))
    probability = torch.softmax(logits, dim=-1)

    hard = nn.functional.cross_entropy(logits, hard_label)

    def soft_ce(tau: float) -> torch.Tensor:
        scores = -masked_distances / tau
        scores = scores.masked_fill(~valid, -1e9)
        soft_target = torch.softmax(scores, dim=-1)
        return -(soft_target * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()

    soft50 = soft_ce(0.05)
    soft100 = soft_ce(0.10)
    expected_distance = (probability * distances.masked_fill(~valid, 0.0)).sum(dim=-1).mean()
    total = (
        hard_weight * hard
        + soft50_weight * soft50
        + soft100_weight * soft100
        + distance_weight * expected_distance
    )
    return total, {
        "hard_ce": hard,
        "soft50_ce": soft50,
        "soft100_ce": soft100,
        "expected_distance": expected_distance,
    }
