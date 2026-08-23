from __future__ import annotations

import math

import torch
from torch import nn


class GeoFusionCandidateRankerV2(nn.Module):
    """Sharper motion-aware geo ranker with explicit recent-state geometry.

    V1 proved candidate/history cross-attention was trainable but became too
    diffuse.  V2 preserves v9's strong discriminative dot-product path while
    adding: padding-safe temporal convolutions, latest-event + attention + mean
    pooling, candidate/history cross-attention, candidate competition, and an
    explicit recent-motion alignment bias.
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
        candidate_layers: int = 1,
        ff_dim: int = 512,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        if event_dim < 25:
            raise ValueError("GeoFusion V2 requires motion-v3 history (25 event channels)")
        if candidate_dim < 9:
            raise ValueError("candidate feature contract is too small")

        em = torch.as_tensor(event_mean, dtype=torch.float32).reshape(event_dim)
        es = torch.as_tensor(event_std, dtype=torch.float32).reshape(event_dim).clamp_min(1e-4)
        cm = torch.as_tensor(candidate_mean, dtype=torch.float32).reshape(candidate_dim)
        cs = torch.as_tensor(candidate_std, dtype=torch.float32).reshape(candidate_dim).clamp_min(1e-4)
        self.register_buffer("event_mean", em)
        self.register_buffer("event_std", es)
        self.register_buffer("candidate_mean", cm)
        self.register_buffer("candidate_std", cs)
        self.config = {
            "event_dim": event_dim,
            "candidate_dim": candidate_dim,
            "sequence_length": sequence_length,
            "countries": countries,
            "conflicts": conflicts,
            "event_mean": em.tolist(),
            "event_std": es.tolist(),
            "candidate_mean": cm.tolist(),
            "candidate_std": cs.tolist(),
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
            nn.Conv1d(d_model, d_model, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, 3, padding=2, dilation=2),
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
        self.event_norm = nn.LayerNorm(d_model)
        self.pool_query = nn.Parameter(torch.empty(d_model))
        nn.init.normal_(self.pool_query, std=0.02)
        self.pool_projection = nn.Sequential(
            nn.LayerNorm(3 * d_model),
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
        )
        self.country = nn.Embedding(countries, d_model)
        self.conflict = nn.Embedding(conflicts, d_model)
        self.identity_projection = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model))

        self.candidate_projection = nn.Sequential(
            nn.Linear(candidate_dim, d_model), nn.GELU(), nn.LayerNorm(d_model), nn.Linear(d_model, d_model)
        )
        self.cross_attention = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
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
        self.candidate_context = nn.TransformerEncoder(candidate_block, num_layers=candidate_layers)

        # Retain a v9-like dot-product path so cross-attention cannot force the
        # model into an overly smooth probability distribution.
        self.candidate_dot = nn.Linear(d_model, d_model, bias=False)
        self.context_dot = nn.Linear(d_model, d_model, bias=False)
        self.learned_score = nn.Sequential(
            nn.LayerNorm(3 * d_model),
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.candidate_bias = nn.Sequential(
            nn.Linear(candidate_dim, 96), nn.GELU(), nn.Linear(96, 1)
        )
        # Candidate offset x/y + latest step x/y + six derived directional
        # relations.  This exposes motion directly instead of requiring a deep
        # network to rediscover elementary vector geometry.
        self.motion_bias = nn.Sequential(
            nn.LayerNorm(10), nn.Linear(10, 64), nn.GELU(), nn.Linear(64, 1)
        )
        self.learned_mix = nn.Parameter(torch.tensor(-1.0))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.25)))
        self.displacement_head = nn.Sequential(
            nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1), nn.Softplus()
        )

    @staticmethod
    def _masked_mean(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        w = valid.to(x.dtype).unsqueeze(-1)
        return (x * w).sum(1) / w.sum(1).clamp_min(1.0)

    def _attention_pool(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        score = (x * self.pool_query).sum(-1) / math.sqrt(self.d_model)
        score = score.masked_fill(~valid, -1e9)
        weight = torch.softmax(score, dim=-1)
        return (x * weight.unsqueeze(-1)).sum(1)

    @staticmethod
    def _motion_features(events: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        candidate_xy = candidates[:, :, 1:3]
        last_step = events[:, -1, 19:21].unsqueeze(1).expand(-1, candidates.shape[1], -1)
        c_norm = torch.linalg.vector_norm(candidate_xy, dim=-1, keepdim=True)
        s_norm = torch.linalg.vector_norm(last_step, dim=-1, keepdim=True)
        dot = (candidate_xy * last_step).sum(-1, keepdim=True)
        cosine = dot / (c_norm * s_norm).clamp_min(1e-5)
        cross = (candidate_xy[..., 0:1] * last_step[..., 1:2] - candidate_xy[..., 1:2] * last_step[..., 0:1])
        projection = dot / s_norm.clamp_min(1e-5)
        radial_delta = c_norm - s_norm
        return torch.cat(
            [candidate_xy, last_step, c_norm, s_norm, cosine, cross, projection, radial_delta], dim=-1
        )

    def forward(
        self,
        events: torch.Tensor,
        candidates: torch.Tensor,
        valid: torch.Tensor,
        country: torch.Tensor,
        conflict: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        event_valid = events[:, :, 0] > 0.5
        standardized = (events - self.event_mean) / self.event_std
        standardized = standardized.masked_fill(~event_valid.unsqueeze(-1), 0.0)
        hidden = self.event_projection(standardized)
        local = self.local_temporal(hidden.transpose(1, 2)).transpose(1, 2)
        local = local.masked_fill(~event_valid.unsqueeze(-1), 0.0)
        hidden = hidden + local + self.position[:, : events.shape[1]]
        hidden = self.event_encoder(hidden, src_key_padding_mask=~event_valid)
        hidden = self.event_norm(hidden)

        latest = hidden[:, -1]
        mean = self._masked_mean(hidden, event_valid)
        attended = self._attention_pool(hidden, event_valid)
        context = self.pool_projection(torch.cat([latest, attended, mean], dim=-1))
        identity = self.identity_projection(self.country(country) + self.conflict(conflict))
        context = context + identity

        standardized_candidates = (candidates - self.candidate_mean) / self.candidate_std
        candidate = self.candidate_projection(standardized_candidates)
        cross, _ = self.cross_attention(
            candidate + context.unsqueeze(1), hidden, hidden,
            key_padding_mask=~event_valid, need_weights=False,
        )
        candidate = self.cross_norm(candidate + cross)
        candidate = self.candidate_context(candidate, src_key_padding_mask=~valid)

        context_expanded = context.unsqueeze(1).expand_as(candidate)
        interaction = candidate * context_expanded
        learned = self.learned_score(torch.cat([candidate, context_expanded, interaction], dim=-1)).squeeze(-1)
        dot = (
            self.candidate_dot(candidate) * self.context_dot(context_expanded)
        ).sum(-1) / math.sqrt(self.d_model)
        static_bias = self.candidate_bias(standardized_candidates).squeeze(-1)
        motion_bias = self.motion_bias(self._motion_features(events, candidates)).squeeze(-1)
        mix = torch.sigmoid(self.learned_mix)
        scale = self.logit_scale.exp().clamp(0.5, 4.0)
        logits = scale * ((1.0 - mix) * dot + mix * learned + static_bias + motion_bias)
        logits = logits.masked_fill(~valid, -1e9)

        pooled = self._masked_mean(candidate, valid)
        displacement = self.displacement_head(torch.cat([context, pooled], dim=-1)).squeeze(-1)
        return logits, displacement
