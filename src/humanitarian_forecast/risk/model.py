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


class MultiscaleTemporalRiskModel(nn.Module):
    """Multiscale humanitarian-risk encoder with magnitude-preserving normalization.

    The legacy model normalizes across feature channels independently at every
    timestep. For sparse count-like humanitarian signals that can erase useful
    absolute magnitude. This model instead stores featurewise statistics learned
    only from the training partition, combines local temporal convolutions with a
    Transformer, and adds explicit short/medium-window summaries.
    """

    def __init__(
        self,
        feature_dim: int,
        sequence_length: int,
        feature_mean: list[float] | torch.Tensor,
        feature_std: list[float] | torch.Tensor,
        d_model: int = 96,
        heads: int = 4,
        layers: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.12,
        summary_dim: int = 96,
    ) -> None:
        super().__init__()
        mean = torch.as_tensor(feature_mean, dtype=torch.float32).reshape(feature_dim)
        std = torch.as_tensor(feature_std, dtype=torch.float32).reshape(feature_dim).clamp_min(1e-3)
        self.register_buffer("feature_mean", mean)
        self.register_buffer("feature_std", std)
        self.config = dict(
            feature_dim=feature_dim,
            sequence_length=sequence_length,
            feature_mean=mean.tolist(),
            feature_std=std.tolist(),
            d_model=d_model,
            heads=heads,
            layers=layers,
            ff_dim=ff_dim,
            dropout=dropout,
            summary_dim=summary_dim,
        )
        self.d_model = d_model
        self.input_projection = nn.Linear(feature_dim, d_model)
        self.position = nn.Parameter(torch.empty(1, sequence_length, d_model))
        nn.init.normal_(self.position, std=0.02)
        self.local_encoder = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=2, dilation=2),
            nn.GELU(),
        )
        block = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=layers)
        self.attention_query = nn.Parameter(torch.empty(d_model))
        nn.init.normal_(self.attention_query, std=0.02)

        # Six summaries per feature: 7d, 14d, full mean, max, 7d delta, std.
        self.summary_encoder = nn.Sequential(
            nn.Linear(6 * feature_dim, summary_dim),
            nn.GELU(),
            nn.LayerNorm(summary_dim),
        )
        trunk_dim = 4 * d_model + summary_dim
        self.trunk = nn.Sequential(
            nn.LayerNorm(trunk_dim),
            nn.Linear(trunk_dim, 256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, 128),
            nn.GELU(),
        )
        self.intensity_head = nn.Sequential(nn.Linear(128, 1), nn.Softplus())
        self.escalation_head = nn.Linear(128, 1)

    @staticmethod
    def _window_mean(features: torch.Tensor, days: int) -> torch.Tensor:
        days = min(days, features.shape[1])
        return features[:, -days:].mean(dim=1)

    def _summary_features(self, features: torch.Tensor) -> torch.Tensor:
        recent_7 = self._window_mean(features, 7)
        recent_14 = self._window_mean(features, 14)
        previous_7 = (
            features[:, -14:-7].mean(dim=1)
            if features.shape[1] >= 14
            else features[:, : max(1, features.shape[1] - 7)].mean(dim=1)
        )
        return torch.cat(
            [
                recent_7,
                recent_14,
                features.mean(dim=1),
                features.amax(dim=1),
                recent_7 - previous_7,
                features.std(dim=1),
            ],
            dim=1,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        standardized = (features - self.feature_mean) / self.feature_std
        hidden = self.input_projection(standardized)
        hidden = hidden + self.position[:, : hidden.shape[1]]
        local = self.local_encoder(hidden.transpose(1, 2)).transpose(1, 2)
        hidden = self.encoder(hidden + local)

        attention = torch.softmax(
            (hidden * self.attention_query).sum(dim=-1) / math.sqrt(self.d_model),
            dim=1,
        )
        attended = (hidden * attention.unsqueeze(-1)).sum(dim=1)
        temporal = torch.cat(
            [attended, hidden[:, -1], hidden.mean(dim=1), hidden.amax(dim=1)],
            dim=1,
        )
        summary = self.summary_encoder(self._summary_features(features))
        representation = self.trunk(torch.cat([temporal, summary], dim=1))
        intensity = self.intensity_head(representation)
        escalation = self.escalation_head(representation)
        return torch.cat([intensity, escalation], dim=1)



class ContextFusionRiskModel(nn.Module):
    """Coarse humanitarian-risk model with separated dynamic/static pathways.

    The first ``dynamic_dim`` channels are treated as causal time-varying
    histories. Remaining channels are assumed static over the sequence and are
    encoded once from the final timestep. A bounded FiLM-style gate lets broad
    population/accessibility context modulate, but not replace, the dynamic
    representation.
    """

    def __init__(
        self,
        feature_dim: int,
        dynamic_dim: int,
        sequence_length: int,
        dynamic_mean: list[float] | torch.Tensor,
        dynamic_std: list[float] | torch.Tensor,
        static_mean: list[float] | torch.Tensor,
        static_std: list[float] | torch.Tensor,
        d_model: int = 96,
        heads: int = 4,
        layers: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.12,
        summary_dim: int = 96,
        static_hidden: int = 64,
    ) -> None:
        super().__init__()
        if not 0 < dynamic_dim < feature_dim:
            raise ValueError("dynamic_dim must split dynamic and static channels")
        static_dim = feature_dim - dynamic_dim
        d_mean = torch.as_tensor(dynamic_mean, dtype=torch.float32).reshape(dynamic_dim)
        d_std = torch.as_tensor(dynamic_std, dtype=torch.float32).reshape(dynamic_dim).clamp_min(1e-3)
        s_mean = torch.as_tensor(static_mean, dtype=torch.float32).reshape(static_dim)
        s_std = torch.as_tensor(static_std, dtype=torch.float32).reshape(static_dim).clamp_min(1e-3)
        self.register_buffer("dynamic_mean", d_mean)
        self.register_buffer("dynamic_std", d_std)
        self.register_buffer("static_mean", s_mean)
        self.register_buffer("static_std", s_std)
        self.config = dict(
            feature_dim=feature_dim,
            dynamic_dim=dynamic_dim,
            sequence_length=sequence_length,
            dynamic_mean=d_mean.tolist(),
            dynamic_std=d_std.tolist(),
            static_mean=s_mean.tolist(),
            static_std=s_std.tolist(),
            d_model=d_model,
            heads=heads,
            layers=layers,
            ff_dim=ff_dim,
            dropout=dropout,
            summary_dim=summary_dim,
            static_hidden=static_hidden,
        )
        self.dynamic_dim = dynamic_dim
        self.d_model = d_model
        self.input_projection = nn.Linear(dynamic_dim, d_model)
        self.position = nn.Parameter(torch.empty(1, sequence_length, d_model))
        nn.init.normal_(self.position, std=0.02)
        self.local_encoder = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=2, dilation=2),
            nn.GELU(),
        )
        block = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=layers)
        self.attention_query = nn.Parameter(torch.empty(d_model))
        nn.init.normal_(self.attention_query, std=0.02)
        self.summary_encoder = nn.Sequential(
            nn.Linear(6 * dynamic_dim, summary_dim),
            nn.GELU(),
            nn.LayerNorm(summary_dim),
        )
        trunk_dim = 4 * d_model + summary_dim
        self.dynamic_trunk = nn.Sequential(
            nn.LayerNorm(trunk_dim),
            nn.Linear(trunk_dim, 256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, 128),
            nn.GELU(),
        )
        self.static_encoder = nn.Sequential(
            nn.Linear(static_dim, static_hidden),
            nn.GELU(),
            nn.LayerNorm(static_hidden),
            nn.Dropout(0.10),
            nn.Linear(static_hidden, static_hidden),
            nn.GELU(),
        )
        self.static_residual = nn.Linear(static_hidden, 128)
        self.film = nn.Linear(static_hidden, 256)
        self.fusion_norm = nn.LayerNorm(128)
        self.intensity_head = nn.Sequential(nn.Linear(128, 1), nn.Softplus())
        self.escalation_head = nn.Linear(128, 1)

    @staticmethod
    def _window_mean(features: torch.Tensor, days: int) -> torch.Tensor:
        days = min(days, features.shape[1])
        return features[:, -days:].mean(dim=1)

    def _summary_features(self, dynamic: torch.Tensor) -> torch.Tensor:
        recent_7 = self._window_mean(dynamic, 7)
        recent_14 = self._window_mean(dynamic, 14)
        previous_7 = (
            dynamic[:, -14:-7].mean(dim=1)
            if dynamic.shape[1] >= 14
            else dynamic[:, : max(1, dynamic.shape[1] - 7)].mean(dim=1)
        )
        return torch.cat(
            [
                recent_7,
                recent_14,
                dynamic.mean(dim=1),
                dynamic.amax(dim=1),
                recent_7 - previous_7,
                dynamic.std(dim=1),
            ],
            dim=1,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        dynamic = features[..., : self.dynamic_dim]
        static = features[:, -1, self.dynamic_dim :]
        standardized = (dynamic - self.dynamic_mean) / self.dynamic_std
        hidden = self.input_projection(standardized) + self.position[:, : dynamic.shape[1]]
        local = self.local_encoder(hidden.transpose(1, 2)).transpose(1, 2)
        hidden = self.encoder(hidden + local)
        attention = torch.softmax(
            (hidden * self.attention_query).sum(dim=-1) / math.sqrt(self.d_model), dim=1
        )
        attended = (hidden * attention.unsqueeze(-1)).sum(dim=1)
        temporal = torch.cat(
            [attended, hidden[:, -1], hidden.mean(dim=1), hidden.amax(dim=1)], dim=1
        )
        summary = self.summary_encoder(self._summary_features(dynamic))
        dynamic_rep = self.dynamic_trunk(torch.cat([temporal, summary], dim=1))

        static_standardized = (static - self.static_mean) / self.static_std
        static_rep = self.static_encoder(static_standardized)
        gamma, beta = self.film(static_rep).chunk(2, dim=1)
        gamma = 0.25 * torch.tanh(gamma)
        beta = 0.25 * torch.tanh(beta)
        fused = self.fusion_norm(
            dynamic_rep * (1.0 + gamma) + beta + 0.25 * self.static_residual(static_rep)
        )
        intensity = self.intensity_head(fused)
        escalation = self.escalation_head(fused)
        return torch.cat([intensity, escalation], dim=1)
