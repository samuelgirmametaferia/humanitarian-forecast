from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class GeoStateConfig:
    event_dim: int
    sequence_length: int
    resolutions: tuple[int, ...]
    cells_per_resolution: tuple[int, ...]
    d_model: int = 384
    heads: int = 8
    layers: int = 8
    ff_dim: int = 1152
    dropout: float = 0.10
    static_dim: int = 0


class GeoStateTransformer(nn.Module):
    """Multiscale dense-cell forecaster.

    The history encoder is shared across resolutions. Each H3 resolution receives
    a learned cell embedding plus an anchor-relative geographic bias. This keeps
    inference dense over the supplied humanitarian map rather than limiting the
    model to historical event coordinates.
    """

    def __init__(self, config: GeoStateConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model
        self.event_proj = nn.Sequential(
            nn.LayerNorm(config.event_dim),
            nn.Linear(config.event_dim, d),
            nn.GELU(),
        )
        self.position = nn.Parameter(torch.zeros(1, config.sequence_length, d))
        block = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=config.layers)
        self.history_norm = nn.LayerNorm(d)
        self.query = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.LayerNorm(d), nn.Linear(d, d))
        # Shared continuous geography is learned globally and transfers to H3
        # cells that never appeared as labelled conflict locations. A small
        # per-cell residual remains available after transfer, but is not the
        # primary representation.
        self.coordinate_encoder = nn.Sequential(
            nn.Linear(10, d), nn.GELU(), nn.LayerNorm(d), nn.Linear(d, d)
        )
        self.static_encoder = (
            nn.Sequential(nn.LayerNorm(config.static_dim), nn.Linear(config.static_dim, d), nn.GELU(), nn.Linear(d, d))
            if config.static_dim > 0 else None
        )
        self.static_bias = (
            nn.Sequential(nn.LayerNorm(config.static_dim), nn.Linear(config.static_dim, 64), nn.GELU(), nn.Linear(64, 1))
            if config.static_dim > 0 else None
        )
        self.cell_embeddings = nn.ModuleDict()
        self.geo_bias = nn.ModuleDict()
        self.temperature = nn.ParameterDict()
        for resolution, count in zip(config.resolutions, config.cells_per_resolution):
            key = str(resolution)
            emb = nn.Embedding(count, d)
            nn.init.normal_(emb.weight, std=0.02)
            self.cell_embeddings[key] = emb
            self.geo_bias[key] = nn.Sequential(
                nn.Linear(6, d // 2),
                nn.GELU(),
                nn.Linear(d // 2, 1),
            )
            self.temperature[key] = nn.Parameter(torch.tensor(0.0))

    def encode_history(self, x: torch.Tensor) -> torch.Tensor:
        valid = x[..., 0] > 0.5
        h = self.event_proj(x) + self.position[:, : x.shape[1]]
        h = self.encoder(h, src_key_padding_mask=~valid)
        weights = valid.float()
        pooled = (h * weights[..., None]).sum(dim=1) / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        return self.history_norm(pooled)

    @staticmethod
    def coordinate_features(latlon: torch.Tensor) -> torch.Tensor:
        """Continuous Fourier geography shared by candidate and H3 heads."""
        lat = latlon[..., 0] / 90.0
        lon = latlon[..., 1] / 180.0
        return torch.stack([
            lat,
            lon,
            torch.sin(math.pi * lat),
            torch.cos(math.pi * lat),
            torch.sin(2.0 * math.pi * lat),
            torch.cos(2.0 * math.pi * lat),
            torch.sin(math.pi * lon),
            torch.cos(math.pi * lon),
            torch.sin(2.0 * math.pi * lon),
            torch.cos(2.0 * math.pi * lon),
        ], dim=-1)

    @staticmethod
    def relative_geo(anchor_latlon: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
        """Anchor-relative smooth geometry for every dense map cell."""
        anchor_lat = anchor_latlon[:, 0:1]
        anchor_lon = anchor_latlon[:, 1:2]
        lat = centroids[:, 0][None, :]
        lon = centroids[:, 1][None, :]
        north = (lat - anchor_lat) * 111.32 / 1000.0
        east = (lon - anchor_lon) * 111.32 * torch.cos(torch.deg2rad(anchor_lat)) / 1000.0
        distance = torch.sqrt(east.square() + north.square() + 1e-8)
        bearing = torch.atan2(east, north)
        return torch.stack(
            [east, north, distance, torch.sin(bearing), torch.cos(bearing), torch.log1p(distance)],
            dim=-1,
        )

    def logits_for_resolution(
        self,
        history: torch.Tensor,
        anchor_latlon: torch.Tensor,
        centroids: torch.Tensor,
        resolution: int,
        static_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        key = str(resolution)
        q = self.query(history)
        embeddings = self.cell_embeddings[key].weight + self.coordinate_encoder(self.coordinate_features(centroids))
        static_bias = 0.0
        if self.static_encoder is not None:
            if static_features is None:
                raise ValueError("static_features are required when config.static_dim > 0")
            embeddings = embeddings + self.static_encoder(static_features)
            static_bias = self.static_bias(static_features).squeeze(-1)[None, :]
        content = q @ embeddings.t() / math.sqrt(self.config.d_model)
        geo = self.relative_geo(anchor_latlon, centroids)
        bias = self.geo_bias[key](geo).squeeze(-1)
        temperature = self.temperature[key].exp().clamp(0.25, 4.0)
        return (content + bias + static_bias) / temperature

    def forward(
        self,
        x: torch.Tensor,
        anchor_latlon: torch.Tensor,
        centroids: dict[int, torch.Tensor],
        static_features: dict[int, torch.Tensor] | None = None,
    ) -> dict[int, torch.Tensor]:
        history = self.encode_history(x)
        return {
            resolution: self.logits_for_resolution(
                history,
                anchor_latlon,
                centroids[resolution],
                resolution,
                None if static_features is None else static_features[resolution],
            )
            for resolution in self.config.resolutions
        }


def multiscale_loss(
    logits: dict[int, torch.Tensor],
    targets: dict[int, torch.Tensor],
    valid: dict[int, torch.Tensor],
    weights: dict[int, float] | None = None,
) -> torch.Tensor:
    weights = weights or {resolution: 1.0 for resolution in logits}
    total = None
    normalizer = 0.0
    for resolution, values in logits.items():
        mask = valid[resolution] & (targets[resolution] >= 0)
        if not bool(mask.any()):
            continue
        loss = nn.functional.cross_entropy(values[mask], targets[resolution][mask])
        weight = float(weights.get(resolution, 1.0))
        total = loss * weight if total is None else total + loss * weight
        normalizer += weight
    if total is None:
        raise ValueError("batch contains no valid multiscale targets")
    return total / max(normalizer, 1e-8)
