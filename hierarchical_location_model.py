from __future__ import annotations

import math
import torch
from torch import nn


class GridCellEncoder(nn.Module):
    """Encodes geographic position into grid cell embeddings."""

    def __init__(self, feature_dim: int = 15, grid_resolution: float = 1.0, d_model: int = 128):
        super().__init__()
        self.grid_resolution = grid_resolution
        self.d_model = d_model
        self.register_buffer("grid_mean", torch.tensor(0.0))
        self.register_buffer("grid_std", torch.tensor(1.0))
        self.global_embedding = nn.Embedding(1, d_model)
        self.lat_proj = nn.Linear(1, d_model // 2)
        self.lon_proj = nn.Linear(1, d_model // 2)
        self._built = False

    def _ensure_built(self, min_lat: float, max_lat: float, min_lon: float, max_lon: float):
        if self._built:
            return
        with torch.no_grad():
            n_lat = max(1, int((max_lat - min_lat) / self.grid_resolution))
            n_lon = max(1, int((max_lon - min_lon) / self.grid_resolution))
            lat_grid = torch.linspace(min_lat, max_lat, n_lat)
            lon_grid = torch.linspace(min_lon, max_lon, n_lon)
            lat_emb = self.lat_proj(lat_grid[:, None]).unsqueeze(1)
            lon_emb = self.lon_proj(lon_grid[:, None]).unsqueeze(1)
            self.grid_lat_emb = lat_emb
            self.grid_lon_emb = lon_emb
            self.grid_position_emb = torch.cat([lat_emb, lon_emb], dim=1).mean(dim=1)
            self._built = True

    def forward(self, lat: torch.Tensor, lon: torch.Tensor) -> torch.Tensor:
        B = lat.shape[0]
        if not self._built:
            self._ensure_built(lat.min().item(), lat.max().item(), lon.min().item(), lon.max().item())
        lat_norm = (lat - lat.min()) / (lat.max() - lat.min() + 1e-8)
        lon_norm = (lon - lon.min()) / (lon.max() - lon.min() + 1e-8)
        lat_emb = self.lat_proj(lat_norm[:, :, None]).reshape(B, -1, self.d_model // 2)
        lon_emb = self.lon_proj(lon_norm[:, :, None]).reshape(B, -1, self.d_model // 2)
        pos = torch.cat([lat_emb, lon_emb], dim=-1)
        pos = pos + self.grid_position_emb[: pos.shape[1], :].unsqueeze(0)
        return pos


class HistoryEncoder(nn.Module):
    """Transformer encoder for historical event sequences."""

    def __init__(self, feature_dim: int, sequence_length: int, d_model: int = 128, heads: int = 8, layers: int = 4, ff_dim: int = 384, dropout: float = 0.15):
        super().__init__()
        self.input_norm = nn.LayerNorm(feature_dim)
        self.input_projection = nn.Linear(feature_dim, d_model)
        self.position = nn.Parameter(torch.empty(1, sequence_length, d_model))
        nn.init.normal_(self.position, std=0.02)
        block = nn.TransformerEncoderLayer(d_model=d_model, nhead=heads, dim_feedforward=ff_dim,
                                           dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, num_layers=layers)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(self.input_norm(features)) * math.sqrt(self.position.shape[1])
        x = x + self.position[:, : x.shape[1]]
        return self.encoder(x, src_key_padding_mask=mask)


class HorizonEmbedding(nn.Module):
    """Learned horizon embedding plus numerical log-horizon feature."""

    def __init__(self, d_model: int = 128, max_horizon: int = 90):
        super().__init__()
        self.d_model = d_model
        self.horizon_embed = nn.Embedding(max_horizon + 1, d_model // 4)
        self.log_horizon = nn.Linear(1, d_model // 4)
        self._initialized = False

    def forward(self, horizon_days: torch.Tensor | None = None) -> torch.Tensor:
        if horizon_days is None:
            zeros = torch.zeros(1, dtype=torch.long, device=self.horizon_embed.weight.device)
            h = self.horizon_embed(zeros)
            return torch.cat([h, torch.zeros(1, self.d_model - h.shape[1], device=h.device)], dim=-1)
        if not self._initialized:
            self._initialized = True
        h_embed = self.horizon_embed(horizon_days.clamp(0, 90))
        log_h = self.log_horizon(torch.log1p(horizon_days[:, None]).squeeze(-1))
        return torch.cat([h_embed, log_h], dim=-1).unsqueeze(1)


class CoarseClassifier(nn.Module):
    """Predicts probability distribution over candidate destination cells."""

    def __init__(self, n_candidates: int, d_model: int = 128):
        super().__init__()
        self.n_candidates = n_candidates
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, n_candidates),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)


class LocalOffsetHead(nn.Module):
    """Predicts latitude/longitude offset inside a selected candidate cell."""

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.offset_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.offset_head(features)


class UncertaintyHead(nn.Module):
    """Predicts aleatoric radius for each forecast component."""

    def __init__(self, d_model: int = 128, n_components: int = 5):
        super().__init__()
        self.radius_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, n_components),
        )
        self.n_components = n_components

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raw = self.radius_head(features)
        return nn.functional.softplus(raw) + 0.01


class HierarchicalLocationTransformer(nn.Module):
    """Hierarchical model: cell classifier -> local offset -> uncertainty.

    Architecture per plan.md §5:
    1. History encoder over events, ReliefWeb, horizon, geographic features
    2. Candidate generator: current cell, neighboring cells, historically active cells
    3. Coarse classifier: probability distribution over candidate cells
    4. Local offset head: lat/lon offset inside selected cell
    5. Uncertainty head: aleatoric radius for each component
    6. Probability calibrator: validation-fitted temperature
    """

    def __init__(
        self,
        feature_dim: int = 15,
        sequence_length: int = 16,
        n_candidates: int = 32,
        d_model: int = 160,
        heads: int = 8,
        layers: int = 5,
        ff_dim: int = 512,
        dropout: float = 0.15,
        components: int = 5,
        grid_resolution: float = 1.0,
    ):
        super().__init__()
        self.n_candidates = n_candidates
        self.components = components
        self.config = dict(
            feature_dim=feature_dim,
            sequence_length=sequence_length,
            n_candidates=n_candidates,
            d_model=d_model,
            heads=heads,
            layers=layers,
            ff_dim=ff_dim,
            dropout=dropout,
            components=components,
            grid_resolution=grid_resolution,
        )
        self.encoder = HistoryEncoder(feature_dim, sequence_length, d_model, heads, layers, ff_dim, dropout)
        self.grid_encoder = GridCellEncoder(feature_dim, grid_resolution, d_model)
        self.horizon = HorizonEmbedding(d_model)
        self.coarse_classifier = CoarseClassifier(n_candidates, d_model)
        self.local_offset = LocalOffsetHead(d_model)
        self.uncertainty = UncertaintyHead(d_model, components)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, components * 4))

    def forward(
        self,
        features: torch.Tensor,
        lat: torch.Tensor | None = None,
        lon: torch.Tensor | None = None,
        horizon_days: torch.Tensor | None = None,
        candidate_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Encode history
        valid = features[:, :, 0] > 0.5
        enc = self.encoder(features, ~valid)

        # Geographic context
        if lat is not None and lon is not None:
            grid_feat = self.grid_encoder(lat, lon)
            enc = enc + grid_feat

        # Horizon embedding
        h_feat = self.horizon(horizon_days)
        enc = enc + h_feat

        # Coarse classifier over candidates
        logits = self.coarse_classifier(enc)

        # Apply candidate masking if provided
        if candidate_mask is not None:
            logits = logits.masked_fill(~candidate_mask, -1e9)

        # Get probabilities over candidates
        probs = logits.softmax(-1)

        # Main head produces components * 4 outputs: logits, center offsets, probs, radii
        # enc shape: (B, seq_len, d_model) - take the last valid event output
        last_enc = enc[:, -1, :]  # (B, d_model)
        head_out = self.head(last_enc)  # (B, components * 4)
        head_logits = head_out[:, 0]  # first component's logit (simplified)
        head_centers = head_out[:, 1:3]  # center offsets (simplified) - shape (B, 2)
        head_sigmas = nn.functional.softplus(head_out[:, 3]) + 0.01  # radius from last dim - shape (B,)

        # Simplified uncertainty - use per-component sigmas
        uncert = self.uncertainty(enc)  # (B, components)

        return logits, probs, head_logits, head_centers, head_sigmas, uncert


def hierarchical_nll(logits, centers, sigmas, target, rank_weight=0.1, temp=100.0):
    """NLL with ranking loss encouraging probability on nearest component.
    
    Args:
        logits: (B, n_candidates) or (B, seq_len, n_candidates)
        centers: (B, 2) predicted center coordinates
        sigmas: (B,) or (B, n_components) predicted radii
        target: (B, 2) ground truth coordinates
    """
    # Ensure logits are (B, n_candidates) - take last timestep if needed
    if logits.dim() == 3:
        logits = logits[:, -1, :]  # (B, n_candidates)
    
    # Ensure centers are (B, 2)
    if centers.dim() == 2 and centers.shape[1] != 2:
        centers = centers[:, :2]  # take first 2 dims if needed
    
    # Ensure sigmas are (B,) or (B, n_components)
    if sigmas.dim() == 1:
        # single sigma per batch element
        pass
    elif sigmas.dim() == 2:
        # use mean across components
        sigmas = sigmas.mean(dim=1)
    
    # Ensure target is (B, 2)
    if target.dim() == 1:
        target = target.unsqueeze(0)
    
    # Compute NLL
    error2 = ((centers - target) ** 2).sum(-1)  # (B,)
    log_component = -math.log(2 * math.pi) - 2 * sigmas.log() - error2 / (2 * sigmas.square() + 1e-8)
    nll = -torch.logsumexp(torch.log_softmax(logits, -1) + log_component.unsqueeze(-1), -1).mean()

    # Ranking loss: encourage higher probability on nearer components
    error2 = ((centers - target[:, None, :]) ** 2).sum(-1)
    soft_nearest = torch.softmax(-error2 / (2 * temp ** 2), dim=-1)
    ranking = -(soft_nearest * torch.log_softmax(logits, dim=-1)).sum(-1).mean()

    return nll + rank_weight * ranking


def gaussian_location_loss(center, sigma, target):
    """Uncertainty-aware location loss."""
    squared_error = ((center - target) ** 2).sum(dim=1)
    return (0.5 * squared_error / sigma.square() + 2.0 * sigma.log()).mean()