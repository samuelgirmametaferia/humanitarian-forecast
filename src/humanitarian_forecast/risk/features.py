from __future__ import annotations

import numpy as np


def engineered_temporal_features(x: np.ndarray) -> np.ndarray:
    """Create magnitude/recency/trend features for a diverse tree expert.

    Input shape is [example, time, feature]. The transform uses only each
    example's pre-cutoff history, so it is safe for chronological evaluation.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"expected [N,T,F] tensor, got shape={x.shape}")

    parts: list[np.ndarray] = [x.reshape(len(x), -1)]
    for window in (1, 3, 7, 14, 28):
        width = min(window, x.shape[1])
        recent = x[:, -width:]
        parts.extend(
            [
                recent.sum(axis=1),
                recent.mean(axis=1),
                recent.max(axis=1),
                recent.std(axis=1),
                (recent > 0).sum(axis=1).astype(np.float32),
            ]
        )

    nonzero = x > 0
    reversed_index = np.argmax(nonzero[:, ::-1, :], axis=1)
    recency = np.where(nonzero.any(axis=1), reversed_index, x.shape[1]).astype(np.float32)
    parts.append(recency)

    time = np.arange(x.shape[1], dtype=np.float32)
    centered = time - time.mean()
    denominator = float((centered * centered).sum()) or 1.0
    parts.append((x * centered[None, :, None]).sum(axis=1) / denominator)

    ages = np.arange(x.shape[1] - 1, -1, -1, dtype=np.float32)
    for half_life in (2.0, 7.0, 14.0):
        weights = np.power(0.5, ages / half_life)
        weights /= weights.sum()
        parts.append((x * weights[None, :, None]).sum(axis=1))

    if x.shape[1] >= 14:
        parts.append(x[:, -7:].sum(axis=1) - x[:, -14:-7].sum(axis=1))
        parts.append(x[:, -14:].sum(axis=1) - x[:, :14].sum(axis=1))
    else:
        midpoint = max(1, x.shape[1] // 2)
        parts.append(x[:, -midpoint:].sum(axis=1) - x[:, :midpoint].sum(axis=1))
        parts.append(np.zeros((len(x), x.shape[2]), dtype=np.float32))

    return np.concatenate(parts, axis=1).astype(np.float32, copy=False)
