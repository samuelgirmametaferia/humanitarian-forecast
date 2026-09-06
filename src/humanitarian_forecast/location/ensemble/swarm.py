#!/usr/bin/env python3
"""Validation-only convex swarm of frozen Ethiopia location experts.

Every expert is loaded from its saved per-row probability export on one of two
supports:

* dense H3 r4 cells (681 Ethiopia cells) — h3_lambdarank scratch/transfer,
  h3_lambdarank_memory;
* the 32 historical candidates of the v5 spatial-ring dataset —
  marked_hawkes, shape_analogue, reliefweb_spatial_specialist, and the live
  v10 candidate-ranker lineage.

All experts are projected onto the shared 681-cell r4 support (candidate
mass is assigned to the nearest cell), then blended with convex weights
selected on chronological validation only, using the project's broad-area
score. The final 15% development block is reported once, diagnostically.

The mixture is packaged as a small JSON state file so a later reinforcement
loop can update weights/temperatures cheaply and statelessly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from humanitarian_forecast.core.model_store import ModelInfo, write_info

EARTH_KM = 111.32
SCALE_KM = 1000.0

# Expert name -> probability-export path, support tag, and probability key
# prefix. ``support='r4'`` files carry (N, 681) matrices plus 'cells';
# ``support='candidates32'`` files carry (N, 32) matrices keyed
# validation_<suffix>/development_<suffix>.
DENSE_EXPERTS = {
    "h3_scratch": ("models/location/h3_lambdarank/ethiopia_r4_v1/h3_lambdarank_probabilities.npz", "validation", "development"),
    "h3_transfer": ("models/location/h3_lambdarank/ethiopia_r4_global_transfer_v1/h3_lambdarank_probabilities.npz", "validation", "development"),
    "h3_memory": ("models/location/h3_lambdarank_memory/ethiopia_r4_v1/h3_memory_lambdarank_probabilities.npz", "validation", "development"),
}

CANDIDATE_EXPERTS = {
    "marked_hawkes": ("models/location/marked_hawkes/ethiopia_v1/marked_hawkes_probabilities.npz", "validation_probability", "development_probability"),
    "shape_analogue": ("models/location/shape_analogue/ethiopia_v1/shape_analogue_probabilities.npz", "validation_probability", "development_probability"),
    "reliefweb_full": ("models/location/reliefweb_spatial_specialist/full_v1/ethiopia_probabilities.npz", "validation", "development"),
}

# Live candidate-ranker checkpoints projected onto the r4 support. Their row
# support is the Ethiopia rows of the 32-candidate v5 dataset (785 validation,
# 968 development), so they use the same projection as the candidate experts.
RANKER_VIEWS = {
    "v9_ethiopia_calibrated": "models/location/candidate_ranker_ethiopia/v10/candidate_ranker_calibrated.pt",
    "v10_prized_ethiopia32": "models/location/candidate_ranker_ethiopia/v10_prized_adapted/model.pt",
}


def _distance(cells: np.ndarray, truth: np.ndarray) -> np.ndarray:
    north = (cells[:, 0] - truth[0]) * EARTH_KM
    east = (cells[:, 1] - truth[1]) * EARTH_KM * np.cos(np.deg2rad(truth[0]))
    return np.sqrt(east * east + north * north).astype(np.float32)


def distances(cells: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.stack([_distance(cells, t) for t in truth])


def broad_metrics(probability: np.ndarray, d: np.ndarray) -> dict:
    top = np.argmax(probability, axis=1)
    e = d[np.arange(len(d)), top]
    top3_idx = np.argpartition(-probability, kth=min(2, probability.shape[1] - 1), axis=1)[:, :3]
    t3 = np.take_along_axis(d, top3_idx, axis=1).min(axis=1)
    out = {
        "samples": float(len(e)),
        "mean_error_km": float(e.mean()),
        "median_error_km": float(np.median(e)),
        "p90_error_km": float(np.quantile(e, 0.9)),
        "mean_entropy": float(np.mean(-(probability * np.log(np.maximum(probability, 1e-12))).sum(axis=1))),
    }
    for r in (25, 50, 100, 200):
        out[f"within_{r}km"] = float(np.mean(e <= r))
        out[f"probability_mass_within_{r}km"] = float(np.mean((probability * (d <= r)).sum(axis=1)))
    out["top3_within_100km"] = float(np.mean(t3 <= 100))
    out["broad_area_score"] = (
        0.55 * out["probability_mass_within_100km"]
        + 0.20 * out["probability_mass_within_200km"]
        + 0.15 * out["within_100km"]
        + 0.10 * out["top3_within_100km"]
    )
    return out


def calibrate_temperature(probability: np.ndarray, d: np.ndarray) -> tuple[float, np.ndarray]:
    """Sharpen/flatten a probability simplex with a temperature, selected on
    the broad-area score (the same metric the weights are selected on)."""
    logp = np.log(np.maximum(probability, 1e-12))
    best = (-np.inf, 1.0, None)
    for temperature in np.geomspace(0.1, 4.0, 31):
        sharpened = np.exp(logp / float(temperature))
        sharpened /= sharpened.sum(axis=1, keepdims=True)
        value = broad_metrics(sharpened, d)["broad_area_score"]
        if value > best[0] + 1e-12:
            best = (value, float(temperature), sharpened)
    assert best[2] is not None
    _, temperature, sharpened = best
    return temperature, sharpened


def simplex(total: int, parts: int, prefix: tuple[int, ...] = ()):  # retained for future full-grid audits
    if parts == 1:
        yield prefix + (total,)
        return
    for i in range(total + 1):
        yield from simplex(total - i, parts - 1, prefix + (i,))


def project_candidates_to_cells(candidate_probability: np.ndarray, absolute_candidates: np.ndarray, cells: np.ndarray, chunk: int = 64) -> np.ndarray:
    """Move candidate-support mass onto the nearest r4 cell per row.

    ``absolute_candidates`` are absolute (lat, lon) per candidate; ``cells``
    are r4 centroid (lat, lon). Mass is assigned to the nearest centroid
    under the project's planar kilometre approximation. Rows are processed
    in chunks to keep the (rows, candidates, cells) intermediates small.
    """
    n, k, _ = absolute_candidates.shape
    projected = np.zeros((n, len(cells)), dtype=np.float64)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        block = absolute_candidates[start:stop].astype(np.float32)
        delta = block[:, :, None, :] - cells[None, None, :, :].astype(np.float32)
        km2 = (
            (delta[..., 0] * EARTH_KM) ** 2
            + (delta[..., 1] * EARTH_KM * np.cos(np.deg2rad(block[:, :, None, 0]))) ** 2
        )
        nearest = np.argmin(km2, axis=2)
        rows = np.repeat(np.arange(start, stop), k)
        np.add.at(projected, (rows, nearest.ravel()), candidate_probability[start:stop].ravel())
    projected /= np.maximum(projected.sum(axis=1, keepdims=True), 1e-12)
    return projected


def load_dense_expert(path: Path, cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    validation = z["validation"].astype(np.float64)
    development = z["development"].astype(np.float64)
    assert np.allclose(z["cells"], cells), f"expert {path} uses a different r4 support"
    return validation, development


def load_candidate_expert(path: Path, val_key: str, dev_key: str) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    return z[val_key].astype(np.float64), z[dev_key].astype(np.float64)


def ranker_views(validation_rows: np.ndarray, development_rows: np.ndarray, data32: np.lib.npyio.NpzFile, meta: list[dict]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Run the frozen candidate-ranker checkpoints on the Ethiopia rows and
    return softmax probabilities over the 32 candidates.

    ``meta`` is the already-parsed metadata list; npz member access is
    expensive (each access re-decompresses the member), so all large arrays
    are read exactly once here.
    """
    import torch

    from humanitarian_forecast.location.models.candidate_rank import ConflictCandidateRanker

    x_all = data32["x"]
    features_all = data32["candidate_features"]
    valid_all = data32["candidate_valid"]
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, checkpoint in RANKER_VIEWS.items():
        path = Path(checkpoint)
        if not path.exists():
            continue
        state = torch.load(path, map_location="cpu", weights_only=False)
        model = ConflictCandidateRanker(**state["model_config"])
        model.load_state_dict(state["model_state"])
        model.eval()
        countries = {str(k): int(v) for k, v in dict(state.get("country_to_id", {})).items()}
        conflicts = {str(k): int(v) for k, v in dict(state.get("conflict_to_id", {})).items()}
        probabilities: dict[str, np.ndarray] = {}
        for split, rows in (("validation", validation_rows), ("development", development_rows)):
            meta_rows = [meta[i] for i in rows]
            x = torch.from_numpy(x_all[rows]).float()
            f = torch.from_numpy(features_all[rows]).float()
            v = torch.from_numpy(valid_all[rows])
            country = torch.tensor([countries.get(m["country"], 0) for m in meta_rows])
            conflict = torch.tensor([conflicts.get(m["conflict_id"], 0) for m in meta_rows])
            temperature = float(state.get("center_calibration", {}).get("temperature", 1.0) or 1.0)
            with torch.no_grad():
                logits = model(x, f, v, country, conflict)
            probabilities[split] = (logits / temperature).softmax(-1).numpy().astype(np.float64)
        out[name] = (probabilities["validation"], probabilities["development"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data32", type=Path, default=Path("data/location/conflict_candidates_32_spatial_v5.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/location/swarm/v1"))
    parser.add_argument("--step", type=float, default=0.05)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)

    data32 = np.load(args.data32, allow_pickle=True)
    meta = [json.loads(str(v)) for v in data32["meta"]]
    n = len(meta)
    ethiopia = np.asarray([i for i, m in enumerate(meta) if m["country"] == "Ethiopia"])
    validation_rows = ethiopia[(ethiopia >= int(0.70 * n)) & (ethiopia < int(0.85 * n))]
    development_rows = ethiopia[ethiopia >= int(0.85 * n)]

    dense = np.load(DENSE_EXPERTS["h3_scratch"][0], allow_pickle=True)
    cells = dense["cells"].astype(np.float64)
    truth_val = np.asarray([[m["target_lat"], m["target_lon"]] for m in (meta[i] for i in validation_rows)], dtype=np.float32)
    truth_dev = np.asarray([[m["target_lat"], m["target_lon"]] for m in (meta[i] for i in development_rows)], dtype=np.float32)
    d_val = distances(cells, truth_val)
    d_dev = distances(cells, truth_dev)

    # Candidate support -> shared r4 support projection.
    # candidate_coordinates are (east_km, north_km)/1000 relative to the row
    # anchor; convert to absolute (lat, lon) before nearest-cell assignment.
    def to_absolute(coordinates: np.ndarray, rows: np.ndarray) -> np.ndarray:
        out = np.empty_like(coordinates)
        for j, i in enumerate(rows):
            m = meta[i]
            lat, lon = float(m["anchor_lat"]), float(m["anchor_lon"])
            out[j, :, 0] = lat + coordinates[j, :, 1] * SCALE_KM / EARTH_KM
            out[j, :, 1] = lon + coordinates[j, :, 0] * SCALE_KM / (EARTH_KM * max(0.1, np.cos(np.deg2rad(lat))))
        return out
    absolute_val = to_absolute(data32["candidate_coordinates"][validation_rows].astype(np.float64), validation_rows)
    absolute_dev = to_absolute(data32["candidate_coordinates"][development_rows].astype(np.float64), development_rows)

    experts: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, (path, val_key, dev_key) in DENSE_EXPERTS.items():
        experts[name] = load_dense_expert(Path(path), cells)
    for name, (path, val_key, dev_key) in CANDIDATE_EXPERTS.items():
        v, d = load_candidate_expert(Path(path), val_key, dev_key)
        experts[name] = (
            project_candidates_to_cells(v, absolute_val, cells),
            project_candidates_to_cells(d, absolute_dev, cells),
        )
    for name, (v, d) in ranker_views(validation_rows, development_rows, data32, meta).items():
        valid_val = data32["candidate_valid"][validation_rows]
        valid_dev = data32["candidate_valid"][development_rows]
        v = np.where(valid_val, v, 0.0)
        v /= v.sum(axis=1, keepdims=True)
        d = np.where(valid_dev, d, 0.0)
        d /= d.sum(axis=1, keepdims=True)
        experts[name] = (
            project_candidates_to_cells(v, absolute_val, cells),
            project_candidates_to_cells(d, absolute_dev, cells),
        )

    # Per-expert temperature calibration on the validation broad-area score
    # (the same metric the mixture weights are selected on), then convex
    # weight search on the same score.
    names = sorted(experts)
    calibrated: dict[str, tuple[float, np.ndarray, np.ndarray]] = {}
    for name in names:
        v, d = experts[name]
        temperature, sharpened = calibrate_temperature(v, d_val)
        calibrated[name] = (temperature, sharpened, d)
        print(json.dumps({"expert": name, "temperature": temperature, "validation_broad_area_score": broad_metrics(sharpened, d_val)["broad_area_score"]}))

    names = sorted(calibrated)
    stack_val = np.stack([calibrated[name][1] for name in names])  # (E, rows, cells)
    def score(weights: np.ndarray) -> float:
        blend = np.tensordot(weights, stack_val, axes=(0, 0))
        blend /= np.maximum(blend.sum(axis=1, keepdims=True), 1e-12)
        return broad_metrics(blend, d_val)["broad_area_score"]

    # Greedy forward selection with weight grid, then hill-climb refinement.
    # Validation-only; each candidate blend is scored on the broad-area metric.
    selected = [max(names, key=lambda n: broad_metrics(calibrated[n][1], d_val)["broad_area_score"])]
    weights = {name: 0.0 for name in names}
    weights[selected[0]] = 1.0
    current = score(np.asarray([weights[n] for n in names]))
    improved = True
    while improved and len(selected) < len(names):
        improved = False
        for name in [n for n in names if n not in selected]:
            for w in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
                trial = dict(weights)
                for s in selected:
                    trial[s] = weights[s] * (1.0 - w)
                trial[name] = w
                value = score(np.asarray([trial[n] for n in names]))
                if value > current + 1e-9:
                    weights, current, improved = trial, value, True
                    if name not in selected:
                        selected.append(name)
    # Hill-climb: nudge individual weights by +/- step, renormalize.
    for _ in range(50):
        changed = False
        for name in names:
            for delta in (+args.step, -args.step):
                trial = dict(weights)
                trial[name] = max(0.0, trial[name] + delta)
                total = sum(trial.values())
                if total <= 0:
                    continue
                trial = {k: v / total for k, v in trial.items()}
                value = score(np.asarray([trial[n] for n in names]))
                if value > current + 1e-9:
                    weights, current, changed = trial, value, True
        if not changed:
            break
    weight_vector = np.asarray([weights[n] for n in names])
    blend_val = np.tensordot(weight_vector, stack_val, axes=(0, 0))
    blend_val /= np.maximum(blend_val.sum(axis=1, keepdims=True), 1e-12)
    validation = broad_metrics(blend_val, d_val)

    # Development diagnostics: apply the same weights to development expert
    # probabilities (temperatures already fixed on validation).
    def sharpen(probability: np.ndarray, temperature: float) -> np.ndarray:
        logp = np.log(np.maximum(probability, 1e-12))
        out = np.exp(logp / temperature)
        return out / out.sum(axis=1, keepdims=True)

    blend_dev = sum(
        w * sharpen(experts[name][1], calibrated[name][0])
        for w, name in zip(weight_vector, names)
    )
    blend_dev /= blend_dev.sum(axis=1, keepdims=True)
    development = broad_metrics(blend_dev, d_dev)

    report = {
        "schema": "swarm-mixture-v1",
        "experts": names,
        "temperatures": {name: calibrated[name][0] for name in names},
        "weights": {name: float(weights[name]) for name in names},
        "validation": validation,
        "development": development,
        "step": args.step,
        "protocol": "per-expert temperature and convex weights selected on chronological validation only; development is diagnostic",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "swarm.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(
        args.output_dir / "probabilities.npz",
        validation=blend_val.astype(np.float32),
        development=blend_dev.astype(np.float32),
        cells=cells.astype(np.float32),
        validation_rows=validation_rows.astype(np.int64),
        development_rows=development_rows.astype(np.int64),
    )
    write_info(
        args.output_dir,
        ModelInfo(
            subsystem="location/swarm",
            version="v1",
            status="research-challenger",
            description="Validation-selected convex swarm of frozen Ethiopia location experts on the shared H3 r4 support.",
            metrics={"validation": validation, "development": development},
            lineage={"experts": {name: str(path) for name, (path, _, _) in {**DENSE_EXPERTS, **CANDIDATE_EXPERTS}.items()} | {name: str(path) for name, path in RANKER_VIEWS.items()}},
            calibration={"weights": report["weights"], "temperatures": report["temperatures"], "selection": "validation broad-area score for both per-expert temperature and convex weights"},
            notes=[
                "All experts share the 681-cell Ethiopia H3 r4 support; candidate-support experts are projected by nearest-cell mass assignment.",
                "Development block is diagnostic, not a prospective test.",
                "Ethiopia rows use the swarm; other countries route through the v10-prized global64 view.",
            ],
        ),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
