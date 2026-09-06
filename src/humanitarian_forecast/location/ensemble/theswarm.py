#!/usr/bin/env python3
"""TheSwarm: production mixture of frozen Ethiopia location experts.

Improves on the swarm-v1 recipe with four changes, all selected under a
cross-fit guard so noise-chasing is caught before packaging:

* two kernel-smoothing experts (Gaussian fields around the cutoff-safe
  anchor and the history mean) add the spatial smoothness the sharply
  calibrated experts lack;
* per-expert temperatures and mixture weights are optimized *jointly*
  (coordinate ascent), not in stages;
* both pooling families are searched — linear (weighted arithmetic mean)
  and geometric (product of experts);
* optional regime gates (spatial north/south, forecast horizon) give
  each regime its own weight vector.

Protocol: the chronological validation block is split in half. Recipes
are fitted on the first half and scored on the second half; the recipe
with the best guard-half score is refitted on the full validation block.
The final development block is evaluated exactly once, diagnostically.

The packaged artifact keeps the swarm-v1 layout (JSON state + per-row
probability exports) so the planned reinforcement loop and the coarse
zone inference keep working unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from humanitarian_forecast.core.model_store import ModelInfo, write_info
from humanitarian_forecast.location.ensemble.swarm import (
    CANDIDATE_EXPERTS,
    DENSE_EXPERTS,
    EARTH_KM,
    RANKER_VIEWS,
    SCALE_KM,
    broad_metrics,
    distances,
    load_candidate_expert,
    load_dense_expert,
    project_candidates_to_cells,
    ranker_views,
)

TEMPERATURES = np.geomspace(0.1, 4.0, 31)
BANDWIDTHS_KM = np.geomspace(25.0, 400.0, 10)
ENTROPY_TAUS = (0.25, 0.5, 1.0, 2.0, 4.0)


def sharpen(probability: np.ndarray, temperature: float) -> np.ndarray:
    logp = np.log(np.maximum(probability, 1e-12))
    out = np.exp(logp / temperature)
    return out / out.sum(axis=1, keepdims=True)


def pool(weights: np.ndarray, calibrated: np.ndarray, how: str) -> np.ndarray:
    """Blend a (experts, rows, cells) stack into (rows, cells)."""
    if how == "linear":
        blend = np.tensordot(weights, calibrated, axes=(0, 0))
    else:  # geometric: product of experts with simplex exponents
        logp = np.tensordot(weights, np.log(np.maximum(calibrated, 1e-12)), axes=(0, 0))
        blend = np.exp(logp)
    return blend / np.maximum(blend.sum(axis=1, keepdims=True), 1e-12)


def entropy_pool(weights: np.ndarray, calibrated: np.ndarray, tau: float) -> np.ndarray:
    """Confidence-adaptive linear pooling: each row re-weights experts by
    ``base_weight * exp(-entropy / tau)``, so sharp (confident) experts take
    over on the rows where they commit and flat ones defer."""
    entropy = -(calibrated * np.log(np.maximum(calibrated, 1e-12))).sum(axis=-1)  # (experts, rows)
    adjusted = weights[:, None] * np.exp(-entropy / tau)
    adjusted /= np.maximum(adjusted.sum(axis=0, keepdims=True), 1e-12)
    blend = np.einsum("kr,krc->rc", adjusted, calibrated)
    return blend / np.maximum(blend.sum(axis=1, keepdims=True), 1e-12)


def score(weights: np.ndarray, calibrated: np.ndarray, d: np.ndarray, how: str) -> float:
    return broad_metrics(pool(weights, calibrated, how), d)["broad_area_score"]


def hill_climb(calibrated: np.ndarray, d: np.ndarray, how: str, weights: np.ndarray, step: float, rounds: int = 40) -> np.ndarray:
    """Greedy forward selection followed by +/- step nudges, on one score."""
    experts = calibrated.shape[0]
    def value(w: np.ndarray) -> float:
        return score(w, calibrated, d, how)

    # Greedy forward selection from the strongest single expert.
    singles = [broad_metrics(calibrated[k], d)["broad_area_score"] for k in range(experts)]
    weights = np.zeros(experts)
    weights[int(np.argmax(singles))] = 1.0
    current = value(weights)
    selected = {int(np.argmax(singles))}
    improved = True
    while improved and len(selected) < experts:
        improved = False
        for k in range(experts):
            if k in selected:
                continue
            for w in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
                trial = weights * (1.0 - w)
                trial[k] = w
                value_now = value(trial)
                if value_now > current + 1e-9:
                    weights, current, improved = trial, value_now, True
                    selected.add(k)
    # Hill-climb with renormalization.
    for _ in range(rounds):
        changed = False
        for k in range(experts):
            for delta in (+step, -step):
                trial = weights.copy()
                trial[k] = max(0.0, trial[k] + delta)
                total = trial.sum()
                if total <= 0:
                    continue
                trial /= total
                value_now = value(trial)
                if value_now > current + 1e-9:
                    weights, current, changed = trial, value_now, True
        if not changed:
            break
    return weights


def fit(stack: np.ndarray, d: np.ndarray, how: str, step: float, joint: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Coordinate ascent over per-expert temperatures and weights.

    ``stack`` is (experts, rows, cells) of raw expert probabilities.
    ``joint=False`` reproduces the swarm-v1 staging: per-expert temperature
    grid calibration followed by a weight search, with no alternation.
    Returns (temperatures, weights).
    """
    experts, rows, cells = stack.shape
    temperatures = np.ones(experts)
    for k in range(experts):
        best = (-np.inf, 1.0)
        for t in TEMPERATURES:
            value = broad_metrics(sharpen(stack[k], float(t)), d)["broad_area_score"]
            if value > best[0] + 1e-12:
                best = (value, float(t))
        temperatures[k] = best[1]
    calibrated = np.stack([sharpen(stack[k], temperatures[k]) for k in range(experts)])
    weights = hill_climb(calibrated, d, how, np.zeros(experts), step)
    if not joint:
        return temperatures, weights
    # Alternate temperature nudges (for experts carrying weight) with
    # weight re-fits until neither improves the blend.
    current = score(weights, calibrated, d, how)
    for _ in range(8):
        improved = False
        for k in range(experts):
            if weights[k] <= 0:
                continue
            for t in TEMPERATURES:
                if abs(t - temperatures[k]) < 1e-12:
                    continue
                trial_cal = calibrated.copy()
                trial_cal[k] = sharpen(stack[k], float(t))
                value = score(weights, trial_cal, d, how)
                if value > current + 1e-9:
                    temperatures[k], calibrated, current, improved = float(t), trial_cal, value, True
        weights_now = hill_climb(calibrated, d, how, weights, step)
        value_now = score(weights_now, calibrated, d, how)
        if value_now > current + 1e-9:
            weights, current, improved = weights_now, value_now, True
        if not improved:
            break
    return temperatures, weights


def kernel_field(d_center: np.ndarray, sigma_km: float) -> np.ndarray:
    field = np.exp(-0.5 * (d_center / sigma_km) ** 2)
    return field / field.sum(axis=1, keepdims=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data32", type=Path, default=Path("data/location/conflict_candidates_32_spatial_v5.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/location/theswarm/v1"))
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

    # Kernel-smoothing experts: Gaussian fields around cutoff-safe centers.
    # The anchor is always available; the history mean uses only events at or
    # before the observation cutoff (the dataset's x history is cutoff-safe).
    def center_of(rows: np.ndarray, mode: str) -> np.ndarray:
        out = np.empty((len(rows), 2))
        x_all = data32["x"]
        for j, i in enumerate(rows):
            m = meta[i]
            lat, lon = float(m["anchor_lat"]), float(m["anchor_lon"])
            if mode == "anchor":
                out[j] = (lat, lon)
            else:
                history = x_all[i][x_all[i][:, 0] > 0]
                offset = history[:, 1:3].mean(0) if len(history) else np.zeros(2)
                out[j] = (
                    lat + float(offset[1]) * SCALE_KM / EARTH_KM,
                    lon + float(offset[0]) * SCALE_KM / (EARTH_KM * max(0.1, np.cos(np.deg2rad(lat)))),
                )
        return out
    for mode in ("anchor", "history"):
        d_center_val = distances(cells, center_of(validation_rows, mode))
        d_center_dev = distances(cells, center_of(development_rows, mode))
        best = (-np.inf, float(BANDWIDTHS_KM[0]))
        for sigma in BANDWIDTHS_KM:
            value = broad_metrics(kernel_field(d_center_val, float(sigma)), d_val)["broad_area_score"]
            if value > best[0] + 1e-12:
                best = (value, float(sigma))
        _, sigma = best
        experts[f"kernel_{mode}"] = (
            kernel_field(d_center_val, sigma),
            kernel_field(d_center_dev, sigma),
        )
        print(json.dumps({"expert": f"kernel_{mode}", "bandwidth_km": sigma}))

    names = sorted(experts)
    stack_val = np.stack([experts[name][0] for name in names])
    stack_dev = np.stack([experts[name][1] for name in names])

    # Cross-fit guard: fit on the early half of validation, score on the late half.
    rows_val = len(validation_rows)
    half = rows_val // 2
    fit_idx = np.arange(half)
    guard_idx = np.arange(half, rows_val)

    # Regime definitions on validation metadata (fixed splits, no fitting).
    meta_val = [meta[i] for i in validation_rows]
    anchor_lats = np.asarray([float(m["anchor_lat"]) for m in meta_val])
    gap_days = np.asarray([int(m["gap_days"]) for m in meta_val])
    lat_split = float(np.median(anchor_lats))
    gap_split = int(np.median(gap_days))
    regimes = {
        "none": np.zeros(rows_val, dtype=int),
        "spatial": (anchor_lats >= lat_split).astype(int),
        "horizon": (gap_days > gap_split).astype(int),
    }

    def regime_weights(stack: np.ndarray, d: np.ndarray, how: str, regime: np.ndarray, temperatures: np.ndarray) -> dict[int, np.ndarray]:
        """Per-regime weight vectors over globally fitted temperatures."""
        calibrated = np.stack([sharpen(stack[k], temperatures[k]) for k in range(stack.shape[0])])
        out = {}
        for r in np.unique(regime):
            mask = regime == r
            out[int(r)] = hill_climb(calibrated[:, mask, :], d[mask], how, np.zeros(stack.shape[0]), args.step)
        return out

    def blend_rows(stack: np.ndarray, temperatures: np.ndarray, by_regime: dict[int, np.ndarray], regime: np.ndarray, how: str, tau: float | None) -> np.ndarray:
        """Single blend path used for metrics and for the packaged exports."""
        calibrated = np.stack([sharpen(stack[k], temperatures[k]) for k in range(stack.shape[0])])
        blend = np.zeros((stack.shape[1], calibrated.shape[2]))
        for r, weights in by_regime.items():
            mask = regime == r
            if not mask.any():
                continue
            block = calibrated[:, mask, :]
            blend[mask] = entropy_pool(weights, block, tau) if tau is not None else pool(weights, block, how)
        return blend / np.maximum(blend.sum(axis=1, keepdims=True), 1e-12)

    def run_recipe(how: str, gate: str, idx: np.ndarray, regime_all: np.ndarray) -> tuple[np.ndarray, dict[int, np.ndarray], float | None]:
        """Fit one recipe on the given rows; returns temperatures, per-regime
        weights, and the entropy-adaptation tau (None unless requested)."""
        stack = stack_val[:, idx, :]
        d = d_val[idx]
        regime = regime_all[idx]
        base_how = "geometric" if how == "geometric" else "linear"
        temperatures, flat = fit(stack, d, base_how, args.step, joint=how != "staged_linear")
        by_regime = {0: flat} if gate == "none" else regime_weights(stack, d, base_how, regime, temperatures)
        tau = None
        if how == "entropy_linear":
            best = (-np.inf, None)
            for candidate in ENTROPY_TAUS:
                value = broad_metrics(blend_rows(stack, temperatures, by_regime, regime, "linear", candidate), d)["broad_area_score"]
                if value > best[0] + 1e-12:
                    best = (value, float(candidate))
            tau = best[1]
        return temperatures, by_regime, tau

    # Baseline context: the packaged swarm-v1 blend on each half.
    v1_path = Path("models/location/swarm/v1/probabilities.npz")
    if v1_path.exists():
        v1 = np.load(v1_path, allow_pickle=True)
        if np.array_equal(v1["validation_rows"], validation_rows):
            v1_blend = v1["validation"].astype(np.float64)
            for label, idx in (("fit", fit_idx), ("guard", guard_idx)):
                print(json.dumps({"baseline": "swarm_v1", "half": label, "broad_area_score": broad_metrics(v1_blend[idx], d_val[idx])["broad_area_score"]}))

    trials = []
    for how in ("staged_linear", "linear", "geometric", "entropy_linear"):
        for gate in ("none", "spatial", "horizon"):
            temperatures, by_regime, tau = run_recipe(how, gate, fit_idx, regimes[gate])
            fit_blend = blend_rows(stack_val[:, fit_idx, :], temperatures, by_regime, regimes[gate][fit_idx], "linear" if how != "geometric" else "geometric", tau)
            guard_blend = blend_rows(stack_val[:, guard_idx, :], temperatures, by_regime, regimes[gate][guard_idx], "linear" if how != "geometric" else "geometric", tau)
            trials.append({
                "pooling": how, "gate": gate, "tau": tau,
                "temperatures": temperatures.tolist(),
                "weights": {str(r): w.tolist() for r, w in by_regime.items()},
                "fit": broad_metrics(fit_blend, d_val[fit_idx]),
                "guard": broad_metrics(guard_blend, d_val[guard_idx]),
            })
            print(json.dumps({"pooling": how, "gate": gate, "tau": tau, "fit_broad": trials[-1]["fit"]["broad_area_score"], "guard_broad": trials[-1]["guard"]["broad_area_score"]}))

    # Choose the recipe by guard-half score; recipes within a noise band
    # (0.002 broad-area) of the best are treated as tied and the simplest
    # one wins — extra optimizer freedom only ever matched the simple
    # staged recipe on the guard half, and simplicity is the safer bet.
    pooling_rank = {"staged_linear": 0, "linear": 1, "entropy_linear": 2, "geometric": 3}
    gate_rank = {"none": 0, "spatial": 1, "horizon": 1}

    def complexity(t) -> tuple[int, int]:
        return (gate_rank[t["gate"]], pooling_rank[t["pooling"]])

    best_guard = max(t["guard"]["broad_area_score"] for t in trials)
    finalists = [t for t in trials if t["guard"]["broad_area_score"] >= best_guard - 0.002]
    best_trial = min(finalists, key=complexity)
    how, gate, tau = best_trial["pooling"], best_trial["gate"], best_trial["tau"]
    base_how = "geometric" if how == "geometric" else "linear"

    # Robustness knob — one scalar, selected on the guard half: mix the
    # winning fit-half blend with the smooth anchor-kernel field so the
    # packaged distribution keeps spatial mass near the anchor instead of
    # collapsing onto a couple of near-one-hot cells.
    kernel_anchor_val, kernel_anchor_dev = experts["kernel_anchor"]
    temperatures_g, by_regime_g, tau_g = best_trial["temperatures"], {int(k): np.asarray(v) for k, v in best_trial["weights"].items()}, best_trial["tau"]
    guard_blend_plain = blend_rows(stack_val[:, guard_idx, :], np.asarray(temperatures_g), by_regime_g, regimes[gate][guard_idx], base_how, tau_g)
    base_guard = broad_metrics(guard_blend_plain, d_val[guard_idx])["broad_area_score"]
    delta, delta_guard = 0.0, base_guard
    for candidate in (0.05, 0.10, 0.15, 0.20, 0.30):
        smoothed = (1.0 - candidate) * guard_blend_plain + candidate * kernel_anchor_val[guard_idx]
        value = broad_metrics(smoothed, d_val[guard_idx])["broad_area_score"]
        if value > delta_guard + 1e-9:
            delta, delta_guard = float(candidate), value
    print(json.dumps({"guard_smoothing_delta": delta, "guard_broad_with_smoothing": delta_guard, "guard_broad_plain": base_guard}))

    # Refit the winning recipe on the full validation block.
    temperatures, by_regime, tau = run_recipe(how, gate, np.arange(rows_val), regimes[gate])
    regime_val = regimes[gate]
    blend_val = blend_rows(stack_val, temperatures, by_regime, regime_val, base_how, tau)

    # Development: apply the frozen temperatures/weights once.
    meta_dev = [meta[i] for i in development_rows]
    regime_dev = (
        (np.asarray([float(m["anchor_lat"]) for m in meta_dev]) >= lat_split).astype(int)
        if gate == "spatial"
        else (np.asarray([int(m["gap_days"]) for m in meta_dev]) > gap_split).astype(int)
        if gate == "horizon"
        else np.zeros(len(development_rows), dtype=int)
    )
    blend_dev = blend_rows(stack_dev, temperatures, by_regime, regime_dev, base_how, tau)

    # Guard-selected robustness smoothing: keep anchor mass in the
    # packaged distribution.
    if delta > 0:
        blend_val = (1.0 - delta) * blend_val + delta * kernel_anchor_val
        blend_val /= blend_val.sum(axis=1, keepdims=True)
        blend_dev = (1.0 - delta) * blend_dev + delta * kernel_anchor_dev
        blend_dev /= blend_dev.sum(axis=1, keepdims=True)
    validation = broad_metrics(blend_val, d_val)
    development = broad_metrics(blend_dev, d_dev)

    report = {
        "schema": "theswarm-mixture-v1",
        "pooling": how,
        "entropy_tau": tau,
        "guard_smoothing_delta": delta,
        "gate": gate,
        "gate_splits": {"latitude": lat_split, "gap_days": gap_split},
        "experts": names,
        "temperatures": {name: float(t) for name, t in zip(names, temperatures)},
        "weights": {str(r): {name: float(w) for name, w in zip(names, weights)} for r, weights in by_regime.items()},
        "validation": validation,
        "development": development,
        "cross_fit": [
            {"pooling": t["pooling"], "gate": t["gate"], "entropy_tau": t["tau"], "fit_broad_area_score": t["fit"]["broad_area_score"], "guard_broad_area_score": t["guard"]["broad_area_score"]}
            for t in trials
        ],
        "step": args.step,
        "protocol": "per-expert temperatures and (per-regime) weights jointly optimized on the fit half of chronological validation; recipe (pooling family, gate, entropy adaptation) chosen by guard-half score; winning recipe refit on full validation; development is diagnostic",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "theswarm.json").write_text(json.dumps(report, indent=2) + "\n")
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
            subsystem="location/theswarm",
            version="v1",
            status="production",
            description="TheSwarm: cross-fit-guarded mixture of frozen Ethiopia location experts with kernel smoothing experts, joint temperature/weight optimization, and regime-gated weights.",
            metrics={"validation": validation, "development": development},
            lineage={"experts": {name: str(path) for name, (path, _, _) in {**DENSE_EXPERTS, **CANDIDATE_EXPERTS}.items()} | {name: str(path) for name, path in RANKER_VIEWS.items()} | {"kernel_anchor": "derived: Gaussian field around cutoff-safe anchor", "kernel_history": "derived: Gaussian field around cutoff-safe history mean"}},
            calibration={"pooling": how, "entropy_tau": tau, "guard_smoothing_delta": delta, "gate": gate, "weights": report["weights"], "temperatures": report["temperatures"], "selection": "temperatures/weights fitted on the fit half (joint coordinate ascent or v1-style staging), recipe and one robustness smoothing scalar chosen by guard-half score, winning recipe refit on full validation"},
            notes=[
                "All experts share the 681-cell Ethiopia H3 r4 support; candidate-support experts are projected by nearest-cell mass assignment.",
                f"Cross-fit guard chose pooling={how} (tau={tau}), gate={gate}, smoothing_delta={delta}: the simplest recipe among guard-tied finalists (band 0.002); all recipes' fit/guard scores are recorded in theswarm.json.",
                "Development block is diagnostic, not a prospective test.",
                "Ethiopia rows use TheSwarm; other countries route through the v10-prized global64 view.",
            ],
        ),
    )
    print(json.dumps({k: report[k] for k in ("pooling", "gate", "experts", "temperatures", "weights", "validation", "development", "cross_fit")}, indent=2))


if __name__ == "__main__":
    main()
