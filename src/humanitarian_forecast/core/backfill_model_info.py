#!/usr/bin/env python3
"""Backfill info.blt files for model artifacts that predate the structured model store."""

from __future__ import annotations

from pathlib import Path

from humanitarian_forecast.core.model_store import ModelInfo, metric_payload, write_info
from humanitarian_forecast.core.paths import PATHS


VARIANTS = {
    "location/candidate_ranker/v1": "candidate-ranker-v1",
    "location/candidate_ranker/v2": "identity-aware",
    "location/candidate_ranker/v3": "identity-aware-distance-weight-10",
    "location/candidate_ranker/v4": "identity-aware-distance-weight-50",
    "location/candidate_ranker/v5": "weighted-center",
    "location/candidate_ranker/v6": "weighted-center-64",
    "location/candidate_ranker/v7": "rolling-hotspot",
    "location/candidate_ranker/v8": "all-event-activity-fatality",
    "location/candidate_ranker/v9": "spatial-rings-geometric-median",
    "location/candidate_center/v10": "candidate-conditioned-residual-center",
    "location/mixture/v1": "five-circle-mixture",
    "location/mixture/v2": "absolute-geography-season",
    "location/mixture/v3": "movement-features",
    "location/mixture/v4": "ranking-auxiliary",
    "location/direct/v1": "direct-probabilistic-center",
    "location/grid/v1": "hierarchical-grid",
    "location/history_ranker/v1": "history-candidate-ranker",
    "location/center_distance/v1": "direct-center-distance",
    "risk/base/v1": "international-base-legacy",
    "risk/base/v2": "international-base",
    "risk/ethiopia/v1": "ethiopia-legacy",
    "risk/ethiopia/v2": "ethiopia-v2",
    "risk/ethiopia/v3": "ethiopia-v3",
    "risk/ethiopia/v4": "ethiopia-v4",
}


def subsystem_for(relative: Path) -> str:
    return "/".join(relative.parts[:-1])


def summary_for(relative: Path, metrics: dict) -> dict:
    key = relative.as_posix()
    if key == "location/candidate_ranker/v9":
        calibration = metrics.get("candidate_ranker_calibration_metrics.json", {})
        return {
            "headline": calibration.get("untouched_test", {}),
            "validation": calibration.get("validation", {}),
            "selection_protocol": calibration.get("selection_protocol"),
            "raw": metrics,
        }
    for value in metrics.values():
        if isinstance(value, dict) and "untouched_test" in value:
            return {"headline": value.get("untouched_test", {}), "raw": metrics}
    for value in metrics.values():
        if isinstance(value, dict) and any(
            key in value for key in ("average_precision", "median_error_km", "top1_median_error_km")
        ):
            return {"headline": value, "raw": metrics}
    return {"raw": metrics}


def main() -> None:
    for directory in sorted(path for path in PATHS.models.rglob("v*") if path.is_dir()):
        relative = directory.relative_to(PATHS.models)
        if len(relative.parts) < 3:
            continue
        version = relative.parts[-1]
        key = relative.as_posix()
        metrics = metric_payload(directory)
        status = "historical"
        notes: list[str] = []
        calibration: dict = {}
        if key == "location/candidate_ranker/v9":
            status = "validated-promoted"
            notes.append("Validated predecessor of the all-data production retrain; untouched-test median center error is 61.01 km.")
            calibration = {"aggregation": "weighted_geometric_median", "temperature": 1.0}
        elif key in {"risk/base/v2", "risk/ethiopia/v4"}:
            status = "validated-reference"
        elif key.startswith("location/hierarchical/"):
            status = "historical-publish-artifact"
            notes.append("No colocated metrics file was present in the legacy publish directory.")

        variant = VARIANTS.get(key, key.replace("/", "-"))
        write_info(directory, ModelInfo(
            subsystem=subsystem_for(relative),
            version=version,
            status=status,
            description=f"Migrated legacy model artifact: {variant}.",
            metrics=summary_for(relative, metrics),
            lineage={"migration": "pre-refactor checkpoint tree", "legacy_variant": variant},
            calibration=calibration,
            notes=notes,
        ))
        print(directory.relative_to(PATHS.root))


if __name__ == "__main__":
    main()
