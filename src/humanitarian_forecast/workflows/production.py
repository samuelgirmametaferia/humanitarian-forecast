from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from humanitarian_forecast.core.paths import PATHS
from humanitarian_forecast.core.registry import get_system
from humanitarian_forecast.core.runner import run_module


@dataclass(frozen=True)
class PipelineStep:
    name: str
    args: tuple[str, ...]
    label: str


@dataclass
class ProductionConfig:
    ethiopia_db: Path
    ucdp: Path = PATHS.raw / "ged261-csv.zip"
    reliefweb: Path = PATHS.data / "reliefweb_full.jsonl.gz"
    processed_dir: Path = PATHS.data / "processed_production"
    location_history: Path = PATHS.location_data / "next_location_production_geo_v2.npz"
    location_candidates: Path = PATHS.location_data / "conflict_candidates_production_spatial_v5.npz"
    start_year: int = 1980
    end_year: int = datetime.now(timezone.utc).year
    base_version: str = "v3"
    ethiopia_version: str = "v5"
    location_version: str = "v10"
    location_ethiopia_version: str = "v10"
    base_epochs: int = 20
    ethiopia_epochs: int = 20
    candidate_epochs: int = 12
    candidate_ethiopia_epochs: int = 4
    skip_download: bool = False
    dry_run: bool = False
    force: bool = False
    from_step: str | None = None
    through_step: str | None = None

    @property
    def base_model_dir(self) -> Path:
        return PATHS.models / "risk" / "base" / self.base_version

    @property
    def ethiopia_model_dir(self) -> Path:
        return PATHS.models / "risk" / "ethiopia" / self.ethiopia_version

    @property
    def location_model_dir(self) -> Path:
        return PATHS.models / "location" / "candidate_ranker" / self.location_version

    @property
    def location_ethiopia_model_dir(self) -> Path:
        return PATHS.models / "location" / "candidate_ranker_ethiopia" / self.location_ethiopia_version


def build_plan(config: ProductionConfig) -> list[PipelineStep]:
    steps: list[PipelineStep] = []
    if not config.skip_download:
        steps.append(
            PipelineStep(
                "data.reliefweb.download",
                (
                    "--output", str(config.reliefweb),
                    "--start-year", str(config.start_year),
                    "--end-year", str(config.end_year),
                ),
                "Download the complete configured ReliefWeb corpus",
            )
        )

    # Build shared corpora, train global systems, then specialize both systems to Ethiopia.
    steps.extend(
        [
            PipelineStep(
                "data.risk.build",
                (
                    "--reliefweb", str(config.reliefweb),
                    "--ucdp", str(config.ucdp),
                    "--ethiopia-db", str(config.ethiopia_db),
                    "--output-dir", str(config.processed_dir),
                ),
                "Build international and Ethiopia temporal-risk datasets",
            ),
            PipelineStep(
                "location.dataset.build",
                (
                    "--ucdp", str(config.ucdp),
                    "--reliefweb", str(config.reliefweb),
                    "--output", str(config.location_history),
                    "--feature-set", "geo-v2",
                ),
                "Build the promoted 19-feature next-location histories",
            ),
            PipelineStep(
                "location.candidates.build",
                (
                    "--data", str(config.location_history),
                    "--output", str(config.location_candidates),
                    "--candidates", "32",
                    "--ucdp-events", str(config.ucdp),
                ),
                "Build cutoff-safe 32-location spatial-ring candidates",
            ),
            PipelineStep(
                "risk.train.production",
                (
                    "--stage", "base",
                    "--data", str(config.processed_dir / "base.npz"),
                    "--output-dir", str(config.base_model_dir),
                    "--version", config.base_version,
                    "--reference-model-dir", str(PATHS.models / "risk" / "base" / "v2"),
                    "--epochs", str(config.base_epochs),
                ),
                "Train the international temporal-risk model on every available row",
            ),
            PipelineStep(
                "location.rank.train.production",
                (
                    "--data", str(config.location_candidates),
                    "--output-dir", str(config.location_model_dir),
                    "--version", config.location_version,
                    "--reference-model-dir", str(PATHS.models / "location" / "candidate_ranker" / "v9"),
                    "--epochs", str(config.candidate_epochs),
                    "--distance-weight", "0",
                    "--center-weight", "50",
                    "--temperature", "1.0",
                    "--aggregation", "weighted_geometric_median",
                ),
                "Retrain the validated v9 location recipe on every available conflict label",
            ),
            PipelineStep(
                "risk.train.production",
                (
                    "--stage", "ethiopia",
                    "--data", str(config.processed_dir / "ethiopia.npz"),
                    "--output-dir", str(config.ethiopia_model_dir),
                    "--version", config.ethiopia_version,
                    "--resume", str(config.base_model_dir / "model.pt"),
                    "--reference-model-dir", str(PATHS.models / "risk" / "ethiopia" / "v4"),
                    "--epochs", str(config.ethiopia_epochs),
                    "--ranking-weight", "0.2",
                ),
                "Fine-tune the international temporal-risk model on all Ethiopia rows",
            ),
            PipelineStep(
                "location.rank.calibrate.production",
                (
                    "--output-dir", str(config.location_ethiopia_model_dir),
                    "--version", config.location_ethiopia_version,
                    "--resume", str(config.location_model_dir / "model.pt"),
                    "--country", "Ethiopia",
                    "--reference-model-dir", str(PATHS.models / "location" / "candidate_ranker" / "v9"),
                    "--temperature", "0.8",
                    "--aggregation", "weighted_geometric_median",
                    "--history-reference", "recent_8_mean",
                    "--candidate-weight", "0.5",
                    "--selected-on", "Ethiopia 70-85%: minimum mean error subject to no validation regression at <=50 km and <=100 km; frozen before final-15% check",
                ),
                "Materialize Ethiopia-calibrated v10 from the all-data v9-lineage weights",
            ),
        ]
    )
    return steps


def _slice_plan(plan: list[PipelineStep], start: str | None, stop: str | None) -> list[PipelineStep]:
    keys = [f"{index}:{step.name}" for index, step in enumerate(plan, 1)]
    names = [step.name for step in plan]

    def resolve(value: str, last: bool = False) -> int:
        if value in keys:
            return keys.index(value)
        matches = [i for i, name in enumerate(names) if name == value]
        if not matches:
            raise ValueError(f"workflow step {value!r} not found; use `main.py workflow plan` to see step keys")
        return matches[-1] if last else matches[0]

    first = resolve(start) if start else 0
    last = resolve(stop, last=True) + 1 if stop else len(plan)
    if first >= last:
        raise ValueError("selected workflow range is empty or reversed")
    return plan[first:last]


def _check_version_targets(config: ProductionConfig, plan: list[PipelineStep]) -> None:
    if config.force or config.dry_run:
        return
    target_by_label = {
        "Train the international temporal-risk model on every available row": config.base_model_dir,
        "Retrain the validated v9 location recipe on every available conflict label": config.location_model_dir,
        "Fine-tune the international temporal-risk model on all Ethiopia rows": config.ethiopia_model_dir,
        "Materialize Ethiopia-calibrated v10 from the all-data v9-lineage weights": config.location_ethiopia_model_dir,
    }
    collisions = [target_by_label[step.label] for step in plan if step.label in target_by_label and (target_by_label[step.label] / "model.pt").exists()]
    if collisions:
        joined = "\n  ".join(str(path) for path in collisions)
        raise FileExistsError(
            "versioned production artifacts already exist; choose new version values or pass --force:\n  " + joined
        )


def print_plan(config: ProductionConfig) -> list[PipelineStep]:
    plan = _slice_plan(build_plan(config), config.from_step, config.through_step)
    print("Production workflow:")
    for index, step in enumerate(plan, 1):
        spec = get_system(step.name)
        print(f"  {index}. {step.name} — {step.label}")
        print(f"     {spec.description}")
    return plan


def run_full_workflow(config: ProductionConfig) -> None:
    plan = print_plan(config)
    _check_version_targets(config, plan)
    print()
    for index, step in enumerate(plan, 1):
        spec = get_system(step.name)
        print(f"[{index}/{len(plan)}] {step.label}", flush=True)
        run_module(spec.module, step.args, dry_run=config.dry_run)
