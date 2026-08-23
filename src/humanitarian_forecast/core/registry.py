from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SystemSpec:
    name: str
    module: str
    category: str
    description: str
    production: bool = False


_SYSTEMS: dict[str, SystemSpec] = {}


def register(spec: SystemSpec) -> SystemSpec:
    if spec.name in _SYSTEMS:
        raise ValueError(f"system already registered: {spec.name}")
    _SYSTEMS[spec.name] = spec
    return spec


def get_system(name: str) -> SystemSpec:
    try:
        return _SYSTEMS[name]
    except KeyError as exc:
        known = ", ".join(sorted(_SYSTEMS))
        raise KeyError(f"unknown system {name!r}; known systems: {known}") from exc


def all_systems() -> list[SystemSpec]:
    return [_SYSTEMS[name] for name in sorted(_SYSTEMS)]


def _builtin(name: str, module: str, category: str, description: str, production: bool = False) -> None:
    register(SystemSpec(name, module, category, description, production))


_builtin("data.reliefweb.download", "humanitarian_forecast.data.download_reliefweb", "data", "Download the full ReliefWeb conflict-report corpus.", True)
_builtin("data.risk.build", "humanitarian_forecast.data.build_risk_datasets", "data", "Build aligned global and Ethiopia temporal-risk datasets.", True)
_builtin("data.risk.spatial.build", "humanitarian_forecast.data.build_spatial_risk_dataset", "data", "Add cutoff-safe PRIO-neighborhood context to a coarse risk dataset.")
_builtin("data.geo.elevation.build", "humanitarian_forecast.data.build_elevation_context", "data", "Materialize coarse Copernicus GLO-90 terrain context for candidate locations.")
_builtin("data.location.geo.augment", "humanitarian_forecast.data.augment_candidate_geo_context", "data", "Append materialized static geo context to a candidate-ranking dataset.")
_builtin("data.humanitarian.prio_context.build", "humanitarian_forecast.data.build_prio_humanitarian_geo_context", "data", "Materialize coarse PRIO-grid population or accessibility context for humanitarian-risk models.")
_builtin("data.humanitarian.geo.augment", "humanitarian_forecast.data.augment_humanitarian_geo_context", "data", "Append coarse static population/accessibility context to a humanitarian-risk tensor.")
_builtin("risk.train.evaluate", "humanitarian_forecast.risk.train", "risk", "Chronological holdout trainer for measured risk-model experiments.")
_builtin("risk.train.challenger", "humanitarian_forecast.risk.train_challenger", "risk", "Train a multiscale coarse humanitarian-risk challenger with chronological evaluation.")
_builtin("risk.train.v6", "humanitarian_forecast.risk.train_v6", "risk", "Train the v6 spatially contextualized coarse humanitarian-risk bundle.")
_builtin("risk.train.transfer", "humanitarian_forecast.risk.train_transfer_challenger", "risk", "Train a coarse humanitarian-risk challenger with an explicit masked-local cold-start transfer objective.")
_builtin("risk.train.cold_start_tree", "humanitarian_forecast.risk.train_cold_start_transfer_tree", "risk", "Train a non-local humanitarian-risk specialist selected on chronologically later held-out PRIO entities.")
_builtin("risk.train.context_fusion", "humanitarian_forecast.risk.train_context_fusion", "risk", "Train separated dynamic/static context fusion over coarse spatial history plus population/accessibility.")
_builtin("risk.ensemble.context", "humanitarian_forecast.risk.ensemble_context_experts", "risk", "Validation-only proper-score ensemble of coarse humanitarian context experts.")
_builtin("risk.predict", "humanitarian_forecast.risk.predict", "inference", "Run calibrated coarse humanitarian-risk inference from a versioned challenger.")
_builtin("risk.predict.v6", "humanitarian_forecast.risk.predict_v6", "inference", "Run v6 calibrated spatial humanitarian-risk inference with uncertainty.")
_builtin("risk.train.production", "humanitarian_forecast.risk.production", "risk", "Train global or Ethiopia risk models on every available row.", True)
_builtin("location.dataset.build", "humanitarian_forecast.data.build_next_location_dataset", "location", "Build next-event location histories.", True)
_builtin("location.candidates.build", "humanitarian_forecast.data.build_candidate_rank_dataset", "location", "Build cutoff-safe historical location candidates and spatial features.", True)
_builtin("location.rank.train", "humanitarian_forecast.location.training.train_candidate_ranker", "location", "Chronological candidate-ranker trainer used for evaluation.")
_builtin("location.geo_fusion.train", "humanitarian_forecast.location.training.train_geo_fusion", "location", "Train a motion-aware broad-area GeoFusion geographical challenger.")
_builtin("location.rank.train.production", "humanitarian_forecast.location.training.train_candidate_ranker_production", "location", "Train the promoted candidate-ranker recipe on all available labels.", True)
_builtin("location.rank.calibrate", "humanitarian_forecast.location.training.calibrate_candidate_center", "location", "Select candidate-center aggregation on validation data.")
_builtin("location.predict", "humanitarian_forecast.location.inference.predict_hackathon", "inference", "Run the promoted candidate-ranker location predictor.", True)
_builtin("location.direct.train", "humanitarian_forecast.location.training.train_location", "experimental", "Historical direct probabilistic location trainer.")
_builtin("location.mixture.train", "humanitarian_forecast.location.training.train_mixture_location", "experimental", "Historical multimodal location trainer.")
_builtin("evaluation.baselines", "humanitarian_forecast.evaluation.evaluate_baselines", "evaluation", "Evaluate chronological location baselines.")
_builtin("evaluation.candidate_ranker", "humanitarian_forecast.evaluation.evaluate_candidate_ranker", "evaluation", "Evaluate candidate ranking experiments.")
_builtin("evaluation.label_ambiguity", "humanitarian_forecast.evaluation.label_ambiguity_audit", "evaluation", "Audit target-day location ambiguity.")
_builtin("evaluation.risk.rolling", "humanitarian_forecast.evaluation.rolling_origin_risk", "evaluation", "Rolling-origin stability and cold-start transfer gate for coarse humanitarian-risk features.")
_builtin("evaluation.location.rolling", "humanitarian_forecast.evaluation.rolling_origin_location", "evaluation", "Rolling-origin Ethiopia broad-area gate for aligned candidate feature ablations.")
_builtin("location.broad_area.ensemble", "humanitarian_forecast.location.training.ensemble_broad_area_probabilities", "location", "Proper-score broad-area ensemble of frozen v9, actor-transfer, and propagation experts.")
