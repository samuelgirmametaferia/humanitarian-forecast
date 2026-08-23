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
_builtin("risk.train.evaluate", "humanitarian_forecast.risk.train", "risk", "Chronological holdout trainer for measured risk-model experiments.")
_builtin("risk.train.challenger", "humanitarian_forecast.risk.train_challenger", "risk", "Train a multiscale coarse humanitarian-risk challenger with chronological evaluation.")
_builtin("risk.train.v6", "humanitarian_forecast.risk.train_v6", "risk", "Train the v6 spatially contextualized coarse humanitarian-risk bundle.")
_builtin("risk.predict", "humanitarian_forecast.risk.predict", "inference", "Run calibrated coarse humanitarian-risk inference from a versioned challenger.")
_builtin("risk.predict.v6", "humanitarian_forecast.risk.predict_v6", "inference", "Run v6 calibrated spatial humanitarian-risk inference with uncertainty.")
_builtin("risk.train.production", "humanitarian_forecast.risk.production", "risk", "Train global or Ethiopia risk models on every available row.", True)
_builtin("location.dataset.build", "humanitarian_forecast.data.build_next_location_dataset", "location", "Build next-event location histories.", True)
_builtin("location.candidates.build", "humanitarian_forecast.data.build_candidate_rank_dataset", "location", "Build cutoff-safe historical location candidates and spatial features.", True)
_builtin("location.rank.train", "humanitarian_forecast.location.training.train_candidate_ranker", "location", "Chronological candidate-ranker trainer used for evaluation.")
_builtin("location.rank.train.production", "humanitarian_forecast.location.training.train_candidate_ranker_production", "location", "Train the promoted candidate-ranker recipe on all available labels.", True)
_builtin("location.rank.calibrate", "humanitarian_forecast.location.training.calibrate_candidate_center", "location", "Select candidate-center aggregation on validation data.")
_builtin("location.predict", "humanitarian_forecast.location.inference.predict_hackathon", "inference", "Run the promoted candidate-ranker location predictor.", True)
_builtin("location.direct.train", "humanitarian_forecast.location.training.train_location", "experimental", "Historical direct probabilistic location trainer.")
_builtin("location.mixture.train", "humanitarian_forecast.location.training.train_mixture_location", "experimental", "Historical multimodal location trainer.")
_builtin("evaluation.baselines", "humanitarian_forecast.evaluation.evaluate_baselines", "evaluation", "Evaluate chronological location baselines.")
_builtin("evaluation.candidate_ranker", "humanitarian_forecast.evaluation.evaluate_candidate_ranker", "evaluation", "Evaluate candidate ranking experiments.")
_builtin("evaluation.label_ambiguity", "humanitarian_forecast.evaluation.label_ambiguity_audit", "evaluation", "Audit target-day location ambiguity.")
