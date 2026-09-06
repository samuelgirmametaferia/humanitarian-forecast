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


# --- Data ---------------------------------------------------------------
_builtin("data.reliefweb.download", "humanitarian_forecast.data.download_reliefweb", "data", "Download the full ReliefWeb conflict-report corpus.", True)
_builtin("data.risk.build", "humanitarian_forecast.data.build_risk_datasets", "data", "Build aligned global and Ethiopia temporal-risk datasets.", True)
_builtin("location.dataset.build", "humanitarian_forecast.data.build_next_location_dataset", "location", "Build next-event location histories.", True)
_builtin("location.candidates.build", "humanitarian_forecast.data.build_candidate_rank_dataset", "location", "Build cutoff-safe historical location candidates and spatial features.", True)
_builtin("location.candidates.build.v6", "humanitarian_forecast.data.build_candidate_rank_dataset_v6", "location", "Build 64-candidate dataset with spillover coverage, ReliefWeb mentions, and elevation features.", True)
_builtin("location.candidates.build.v7", "humanitarian_forecast.data.build_candidate_rank_dataset_v7", "location", "Build 96-candidate dataset adding ReliefWeb mention sites and Hawkes kernel features.", True)

# --- Data pipeline helpers ------------------------------------------------
_builtin("data.elevation.build", "humanitarian_forecast.data.build_elevation_grid", "data", "Fetch the Ethiopia terrain grid (elevation + ruggedness) from AWS terrain tiles.", True)

# --- Risk ---------------------------------------------------------------
_builtin("risk.train.production", "humanitarian_forecast.risk.production", "risk", "Train global or Ethiopia risk models on every available row.", True)
_builtin("risk.predict", "humanitarian_forecast.risk.predict", "inference", "Run calibrated coarse humanitarian-risk inference from a versioned challenger.")

# --- Location: live candidate-ranker lineage -----------------------------
_builtin("location.rank.train", "humanitarian_forecast.location.training.train_candidate_ranker", "location", "Chronological candidate-ranker trainer used for evaluation.")
_builtin("location.rank.train.production", "humanitarian_forecast.location.training.train_candidate_ranker_production", "location", "Train the promoted candidate-ranker recipe on all available labels.", True)
_builtin("location.rank.calibrate", "humanitarian_forecast.location.training.calibrate_candidate_center", "location", "Select candidate-center aggregation on validation data.")
_builtin("location.rank.calibrate.production", "humanitarian_forecast.location.training.materialize_country_calibrated_ranker", "location", "Materialize a country-calibrated wrapper around an all-data candidate-ranker checkpoint.", True)
_builtin("location.v10_prized.materialize", "humanitarian_forecast.location.training.materialize_v10_prized", "location", "Package the all-data v9-family two-support v10-prized humanitarian location model.", True)

# --- Location: swarm mixture-of-experts ----------------------------------
_builtin("location.swarm.train", "humanitarian_forecast.location.ensemble.swarm", "location", "Validation-only convex mixture of frozen H3-r4 experts plus the candidate-ranker lineage.", True)
_builtin("location.swarm.predict", "humanitarian_forecast.location.inference.predict_swarm", "inference", "Run the swarm mixture and emit a coarse humanitarian early-warning zone.", True)

# --- Location: TheSwarm production mixture --------------------------------
_builtin("location.theswarm.train", "humanitarian_forecast.location.ensemble.theswarm", "location", "Cross-fit-guarded production mixture: kernel experts, joint temperature/weight fit, regime gates.", True)
_builtin("location.theswarm.predict", "humanitarian_forecast.location.inference.predict_theswarm", "inference", "Run TheSwarm and emit a coarse humanitarian early-warning zone.", True)

# --- Location: TheSwarm fine-resolution candidate layer --------------------
_builtin("location.theswarm.fine.package", "humanitarian_forecast.location.training.package_theswarm_fine", "location", "Package a v11-family candidate ranker as TheSwarm's 20 km multi-candidate fine layer.", True)
_builtin("location.theswarm.fine.predict", "humanitarian_forecast.location.inference.predict_theswarm_fine", "inference", "Emit top-k candidate points with probabilities, 20 km advisory zones, and terrain context.", True)
_builtin("location.rank.train.ethiopia_ft", "humanitarian_forecast.location.training.finetune_ethiopia_v11", "location", "Ethiopia-adapted fine-tune of the v11 spillover candidate ranker.")
_builtin("location.rank.train.soft", "humanitarian_forecast.location.training.train_candidate_ranker_soft", "location", "Distance-softened-label candidate-ranker trainer (evaluation variant).")
_builtin("location.rank.train.lambdarank", "humanitarian_forecast.location.training.train_candidate_ranker_lambdarank", "location", "LambdaRank listwise candidate-ranker trainer (evaluation variant).")
_builtin("location.rank.train.cascade", "humanitarian_forecast.location.training.train_candidate_cascade", "location", "Two-stage cascade: stage-1 shortlist, stage-2 reranker (evaluation variant).")

# --- Inference -----------------------------------------------------------
_builtin("location.v10_prized.predict", "humanitarian_forecast.location.inference.predict_v10_prized", "inference", "Run v10-prized and emit only a coarse humanitarian early-warning zone.", True)
_builtin("location.predict", "humanitarian_forecast.location.inference.predict_hackathon", "inference", "Run the promoted candidate-ranker location predictor.", True)

# --- Evaluation ----------------------------------------------------------
_builtin("evaluation.candidate_ranker", "humanitarian_forecast.evaluation.evaluate_candidate_ranker", "evaluation", "Evaluate candidate ranking experiments.")
