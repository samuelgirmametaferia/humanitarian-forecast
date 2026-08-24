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
_builtin("data.reliefweb.merge_created", "humanitarian_forecast.data.merge_reliefweb_created_corpus", "data", "Merge publication-time ReliefWeb year shards into one deduplicated causal corpus with checksum manifest.")
_builtin("data.reliefweb.spatial_mentions.build", "humanitarian_forecast.data.build_reliefweb_spatial_mentions", "data", "Extract publication-time Ethiopia gazetteer mentions and semantic onset signals from ReliefWeb.")
_builtin("data.location.reliefweb_spatial.augment", "humanitarian_forecast.data.augment_candidate_reliefweb_spatial", "data", "Append cutoff-safe candidate-local ReliefWeb semantic kernels and onset deltas.")
_builtin("data.location.reliefweb_spatial.ablate", "humanitarian_forecast.data.ablate_reliefweb_candidate_context", "data", "Create count-only or semantic-content-only candidate ReliefWeb ablations.")
_builtin("data.risk.build", "humanitarian_forecast.data.build_risk_datasets", "data", "Build aligned global and Ethiopia temporal-risk datasets.", True)
_builtin("data.risk.spatial.build", "humanitarian_forecast.data.build_spatial_risk_dataset", "data", "Add cutoff-safe PRIO-neighborhood context to a coarse risk dataset.")
_builtin("data.geo.elevation.build", "humanitarian_forecast.data.build_elevation_context", "data", "Materialize coarse Copernicus GLO-90 terrain context for candidate locations.")
_builtin("data.location.geo.augment", "humanitarian_forecast.data.augment_candidate_geo_context", "data", "Append materialized static geo context to a candidate-ranking dataset.")
_builtin("data.location.h3.build", "humanitarian_forecast.data.build_h3_multiscale", "data", "Build dense multiresolution H3 support for broad humanitarian geographical forecasting.")
_builtin("data.location.h3.static", "humanitarian_forecast.data.build_h3_static_context", "data", "Project validated PRIO population/accessibility context onto dense H3 cells.")
_builtin("data.views.prior.download", "humanitarian_forecast.data.download_views_prior", "data", "Download one immutable VIEWS PRIO-grid production forecast prior with provenance.")
_builtin("data.views.archive.download", "humanitarian_forecast.data.download_views_archive", "data", "Probe and cache immutable monthly VIEWS production runs for causal backtesting.")
_builtin("data.views.prior.h3", "humanitarian_forecast.data.build_views_h3_prior", "data", "Project an immutable VIEWS PRIO-grid forecast prior onto dense H3 cells.")
_builtin("data.chirps.h3.build", "humanitarian_forecast.data.build_chirps_h3_context", "data", "Materialize causal monthly CHIRPS v3 rainfall context at H3 cells.")
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

_builtin("location.rank.relative.train", "humanitarian_forecast.location.training.train_geo_lambdarank_relative", "location", "Train Ethiopia-weighted LambdaRank with explicit within-query candidate rank/z-score features.")
_builtin("location.rank.multiradius.train", "humanitarian_forecast.location.training.train_geo_multiradius", "location", "Train nested 25/50/100/200 km proximity classifiers as a heterogeneous candidate ranker.")
_builtin("location.rank.alt.train", "humanitarian_forecast.location.training.train_geo_alt_rankers", "location", "Benchmark XGBoost LambdaMART and CatBoost YetiRank candidate experts.")
_builtin("location.refiner.train", "humanitarian_forecast.location.training.train_candidate_refiner", "location", "Train continuous coarse-to-fine coordinate refinement on a frozen candidate ranker.")
_builtin("location.rank.train.production", "humanitarian_forecast.location.training.train_candidate_ranker_production", "location", "Train the promoted candidate-ranker recipe on all available labels.", True)
_builtin("location.rank.calibrate", "humanitarian_forecast.location.training.calibrate_candidate_center", "location", "Select candidate-center aggregation on validation data.")
_builtin("location.rank.calibrate.production", "humanitarian_forecast.location.training.materialize_country_calibrated_ranker", "location", "Materialize a country-calibrated wrapper around an all-data candidate-ranker checkpoint.", True)
_builtin("location.v10_prized.materialize", "humanitarian_forecast.location.training.materialize_v10_prized", "location", "Package the all-data v9-family two-support v10-prized humanitarian location model.", True)
_builtin("location.v10_prized.predict", "humanitarian_forecast.location.inference.predict_v10_prized", "inference", "Run v10-prized and emit only a coarse humanitarian early-warning zone.", True)
_builtin("location.predict", "humanitarian_forecast.location.inference.predict_hackathon", "inference", "Run the promoted candidate-ranker location predictor.", True)
_builtin("location.direct.train", "humanitarian_forecast.location.training.train_location", "experimental", "Historical direct probabilistic location trainer.")
_builtin("location.mixture.train", "humanitarian_forecast.location.training.train_mixture_location", "experimental", "Historical multimodal location trainer.")
_builtin("evaluation.baselines", "humanitarian_forecast.evaluation.evaluate_baselines", "evaluation", "Evaluate chronological location baselines.")
_builtin("evaluation.candidate_ranker", "humanitarian_forecast.evaluation.evaluate_candidate_ranker", "evaluation", "Evaluate candidate ranking experiments.")
_builtin("evaluation.label_ambiguity", "humanitarian_forecast.evaluation.label_ambiguity_audit", "evaluation", "Audit target-day location ambiguity.")
_builtin("evaluation.risk.rolling", "humanitarian_forecast.evaluation.rolling_origin_risk", "evaluation", "Rolling-origin stability and cold-start transfer gate for coarse humanitarian-risk features.")
_builtin("evaluation.location.rolling", "humanitarian_forecast.evaluation.rolling_origin_location", "evaluation", "Rolling-origin Ethiopia broad-area gate for aligned candidate feature ablations.")
_builtin("evaluation.location.h3_baselines", "humanitarian_forecast.evaluation.evaluate_h3_baselines", "evaluation", "Evaluate dense-H3 frequency, persistence, and recency baselines on Ethiopia chronology.")
_builtin("evaluation.views.archive", "humanitarian_forecast.evaluation.evaluate_views_archive", "evaluation", "Causally evaluate archived VIEWS production forecasts against Ethiopia next-event outcomes.")
_builtin("location.broad_area.ensemble", "humanitarian_forecast.location.training.ensemble_broad_area_probabilities", "location", "Proper-score broad-area ensemble of frozen v9, actor-transfer, and propagation experts.")
_builtin("location.shape_analogue.evaluate", "humanitarian_forecast.location.training.evaluate_shape_analogue", "location", "Evaluate a cutoff-safe 3D historical conflict-shape analogue expert.")
_builtin("location.hawkes.train", "humanitarian_forecast.location.training.train_marked_hawkes", "location", "Train a marked spatiotemporal Hawkes-style broad-area expert.")
_builtin("location.regime_gate.train", "humanitarian_forecast.location.training.train_regime_gate", "location", "Fit a validation-only contextual gate over heterogeneous broad-area experts.")
_builtin("location.geostate.pretrain", "humanitarian_forecast.location.training.pretrain_geostate", "location", "Globally pretrain the dense GeoState temporal backbone on causal candidate-ranking supervision.")
_builtin("location.geostate.train", "humanitarian_forecast.location.training.train_geostate", "location", "Train a dense multiresolution H3 GeoState geographical challenger.")
_builtin("location.h3_lambdarank.train", "humanitarian_forecast.location.training.train_h3_lambdarank", "location", "Train a dense H3 propagation LambdaRank expert over all Ethiopia cells.")
_builtin("location.h3_lambdarank_actor.train", "humanitarian_forecast.location.training.train_h3_lambdarank_actor", "location", "Train dense H3 LambdaRank with cutoff-safe cross-conflict actor geography.")
_builtin("location.h3_lambdarank_global.pretrain", "humanitarian_forecast.location.training.pretrain_h3_lambdarank_global", "location", "Pretrain the dense-H3 scoring contract on global causal replay before Ethiopia continuation.")
_builtin("location.h3_lambdarank_memory.train", "humanitarian_forecast.location.training.train_h3_lambdarank_memory", "location", "Train dense H3 LambdaRank with long-memory recurrence summaries.")
_builtin("location.h3_r4.ensemble", "humanitarian_forecast.location.training.ensemble_h3_r4", "location", "Validation-select a probability ensemble over dense H3-r4 experts.")
_builtin("location.h3.reconcile", "humanitarian_forecast.location.training.reconcile_h3_multiresolution", "location", "Reconcile multiresolution H3 probability fields into coherent coarse/fine mass.")
_builtin("evaluation.location.h3.rolling", "humanitarian_forecast.evaluation.rolling_origin_h3_lambdarank", "evaluation", "Rolling-origin Ethiopia evaluation for dense-H3 location challengers.")
_builtin("location.candidate_refiner.train", "humanitarian_forecast.location.training.train_candidate_refiner", "location", "Train continuous local residual refinement beneath a frozen coarse candidate ranker.")
_builtin("location.candidate_child_gate.train", "humanitarian_forecast.location.training.train_candidate_child_gate", "location", "Train a chronological out-of-sample gate for parent-preserving refined child hypotheses.")
_builtin("location.candidate_mdn_refiner.train", "humanitarian_forecast.location.training.train_candidate_mdn_refiner", "location", "Train a multimodal Gaussian-mixture local residual density beneath frozen coarse candidate probabilities.")
_builtin("location.child_shadow.predict", "humanitarian_forecast.location.inference.predict_child_challenger", "inference", "Run shadow-only parent+refined-child location inference without changing the production champion.")
