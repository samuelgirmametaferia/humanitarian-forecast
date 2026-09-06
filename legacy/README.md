# Legacy archive

This directory holds source modules removed from the live package during the
2026-09 consolidation. Nothing under `legacy/` is imported by the live code.

Contents (mirroring the old package layout under `legacy/src/humanitarian_forecast/`):

- `experimental/` — early RL trainer, warzone predictor, and mixture-location trainer.
- `location/training/` — the full challenger family: geo_lambdarank (+kernel/local/relative/
  reliefweb variants), h3_lambdarank (+actor/memory/global-pretrain), geostate, marked_hawkes,
  shape_analogue, multiradius/policy-aligned/relative refiners, child gates/rankers,
  v9->v10 adapters, and the ensemble tooling those trainers used.
- `location/models/` — architectures for the archived trainers (mixture, hierarchical,
  geostate, geo_fusion, direct, grid, history_candidate, candidate_center).
- `location/inference/` — predictors for the archived families (mixture, location, child).
- `evaluation/` — baselines, ambiguity/identifiability audits, rolling-origin harnesses.
- `data/` — one-off builders/augmenters (H3 multiscale, CHIRPS, VIEWS, elevation, OSM,
  ReliefWeb spatial mentions, actor/geo candidate augmentation).
- `web/` — the old Flask server (served the obsolete hierarchical model; the future
  interface is a separate Node application).

Companion archives:

- `legacy_models/` — model weights for everything except the live lineage
  (candidate_ranker v9/v10/v10_prized, candidate_ranker_ethiopia, and the frozen
  H3-r4 experts the swarm loads).
- `legacy_data/` — ~11 GB of experiment datasets and ReliefWeb corpus variants
  (git-ignored).

To revive anything: `git log --follow` on the moved file, or copy it back from here.
