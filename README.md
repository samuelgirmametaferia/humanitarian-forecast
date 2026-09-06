# Humanitarian Forecast

A focused pipeline for humanitarian conflict next-location forecasting using UCDP and ReliefWeb, with Ethiopia specialization. The live package trains and serves the **candidate-ranker lineage (v9 → v10-prized)** and **TheSwarm**, the production mixture-of-experts built on top of the frozen H3-r4 expert family.

Archived code, weights, and datasets live in `legacy/`, `legacy_models/`, and `legacy_data/` (see `legacy/README.md`). Nothing in the live package imports them.

## One controller

Use `main.py` from the repository root. It adds `src/` to the Python path automatically.

```bash
# Show every registered subsystem.
.venv/bin/python main.py systems

# Inspect the complete production workflow without running it.
.venv/bin/python main.py workflow plan

# Run global all-data training, then Ethiopia fine-tuning.
.venv/bin/python main.py workflow full --skip-download

# Inspect model versions and their metadata.
.venv/bin/python main.py models list
.venv/bin/python main.py models info location/candidate_ranker v9

# Run any registered subsystem directly.
.venv/bin/python main.py run location.predict -- --index -1 --top 5
.venv/bin/python main.py run location.swarm.predict -- --index -1
```

The Ethiopia database defaults to `/Users/sam/Documents/wfp/database.sqlite`. Override it with `--ethiopia-db PATH` or the `HUMANITARIAN_ETHIOPIA_DB` environment variable.

## Production workflow

`main.py workflow full` runs eight steps:

1. Download the configured full ReliefWeb corpus (unless `--skip-download`).
2. Build international and Ethiopia risk datasets from UCDP + ReliefWeb.
3. Build the promoted 19-feature (`geo-v2`) next-location history dataset.
4. Build 32 cutoff-safe historical candidates with the spatial-ring feature system.
5. Train the international temporal-risk model on every available global row.
6. Retrain the validated v9 candidate-location recipe on every available global location label.
7. Fine-tune the temporal-risk model on all Ethiopia rows.
8. Fine-tune the candidate-location model on all Ethiopia rows while preserving global embedding IDs.

Steps can be resumed or restricted with `--from-step` / `--through-step`; `workflow plan` shows the order. Versioned model directories are protected against overwrite unless `--force` is supplied.

## The models

| Model | Path | Role |
|---|---|
| v9 candidate ranker | `models/location/candidate_ranker/v9/` | validated reference (61.01 km median center error on the fixed chronological test) |
| v10 | `models/location/candidate_ranker/v10/` | all-data v9 recipe retrain |
| v10-prized | `models/location/candidate_ranker/v10_prized/` | packaged champion: global64 + Ethiopia32 support views, coarse humanitarian zone output |
| swarm v1 | `models/location/swarm/v1/` | first validation-selected convex mixture (superseded by TheSwarm) |
| TheSwarm v1 | `models/location/theswarm/v1/` | production mixture of experts — the verified optimum of the frozen expert pool |

### TheSwarm (production mixture-of-experts)

`models/location/theswarm/v1/` blends ten frozen experts that all score the
same dense H3 r4 cell support over Ethiopia:

- dense H3 experts: `h3_lambdarank` (scratch + global-transfer), `h3_lambdarank_memory`
- classical experts: `marked_hawkes`, `shape_analogue`, `reliefweb_spatial_specialist`
- the candidate-ranker lineage projected onto the same support (v9 calibrated, v10-prized Ethiopia32)
- kernel-smoothing experts around the cutoff-safe anchor and history mean

The recipe is chosen under a chronological cross-fit guard: per-expert
temperatures and weights are fitted on the first half of the validation
block, twelve recipes (staged/joint linear, geometric product-of-experts,
entropy-adaptive pooling, spatial/horizon regime gates, anchor-kernel
smoothing) compete on the unseen second half, and the simplest recipe in
the guard-tied band wins. The winning recipe is refit on the full
validation block; the development block is scored exactly once.

The search converged on the same optimum from two independent paths
(staged fit and joint coordinate ascent): `reliefweb_full` 0.72 +
`v10_prized_ethiopia32` 0.28. Validation: broad-area 0.511, median 120.7
km (previous fixed ensemble 0.469 / 133.7 km; best single expert 0.508).
Development: broad-area 0.344, median 180.7 km. Every enhancement the
guard evaluated — gates, geometric pooling, entropy weighting, kernel
smoothing — was rejected, which pins the ceiling at the expert pool
itself: the next real gain must come from new signal (the planned
Telegram/Groq ingestion feeding the RL loop), not from richer blending.

The state file (`theswarm.json`: experts, temperatures, weights, full
recipe-search record) is small and stateless so the planned 3-day
reinforcement loop can update it cheaply.

```bash
.venv/bin/python main.py run location.theswarm.train -- --help
.venv/bin/python main.py run location.theswarm.predict -- --index -1
```

Routing at inference: Ethiopia rows use TheSwarm; all other countries fall
back to the v10-prized global64 view.

### Swarm v1 (reference)

`models/location/swarm/v1/` is the first-generation mixture (staged
temperature calibration + convex weights, no cross-fit guard). TheSwarm
reproduces its blend exactly while proving it optimal over a much larger
recipe space; it is kept as the regression reference.

## Project layout

```text
.
├── main.py                         # single project controller
├── pyproject.toml                  # installable src-layout package
├── src/humanitarian_forecast/
│   ├── cli.py
│   ├── core/                       # registry, paths, runner, model metadata
│   ├── data/                       # ReliefWeb/UCDP ingestion + dataset builders
│   ├── risk/                       # global/Ethiopia temporal-risk system
│   ├── location/
│   │   ├── models/                 # candidate-ranker architecture
│   │   ├── training/               # train / calibrate / materialize
│   │   ├── ensemble/               # swarm mixture-of-experts
│   │   └── inference/              # promoted inference paths
│   ├── evaluation/                 # candidate-ranker evaluation
│   └── workflows/                  # composed end-to-end workflow
├── models/location/                # live lineage + swarm + frozen experts
├── legacy/                         # archived source (see legacy/README.md)
├── legacy_models/                  # archived weights (~440 MB)
├── legacy_data/                    # archived datasets (~11 GB, git-ignored)
├── data/                           # canonical datasets (git-ignored)
└── reports/                        # evaluation reports
```

## Model store and `info.blt`

Every model version lives in its own directory with `info.blt`, a machine-readable JSON metadata file: subsystem, version, status, measured performance where an honest held-out result exists, lineage, calibration, artifacts, and notes/warnings. Production artifacts trained on all data do not claim a fresh holdout score; their `info.blt` records the measured performance of their validated ancestors.

## Evaluation versus production

Chronological holdout trainers stay in the package for honest model selection and regression testing. Production trainers consume all labeled data only after a recipe has been selected. Do not compare a production training loss to the v9 test error.

## Safety and interpretation

The outputs are probabilistic humanitarian research signals, not verified front lines, safe routes, evacuation orders, or individual/unit tracking. Operational use requires independent current-source corroboration and human review.
