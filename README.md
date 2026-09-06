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
| TheSwarm fine v1 | `models/location/theswarm/fine_v1/` | first 20 km multi-candidate layer (single v11 ranker) |
| TheSwarm fine v2 | `models/location/theswarm/fine_v2/` | 20 km multi-candidate layer: v11 ensemble + greedy zone-diversity selection (current) |

### TheSwarm fine layer (20 km multi-candidate output)

`models/location/theswarm/fine_v1/` answers the evacuation-scale question:
which specific points, with what probabilities, should carry 20 km advisory
zones. It is the v11 candidate ranker trained on the v6 dataset
(`location.candidates.build.v6`), which doubles the candidate pool from 32
conflict-frequency sites to 64 by adding the most recent cross-conflict
event sites (spillover), and adds ReliefWeb mention counts, UCDP
any-conflict activity windows, and per-candidate elevation/ruggedness from
the AWS terrain grid (`data.elevation.build`).

Coverage was the binding constraint at 20 km scale: the frequency-only pool
put only 47.4% of Ethiopia validation truths within 20 km of *any*
candidate; spillover candidates lift that oracle to 67.3% (55.3% on the
untouched development block). Ranking then remains hard. The packaged v2
layer blends the hard-CE v11 ranker with a LambdaRank variant (mean of
softmax) and emits zones greedily with a 30 km spatial-spread discount so
advisory zones cover distinct areas instead of stacking on one cluster.
Ethiopia validation: within-20km 13.4% at top-1, **33.0% for the diverse
5-zone emission set** (44.3% at 10 zones), median top-1 error 124.9 km.
Development (untouched): 4.2% / 15.8% / 24.8%.

Ablations that did **not** beat the packaged v11 ensemble and are kept only
as evaluation variants: distance-softened labels
(`location.rank.train.soft`), an Ethiopia-only fine-tune
(`location.rank.train.ethiopia_ft`), a two-stage shortlist cascade
(`location.rank.train.cascade`, top-1 0.112 vs 0.134), the 96-candidate v7
dataset (`location.candidates.build.v7` — oracle rises to 0.744 with
ReliefWeb mention-site candidates and Hawkes kernels, but ranking dilutes;
realized accuracy matched, not beat, the 64-candidate ensemble), and
5-model seed ensembling (0.131/0.310 — no gain over the 2-model blend).

Half of the remaining gap is data, not model: ~50% of Ethiopia validation
targets are geolocated by UCDP to a named-place radius (`where_prec >= 2`),
so 20 km hits on those rows are partly coordinate noise — a ceiling no
ranker can train through.

```bash
.venv/bin/python main.py run location.theswarm.fine.predict -- --index -1
```

Output: ranked candidate points, each with probability, coordinates,
distance from the last event, site type (frequency site vs recent event),
days since the last event at that site, local elevation/ruggedness, and a
20 km advisory radius. Research signal, not a tactical coordinate
forecast.

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
