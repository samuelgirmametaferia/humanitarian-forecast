# Humanitarian Forecast

A structured research/production pipeline for humanitarian conflict-risk and next-location forecasting using UCDP and ReliefWeb, with Ethiopia specialization.

## One controller

Use `main.py` from the repository root. It adds `src/` to the Python path automatically, so the project does not depend on the current working directory or a collection of loose scripts.

```bash
# Show every registered subsystem.
.venv/bin/python main.py systems

# Inspect the complete production workflow without running it.
.venv/bin/python main.py workflow plan

# Run global all-data training, then Ethiopia fine-tuning.
.venv/bin/python main.py workflow full

# Reuse an already-downloaded ReliefWeb corpus.
.venv/bin/python main.py workflow full --skip-download

# Inspect model versions and their metadata.
.venv/bin/python main.py models list
.venv/bin/python main.py models info location/candidate_ranker v9

# Run any registered subsystem directly.
.venv/bin/python main.py run location.predict -- --index -1 --top 5
```

The Ethiopia database defaults to `/Users/sam/Documents/wfp/database.sqlite`. Override it with `--ethiopia-db PATH` or the `HUMANITARIAN_ETHIOPIA_DB` environment variable.

## Production workflow

`main.py workflow full` is intentionally split into global training and Ethiopia specialization:

1. Download the configured full ReliefWeb corpus (unless `--skip-download`).
2. Build international and Ethiopia risk datasets from UCDP + ReliefWeb.
3. Build the promoted 19-feature (`geo-v2`) next-location history dataset.
4. Build 32 cutoff-safe historical candidates with the spatial-ring feature system.
5. Train the international temporal-risk model on every available global row.
6. Retrain the validated v9 candidate-location recipe on every available global location label.
7. Fine-tune the temporal-risk model on every available Ethiopia row.
8. Fine-tune the candidate-location model on every Ethiopia location row while preserving the global country/conflict embedding IDs.

Production checkpoints intentionally use all currently available labels. They therefore **do not claim a fresh holdout score**. Their `info.blt` files retain the measured performance of their validated ancestors and explicitly mark the production artifact as all-data training.

Using post-cutoff rows for production means those rows become additional training examples. Feature/candidate construction remains causal for each historical example: an example never sees observations occurring after its own observation cutoff.

The workflow can be resumed or restricted with `--from-step` and `--through-step`; use `workflow plan` to see the order. Versioned model directories are protected against accidental overwrite unless `--force` is supplied.

## Project layout

```text
.
├── main.py                         # single project controller
├── pyproject.toml                  # installable src-layout package
├── src/humanitarian_forecast/
│   ├── cli.py
│   ├── core/                       # registry, paths, runners, model metadata
│   ├── data/                       # ReliefWeb/UCDP ingestion + dataset builders
│   ├── risk/                       # global/Ethiopia temporal-risk system
│   ├── location/
│   │   ├── models/                 # model architectures
│   │   ├── training/               # evaluation + all-data production trainers
│   │   └── inference/              # promoted inference paths
│   ├── evaluation/                 # baselines, audits, diagnostics
│   ├── workflows/                  # composed end-to-end workflows
│   ├── web/                        # web presentation layer
│   └── experimental/               # preserved historical/experimental systems
├── models/
│   ├── risk/base/vN/
│   ├── risk/ethiopia/vN/
│   ├── location/candidate_ranker/vN/
│   ├── location/candidate_ranker_ethiopia/vN/
│   └── ...
├── data/                           # ignored raw/processed datasets
└── reports/                        # research/evaluation reports
```

## Model store and `info.blt`

Every migrated or newly trained model version lives in its own `vN` directory. Each version contains `info.blt`, a machine-readable JSON metadata file with:

- subsystem and version
- status (`historical`, `validated-promoted`, `production-all-data`, etc.)
- measured performance where an honest held-out result exists
- training configuration and dataset lineage
- calibration configuration
- model artifact filenames
- warnings/notes about what the metrics mean

The validated next-location reference is:

```text
models/location/candidate_ranker/v9/candidate_ranker_calibrated.pt
```

Its fixed chronological test result is 61.01 km median center error, 194.36 km mean error, 435.09 km p90, and 34.39% within 25 km. It uses the spatial-ring candidate Transformer with 32 cutoff-safe candidates and validation-selected weighted geometric-median aggregation at temperature 1.0.

The latest **coarse humanitarian-risk challenger** is `models/risk/ethiopia/v6/`. It combines the calibrated v5 local-history predictor with a low-weight causal PRIO-neighborhood expert, uses a separately optimized spatial intensity model, and reports split-conformal intensity uncertainty. The final historical block is explicitly a development benchmark rather than a fresh prospective test.

```bash
# Rebuild v6's causal spatial context from the canonical Ethiopia risk tensor.
.venv/bin/python main.py run data.risk.spatial.build -- \
  --input data/processed_v3/ethiopia.npz \
  --output data/processed_v6/ethiopia_spatial.npz

# Reproduce the v6 challenger from v5 + the spatial tensor.
.venv/bin/python main.py run risk.train.v6 -- \
  --base-data data/processed_v3/ethiopia.npz \
  --spatial-data data/processed_v6/ethiopia_spatial.npz \
  --base-model-dir models/risk/ethiopia/v5 \
  --output-dir models/risk/ethiopia/v6_reproduction

# Run calibrated v6 inference on one prepared coarse-area history.
.venv/bin/python main.py run risk.predict.v6 -- \
  --model-dir models/risk/ethiopia/v6 \
  --input data/processed_v6/ethiopia_spatial.npz \
  --index 0
```

## Adding a future subsystem

Place its implementation under the appropriate `src/humanitarian_forecast/<subsystem>/` package, then register one `SystemSpec` in:

```text
src/humanitarian_forecast/core/registry.py
```

It immediately becomes visible through:

```bash
.venv/bin/python main.py systems
.venv/bin/python main.py run <new-system> -- <arguments>
```

If it belongs in the production sequence, add a `PipelineStep` in `src/humanitarian_forecast/workflows/production.py`. The controller itself does not need a new one-off command for every experiment.

## Evaluation versus production

Chronological holdout trainers remain in the package for honest model selection and regression testing. Production trainers are separate modules that consume all labeled data only after a recipe has been selected. Do not compare a production training loss to the v9 test error or treat it as a generalization metric.

## Safety and interpretation

The outputs are probabilistic humanitarian research signals, not verified front lines, safe routes, evacuation orders, or individual/unit tracking. Operational use requires independent current-source corroboration and human review.
