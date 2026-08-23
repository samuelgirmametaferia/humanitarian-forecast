# 25 km Humanitarian Location Forecast Plan

## Objective

Build a Telegram-free model that predicts multiple probability-weighted circles
for the next documented conflict location. Each prediction contains:

- center latitude and longitude
- probability
- calibrated uncertainty radius
- explicit forecast horizon
- observation cutoff and data provenance

The primary stretch target is **25 km or lower top-1 mean center error** on a
chronologically later, untouched test set. This is a target, not a current
capability claim.

Outputs are research-grade humanitarian warning signals. They are not verified
front lines, safe routes, evacuation orders, or real-time force tracking.

## Non-negotiable data policy

- Use UCDP, ReliefWeb, and versioned public geographic datasets.
- Exclude Telegram from training, validation, inference, and reported metrics.
- Never use information published after the forecast cutoff in a prospective metric.
- Never improve a headline metric by removing difficult test examples.
- Keep the international base model and Ethiopia specialization separately.
- Publish coarse, delayed predictions with uncertainty and human review.

## Current measured baseline

Recommended checkpoint:

```text
checkpoints_candidate_ranker_spatial_v9/candidate_ranker_calibrated.pt
```

Training corpus:

```text
data/location/conflict_candidates_32_spatial_v5.npz
```

Data and model summary:

- 157,932 examples from 1,517 UCDP conflicts
- 120 countries represented in usable sequences
- 16 previous events per example
- 25 features per historical event
- 32 cutoff-safe conflict-location candidates
- 28 features per candidate, including candidate-centered activity and fatality rings
- Transformer ranking followed by a validation-selected weighted geometric median
- 23,690 examples in the chronologically later untouched test split

Untouched-test results:

| Metric | Current | Target |
|---|---:|---:|
| Top-1 mean error | 194.4 km | 25 km |
| Top-1 median error | 61.0 km | 15 km |
| Top-1 p90 error | 435.1 km | 75 km |
| Top-1 within 25 km | 34.4% | — |
| Best-of-32 mean error | 36.0 km | diagnostic only |

The model is **169.4 km above the primary target**. Reaching 25 km will require a
different spatial formulation, better geographic context, cleaner labels, and
stronger validation—not a threshold adjustment.

## Experiment record

| Experiment | Test mean | Decision |
|---|---:|---|
| Original five-circle mixture | 250.2 km | Superseded |
| Absolute geography + season | 245.3 km | Promoted |
| Per-step movement features | 250.1 km | Rejected |
| First auxiliary ranking loss | 267.1 km | Rejected |
| Full-history nearest transition | 247.8 km | Rejected |
| Direct center-distance Transformer | 270.5 km validation | Stopped/rejected |
| Validation-shrunk history mean | 235.9 km | Best cutoff-safe benchmark |
| Country-bank candidate snap, coarse blend | 220.0 km | Superseded by refined validation search |
| Country-bank candidate snap, refined blend | 220.6 km | Promoted center benchmark |
| Frequency-aware country candidate ranker | 219.9 km | Promoted center benchmark |
| Identity-aware top-32 candidate ranker | 210.0 km | Superseded |
| End-to-end weighted candidate center | 199.4 km | Promoted prospective center |
| Rolling-hotspot weighted center | 195.3 km | Superseded on validation |
| All-event activity/fatality center | 195.9 km | Promoted by validation |
| Candidate-centered spatial rings | 195.4 km | Superseded by calibrated aggregation |
| Validation-calibrated geometric median | 194.4 km | Promoted prospective center |
| Cutoff-safe history-center ensemble | 205.7 km validation | Rejected; selected zero history weight |
| Candidate-conditioned residual center | 208.1 km validation | Rejected before test |
| 30-day post-cutoff-assisted center | 195.6 km | Retrospective-only demo |
| Top-32 high-recall candidate set | 36.0 km best-of-set | Hackathon multi-location output |

Keep rejected checkpoints for reproducibility, but do not use them for forecasts.

The 235.9 km benchmark uses a 5% validation-selected shrinkage toward the most
recent location and 95% of the mean of the 16 prior locations. Its untouched-test
95% bootstrap interval is 230.2–242.3 km. It is a benchmark, not a calibrated
multi-circle forecast checkpoint.

A diagnostic oracle that may select the closest of the 16 historical locations
after seeing the target still has 49.5 km test mean error and reaches 25 km on
66.5% of examples. Therefore the existing candidate/history representation alone
cannot meet the 25 km headline target; new candidate locations and geographic
signals are required.

The previous center benchmark uses the existing mixture top-1 center blended
80%/20% with the shrunk history mean, then selects the nearest location from the
country's cutoff-safe historical candidate bank. The blend and bank were chosen
by validation mean error. On untouched test it records 220.6 km mean, 77.1 km
median, 494.4 km p90, and 26.4% within 25 km. The 95% bootstrap interval for mean
error is 214.5–226.3 km. This candidate snap is not yet integrated with the five
calibrated uncertainty circles.

The current promoted center benchmark adds a validation-selected 7.5 km bonus
per log unit of a candidate's cutoff-safe country frequency. Conflict frequency
and anchor-distance weights were rejected by validation. Untouched-test mean is
219.9 km (95% bootstrap interval 213.9–225.8), median is 77.7 km, p90 is
495.6 km, and 27.9% of predictions are within 25 km.

Expanded-bank diagnostics show that the candidate-generation problem is now
tractable: the conflict-history oracle is 10.3 km mean with 91.5% within 25 km,
and the country-history oracle is 3.1 km mean with 97.4% within 25 km. These are
retrospective recall diagnostics, not forecast claims. Candidate ranking—not
candidate availability—is now the primary modeling bottleneck.

Hackathon packaging is documented in `HACKATHON_README.md`. The 36.0 km result
is explicitly a best-of-32 high-recall metric and must never be described as
top-1 accuracy. The validation-promoted strict-cutoff top-1 center is 194.4 km,
with 61.0 km median error and 34.4% of all fixed-test examples within 25 km. A
post-cutoff-assisted mode remains retrospective and is not deployable for
forecasts of genuinely future events.

## Ordered implementation roadmap

### 1. Freeze a trustworthy evaluation harness

This must happen before further architecture tuning.

1. Create fixed train, validation, and test manifests containing example IDs,
   cutoff dates, target dates, country, conflict ID, horizon, and label precision.
2. Hash and save the manifests so future runs use identical examples.
3. Add rolling-origin backtests with at least five historical cutoff dates.
4. Report results separately for 1–3, 4–7, 8–14, 15–30, and 31–90 day horizons.
5. Report results by country, Ethiopia/non-Ethiopia, violence type, UCDP location
   precision, urban/rural setting, and source coverage.
6. Add last-location, constant-velocity, historical-grid-frequency, and nearest
   conflict-cluster baselines.
7. Bootstrap confidence intervals for every headline metric.

Deliverables:

```text
evaluation/build_manifests.py
evaluation/evaluate_location.py
evaluation/baselines.py
data/splits/location_split_v1.json
reports/location_baseline_v1.json
```

Acceptance rule: never promote a checkpoint based on the untouched test split.
Choose it using validation/backtests, then evaluate test once.

### 2. Make forecast horizon a first-class input

The predictor now accepts `forecast_horizon_days`, but training and evaluation
must treat horizon explicitly.

1. Store horizon separately rather than only encoding it through event recency.
2. Add a learned horizon embedding plus numerical log-horizon feature.
3. Train conditional forecasts at 1, 3, 7, 14, 30, 60, and 90 days.
4. Mask events and ReliefWeb reports after the forecast cutoff.
5. Calibrate probabilities and radii independently for each horizon bucket.
6. Default humanitarian displays to 7 days while preserving all research outputs.

### 3. Improve label quality without hiding hard cases

UCDP location precision can be broader than the 25 km target.

1. Convert UCDP precision fields into a label-uncertainty radius.
2. Train with uncertainty-aware targets instead of treating every coordinate as
   equally exact.
3. Preserve all qualifying examples in the primary test result.
4. Also publish a precise-label subset result to distinguish model error from
   annotation uncertainty.
5. Collapse duplicate records for the same underlying event before splitting.
6. Keep every duplicate cluster entirely inside one temporal split.
7. Audit the largest 500 errors for bad coordinates, date ambiguity, duplicates,
   cross-border conflicts, and legitimately multimodal outcomes.

Deliverables:

```text
data/quality/location_label_audit.csv
data/quality/duplicate_clusters.parquet
reports/error_audit_top500.md
```

### 4. Build geographic context features

Add only versioned sources whose historical availability and license are recorded.

Static features:

- administrative boundaries and adjacency
- populated places and population density
- road density, intersections, and travel-time connectivity
- terrain elevation, slope, and ruggedness
- rivers, lakes, and crossing points
- distance to national and regional borders
- distance to major cities, airports, and humanitarian facilities
- land cover and settlement type

Dynamic features available before cutoff:

- event counts and fatalities in concentric 25/50/100/250 km rings
- time since last event by violence type
- movement vector and dispersion over 7/30/90 days
- neighboring-cell activity and direction of spread
- ReliefWeb topic counts and semantic embeddings
- displacement and access-constraint indicators

Implementation steps:

1. Select one canonical grid system and document its resolution.
2. Download and checksum every raw geographic release.
3. Precompute static grid features once.
4. Compute dynamic features using cutoff-safe rolling windows.
5. Fit normalization statistics on training data only.
6. Add missing-value flags rather than silently using zero.
7. Run feature ablations; retain only features that improve rolling backtests.

### 5. Replace direct global offsets with a hierarchical model

The current model predicts unrestricted east/north offsets. A hierarchical density
model should better represent geography and multiple plausible destinations.

Proposed architecture:

1. **History encoder:** Transformer over event, ReliefWeb, horizon, and geographic
   features.
2. **Candidate generator:** current cell, neighboring cells, historically active
   conflict cells, connected settlements, and a global fallback set.
3. **Coarse classifier:** probability distribution over candidate destination cells.
4. **Local offset head:** latitude/longitude offset inside each selected cell.
5. **Uncertainty head:** aleatoric radius for each component.
6. **Probability calibrator:** validation-fitted temperature or isotonic mapping.

Use learned embeddings for country, conflict, coarse grid, violence type, and
horizon. Unseen identities must map to an explicit unknown embedding. Do not allow
test identities or future event counts into the vocabulary statistics.

Loss components:

- cross-entropy for destination cell
- geodesic Huber loss for local center
- proper probabilistic density loss
- calibrated containment loss for radius
- diversity penalty to prevent five identical circles
- optional distance-aware ranking loss only after stable component assignment

### 6. Improve ReliefWeb signals

The current 2,200-report balanced sample is small and keyword-based.

1. Download the maximum historically available ReliefWeb corpus using approved
   appname `HMA-research-W5F2P`.
2. Save immutable raw responses, request parameters, timestamps, and checksums.
3. Deduplicate updates and translated copies.
4. Separate publication date, event date, and reporting period.
5. Extract country, administrative area, displacement, access, infrastructure,
   civilian harm, violence, and humanitarian-response topics.
6. Encode report text with a frozen multilingual encoder and cache embeddings.
7. Aggregate only reports available before each cutoff.
8. Compare keyword-only, embedding-only, and combined versions through ablation.

Never use the NVIDIA key to generate ground-truth coordinates. Language-model
outputs may create candidate features, but must be traceable to source text and
must not become unverified labels.

### 7. International pretraining and Ethiopia specialization

1. Train the hierarchical model on the full international UCDP corpus.
2. Save the international checkpoint before any Ethiopia specialization.
3. Fine-tune a copy using Ethiopia examples plus a replay sample from other
   countries to reduce catastrophic forgetting.
4. Oversample Ethiopia only in training; keep natural validation/test prevalence.
5. Compare global-only, Ethiopia-only, and global-then-Ethiopia checkpoints.
6. Evaluate Ethiopia across several historical periods, not one recent interval.
7. Keep a global fallback for unseen or sparse Ethiopian areas.

Reinforcement learning is not a priority. This is a supervised probabilistic
forecasting problem; RL should be considered only if a defensible reward and an
offline-policy evaluation protocol exist.

### 8. Train robustly on Apple MPS

1. Use deterministic seeds and log the exact data/model configuration.
2. Save the best validation checkpoint, last checkpoint, optimizer, scheduler,
   random state, and split-manifest hash.
3. Use mixed precision only after verifying numerical stability on MPS.
4. Add early stopping based on rolling validation geodesic error and NLL.
5. Train at least five seeds for promising configurations.
6. Ensemble only checkpoints that independently pass the validation gates.
7. Resume interrupted runs without changing the data order.
8. Write one machine-readable metrics file per run.

### 9. Calibrate probabilities and uncertainty

1. Calibrate component probabilities on validation predictions.
2. Calibrate radii separately by horizon and label-precision bucket.
3. Report observed containment at nominal 50%, 80%, 90%, and 95% levels.
4. Plot radius versus empirical error and flag undercoverage.
5. Report top-k recall within 25, 50, and 100 km.
6. Allow the model to abstain when entropy, radius, or data staleness is high.
7. Never shrink radii merely to meet the below-25-km display target.

### 10. Humanitarian map and review workflow

Each map prediction must show:

- up to five circles ordered by calibrated probability
- probability and uncertainty radius for every circle
- forecast horizon and observation cutoff
- data freshness and source coverage
- model version and evaluation status
- an explicit research-only warning

Before external use:

1. Require analyst review.
2. Require independent current-source corroboration.
3. Suppress exact coordinates and routes.
4. Delay public outputs where disclosure could increase harm.
5. Record approvals, corrections, and observed outcomes for later evaluation.

## Promotion gates

A new model becomes recommended only if it:

1. Beats the 245.3 km mean-error checkpoint on rolling validation windows.
2. Does not materially worsen median, p90, or Ethiopia-specific performance.
3. Beats the last-location and historical-frequency baselines.
4. Maintains 80–90% calibrated union-circle coverage.
5. Improves top-k recall within 25 and 50 km.
6. Shows no temporal, geographic, duplicate-event, or identity leakage.
7. Passes an error audit of its largest misses.

The 25 km goal is achieved only when the final untouched chronological test has:

- top-1 mean center error at or below 25 km
- top-1 median center error at or below 15 km
- 80–90% calibrated containment
- no subgroup with dangerously poor unreported performance
- confidence intervals and all baselines published alongside the result

## Immediate next implementation sequence

1. [ ] Freeze and hash temporal split manifests.
2. [ ] Implement strong geographic and temporal baselines.
3. [ ] Add horizon-bucket evaluation and calibration.
4. [ ] Add label-precision radii and duplicate clustering.
5. [ ] Audit the current model's 500 largest errors.
6. [ ] Choose and build the canonical spatial grid feature store.
7. [ ] Implement candidate-cell generation.
8. [ ] Implement hierarchical cell classification plus local offset heads.
9. [ ] Expand and embed the historical ReliefWeb corpus.
10. [ ] Pretrain internationally and save the new base checkpoint.
11. [ ] Fine-tune an Ethiopia copy with international replay.
12. [ ] Run five-seed rolling backtests and feature ablations.
13. [ ] Calibrate component probabilities and radii.
14. [ ] Evaluate the untouched test once and update `MODEL_CARD.md`.
15. [ ] Integrate the promoted checkpoint into the coarse humanitarian map.

## Current commands

Build the promoted 19-feature dataset:

```bash
.venv/bin/python build_next_location_dataset.py \
  --ucdp data/raw/ged261-csv.zip \
  --reliefweb data/reliefweb_balanced.jsonl.gz \
  --output data/location/next_location_geo_v2.npz
```

Train the five-circle model on Apple MPS:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python train_mixture_location.py \
  --data data/location/next_location_geo_v2.npz \
  --output-dir checkpoints_mixture_geo_v2 \
  --epochs 10 --batch-size 384 --components 5
```

Produce a seven-day scenario forecast:

```bash
.venv/bin/python predict_mixture.py \
  --data data/location/next_location_geo_v2.npz \
  --checkpoint checkpoints_mixture_geo_v2/mixture_best.pt \
  --calibration checkpoints_mixture_geo_v2/mixture_calibration.json \
  --scenario example_scenario.json \
  --index -1
```

## Final principle

Accuracy is essential because errors can affect real people. The way to pursue it
is through better evidence, spatial modeling, uncertainty calibration, and honest
future-held-out evaluation. A smaller reported number is useful only when it
represents a genuine improvement in forecasting reality.
