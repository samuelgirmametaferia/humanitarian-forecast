# The Humanitarian Forecast — Execution Plan

## Objective

## 2026-08-23 Geo Supercharge Program — ACTIVE

### Immutable restore point / non-destruction rule

Before this program began, the repository was pinned at:

```text
git tag: geo-supercharge-preflight-2026-08-23
commit:  1adf232
branch:  main
```

This tag is the emergency restore point for all pre-supercharge work. **Do not move or delete it.**

From this point forward:

- [x] Keep `models/location/candidate_ranker/v9/` untouched as the promoted geographical baseline.
- [x] Keep the temporal-risk v5/v6 artifacts; they may become auxiliary context experts.
- [x] Never overwrite or delete a trained model version merely because a newer experiment wins.
- [x] Never use `git reset --hard`, destructive cleanup, or in-place model replacement as part of model research.
- [x] Every new dataset/model is additive and gets a schema/version plus lineage back to the source data and commit.
- [x] Failed experiments may be marked rejected in metadata, but artifacts produced with compute are preserved unless the project owner explicitly asks for removal.

### Primary modeling objective

The research target is now explicitly **broad-area geographical early warning**: estimate the probability distribution over the next coarse geographic zones where organized conflict is likely to emerge, persist, or intensify. The system should learn geographic propagation rather than emit only one scalar risk value.

The public/humanitarian product should expose coarse zones/admin areas and calibrated uncertainty. Internal training may use continuous coordinates and finer cells as latent supervision, but production interfaces must not become a unit-tracking, tactical-routing, or exact strike-coordinate system.

### Why v9 is the baseline, not the endpoint

The promoted v9 candidate ranker has approximately:

```text
median error              61.01 km
mean error               194.36 km
p90 error                435.09 km
within 25 km              34.39%
candidate-oracle mean      35.99 km
```

The large gap between candidate-oracle error and final prediction error shows two distinct bottlenecks:

1. **Ranking/representation bottleneck:** the model often has a useful candidate available but assigns probability poorly.
2. **Candidate-coverage bottleneck:** a historical-location-only candidate bank cannot represent every new geographic expansion.

The supercharge program therefore improves both the representation/ranker and the spatial support of the forecast.

### Research findings that directly change the architecture

Current high-quality conflict forecasting systems suggest the following design choices:

- **ACLED CAST:** strong tree-based forecasting with conflict lags/trends, neighboring-area spillover, population/development covariates, and rolling-origin validation. Treat this as evidence that static geography + explicit lag features + rigorous temporal validation matter as much as model size.
- **VIEWS:** ensembles of specialized random-forest/gradient-boosting/Markov/hurdle models are more robust than a single learner; high-resolution systems also use spatial convolutions/recurrent or graph models.
- **VIEWS prediction challenge:** uncertainty-aware, zero-inflated/hurdle and global/local ensemble approaches are competitive; news-derived semantic topics can add predictive information.

Implementation consequence: build a **heterogeneous ensemble** whose neural geo model, tabular/tree model, v9 expert, and persistence/hotspot experts make different errors. Do not use model size as a substitute for feature quality.

### GeoBrain data contract

All geographical context is compiled into a reusable, cutoff-safe feature store before training. Raw rasters/vectors are never read inside the neural training loop. The canonical coarse-cell feature families are:

#### Dynamic conflict state

- event counts and fatalities over 1d / 3d / 7d / 30d / 90d / 365d
- time since last event
- active-day density
- historical recurrence
- event-type mixture
- local acceleration/deceleration
- transition counts between areas
- source diversity / confidence

#### Direction and momentum

- last-step east/north displacement
- elapsed time and speed
- bearing sin/cos
- EWMA velocity over multiple recent-event windows
- acceleration
- directional concentration / entropy
- hotspot centroid motion
- spread / contraction of the active area

**Immediate existing-data win:** `data/location/next_location_motion_v3.npz` already contains six motion channels that promoted v9 does not consume. The first challenger must use them.

#### Neighbor / spillover state

For each candidate/coarse cell:

- ring counts and fatalities at multiple radii
- neighboring-cell activity and recency
- distance-decayed event intensity
- adjacent-cell acceleration
- cross-border spillover where applicable
- source/report velocity nearby

#### Terrain and hydrology

Primary source: **Copernicus DEM GLO-30/GLO-90**, obtainable as open Cloud-Optimized GeoTIFFs from the public AWS registry. Aggregate to the forecasting cell rather than preserving 30 m tactical detail.

Features:

- elevation mean / std / p10 / p50 / p90
- slope mean / p90
- ruggedness / local relief
- elevation gradient
- fraction of steep terrain
- optional Height Above Nearest Drainage (HAND) and coarse drainage accessibility

#### Roads / accessibility / settlements

Primary source: **OpenStreetMap**, with versioned Ethiopia extracts from Geofabrik. A dated snapshot is mandatory for historical reproducibility.

Aggregate features:

- road length by broad class
- intersection density
- settlement/place counts
- building density when available
- distance/travel-access proxy to major settlements
- border proximity
- health/humanitarian POI density only as coarse accessibility context

Do not encode or expose inferred military routes or unit positions.

#### Population / built environment

Primary source: **WorldPop** Ethiopia gridded population; pin the exact release/year used in each experiment.

Features:

- total population
- log population
- population density
- urban concentration
- population-weighted settlement accessibility

#### Land cover

Primary source: **ESA WorldCover** 10 m global product, aggregated to coarse cells.

Features:

- built-up share
- cropland share
- forest/shrub/grass shares
- water/wetland share
- bare/sparse share
- land-cover entropy

#### Optional dynamic Earth-observation context

Only add a source after a causal ablation demonstrates value. Candidates:

- CHIRPS precipitation / rainfall anomaly
- VIIRS monthly night-light radiance and change, with observation-coverage mask
- coarse vegetation/drought indicators

Never use imagery acquired after an example's forecast cutoff.

#### Humanitarian / semantic context

- ReliefWeb localized report intensity and semantic embeddings
- displacement/access/infrastructure/civilian-impact signals
- eventually structured public/Telegram event semantics with publication-time provenance
- source reliability and contradiction features

### GeoBrain storage layout

```text
data/geo/raw/<source>/<snapshot>/...
data/geo/derived/<schema>/cells.parquet|npz
data/geo/manifests/<snapshot>.json
data/location/<dataset-version>.npz
models/location/<model-family>/<version>/
reports/location/experiments/<experiment-id>/
```

Every geo manifest records source URL/provider, license/citation, acquisition timestamp, release date, checksum, spatial resolution, temporal coverage, and transformation code version.

### Model architecture: GeoFusion challenger

The first large neural challenger should not simply enlarge v9. Use a factorized architecture:

```text
recent event sequence (motion-aware) ──► temporal Transformer + local temporal conv
                                               │
static/dynamic candidate-cell features ─► candidate encoder
                                               │
country/conflict context embeddings ───────────┤
                                               ▼
                              candidate → history cross-attention
                                               │
                              candidate self-attention / competition
                                               │
                      ┌────────────────────────┴──────────────────────┐
                      ▼                                               ▼
             broad-area probability head                    uncertainty/radius head
```

Required properties:

- candidate queries attend directly to the entire event sequence instead of only one pooled vector;
- candidate self-attention lets plausible areas compete in context;
- magnitude-preserving featurewise normalization for sparse count channels;
- motion-v3 history is consumed from the first experiment onward;
- static geography is encoded separately from rapidly changing conflict signals;
- coarse-cell probability is the primary product; continuous point estimates remain diagnostic only;
- model size is increased only after representation and leakage tests pass.

### Spatial support / candidate expansion

v9 only ranks historically observed locations. Add an expanded broad-area candidate bank consisting of:

- historically active cells,
- neighboring coarse cells around recent activity,
- directionally projected coarse cells derived from momentum,
- persistent historical hotspots,
- admin-region centroids / populated coarse cells where appropriate.

All candidate generation must use only pre-cutoff information. Store a `candidate_source` bitmask so ablations can identify whether a gain came from history, neighbors, momentum projection, or static hotspot coverage.

### Losses

Do not supervise only the single nearest candidate. Train a spatial distribution:

```text
L = hard_target_CE
  + λ_soft * distance_soft_label_CE
  + λ_dist * expected_geodesic_distance
  + λ_area * multiresolution_area_CE
  + λ_cal * probability_calibration_term
```

Distance-soft labels assign partial probability to nearby coarse zones instead of declaring a 1 km boundary crossing totally wrong. Multi-resolution heads predict the same target at coarse and medium scales, forcing the representation to learn the general war-zone geometry before precise ranking.

### Ensemble

Keep all of these as independently measurable experts:

1. frozen v9 ranker;
2. GeoFusion neural model;
3. gradient-boosted tabular geo expert;
4. historical hotspot/persistence expert;
5. optional temporal-risk v6 coarse escalation context.

Fit simple ensemble weights on rolling validation forecasts. Prefer simple averaging/regularized weights over fragile high-dimensional stacking unless rolling-origin evidence clearly favors it.

### Evaluation protocol — mandatory before promotion

A single 70/15/15 split is no longer enough for architecture search. Add expanding-window rolling origins modeled after current operational forecasting practice. At minimum report:

```text
mean / median / p90 geodesic error (diagnostic)
within 50 / 100 / 200 km
coarse-cell top-1 hit
coarse-cell recall@3 / recall@5
ADMIN1 hit rate where labels permit
negative log likelihood / Brier-style area score
expected calibration error
50 / 80 / 90% region empirical coverage
performance by year, geography, conflict, and event-gap bucket
```

The 2023–2025 block used during repeated historical research is a **development benchmark**, not a pristine future test. Final promotion claims require a genuinely later prospective window or a newly frozen untouched period.

### 2026-08-23 Phase 2 findings — static humanitarian context and robustness gates

The second supercharge pass added a coarse PRIO-grid humanitarian context path rather than feeding raw fine-resolution rasters directly into training. The source snapshots currently pinned are:

```text
Copernicus DEM GLO-90      173/173 required Ethiopia tiles cached locally
OpenStreetMap/Geofabrik   Ethiopia monthly snapshot 2026-08-01
WorldPop R2025A            Ethiopia 2026 constrained 1 km product
```

The OSM compiler aggregated 442,899 road geometries and 10,507 settlements into 208 PRIO cells. WorldPop was aggregated to the same 208 cells. The intentionally incomplete 100 m WorldPop transfer is preserved as a raw failed-transfer artifact; the production experiment uses the much smaller 1 km product because the forecasting scale is tens of kilometres.

#### Rolling-origin acceptance gate

A feature family is no longer accepted because it looks good on the historical 70/15/15 split. `evaluation.risk.rolling` runs expanding chronological origins and separately scores PRIO cells absent from the fold's training history. This gate caught two false-looking wins:

```text
Accessibility alone
  matched neural development AP              0.16760
  rolling mean AP delta                     +0.00165
  rolling cold-start AP delta               -0.06081
  decision                                  REJECT standalone

Population alone
  matched neural development AP              0.15436
  matched neural development intensity MAE   0.13398
  rolling mean AP delta                     -0.00176
  rolling cold-start AP delta               -0.09768
  decision                                  REJECT standalone

Population + accessibility interaction
  rolling mean AP delta                     +0.00191
  rolling mean intensity-MAE delta          -0.00048
  rolling cold-start AP delta               +0.08366
  rolling cold-start intensity-MAE delta    -0.00319
  decision                                  ACCEPT representation for further research
```

This is the intended promotion behavior: a spectacular later-block number does not override temporal instability or cold-start regression.

#### Proper-score ensemble result

A validation-only ensemble using binary log loss for escalation and MAE for intensity refused to chase the static-context AP gains:

```text
escalation probability weights
  v6              1.00
  accessibility   0.00
  population      0.00

intensity weights
  v6              0.50
  population      0.50

validation intensity MAE     0.09127   (v6: 0.09264)
development intensity MAE    0.13498   (v6: 0.13684)
```

Therefore v6 remains the calibrated escalation reference. Static context is currently useful as a magnitude/cold-start support signal, not as a universally trusted escalation prior.

#### Explicit transfer objective

Two transfer formulations were tested and preserved:

1. A shared neural masked-local objective randomly removed the target cell's eight local channels during training. It damaged the primary representation (best ordinary validation AP about 0.106) and is preserved under `v7_transfer_accessibility_failed/` as a negative result. The trainer was patched to persist selected checkpoints before any later packaging step so future completed compute cannot be lost to bookkeeping errors.
2. A separate non-local LightGBM specialist excludes all local target-cell history and selects hyperparameters on chronologically later PRIO entities deliberately held out from fit. It reached transfer-selection AP 0.09239 on 1,441 unseen-area rows / 101 positives, proving the objective is learnable, but ordinary validation/development AP (~0.048/~0.066) is too weak for promotion.
3. A second masked-local neural run on the stable 26-channel spatial tensor (without static context) confirmed the same failure mode: validation AP 0.12853, development AP 0.14737, validation intensity MAE 0.12542, and synthetic masked-local AP only 0.05280/0.07025. It is preserved under `models/risk/ethiopia/v7_transfer/` with status `research-rejected-transfer`.
4. Gating the non-local tree on generic low-local-activity rows does not help: validation selects only 5% specialist probability weight, and that weight slightly worsens the later low-activity AP/log loss. A held-out-entity-only 5% blend nudges development held-out AP 0.14303 -> 0.14527, but this is too small/brittle for production.

The lesson is to keep transfer as a specialist/gating research problem instead of forcing the main temporal encoder to sacrifice ordinary accuracy. The strongest current cold-start signal is the **combined population + accessibility representation** in rolling evaluation, not the standalone transfer specialist.

#### Static/dynamic neural fusion

`ContextFusionRiskModel` now encodes the 26 dynamic channels temporally and static population/accessibility once through a bounded FiLM-style branch. The first seed was not competitive and validation-only proper-score ensembling assigned it zero weight. Preserve it as an architecture ablation rather than promoting it.

#### Proper-score broad-area geographical ensemble

The frozen location experts are now combined using a fixed **100 km Gaussian distance-soft target** rather than an exact-candidate label. To avoid validation leakage, v9's 70%-trained phase-1 recipe was reproduced specifically for validation export instead of using its 85%-trained production checkpoint. Torch and LightGBM exports run in separate processes to avoid the macOS OpenMP runtime collision discovered during research.

Validation-only temperature calibration and convex weight selection produced:

```text
expert validation distance-soft CE
  v9 phase-1 reproduction    3.07405
  actor transfer             3.06941
  propagation kernel         3.06853

global proper-score ensemble
  v9                         42.5%
  actor transfer             15.0%
  propagation kernel         42.5%
  validation CE              3.05931
  development CE             3.11181
  development median         60.56 km       diagnostic
  development within 100 km  59.31%
  development top3 <=100 km  68.95%
```

The later-block CE improves over the best individual expert (kernel 3.12161) without using that block for temperature or weight selection. For Ethiopia, the globally optimal mixture does not transfer: a validation proper-score selector with a 0.5% global-nonregression guard chooses **100% propagation kernel**. This local override is intentional and prevents a global ensemble from degrading the target geography.

Artifact: `models/location/broad_area_ensemble/v1/`. Registered system: `location.broad_area.ensemble`.

#### 2026-08-24 v9-preserving Ethiopia v10 experiment

The Ethiopia program returned to the validated v9 architecture rather than replacing it wholesale. A conservative v10 challenger now starts from the exact 70%-trained v9 phase-1 checkpoint, freezes the temporal Transformer representation, and fine-tunes only the late context/candidate scoring modules and Ethiopia-relevant embeddings on Ethiopia rows from the pre-validation training era. Training uses a 100 km Gaussian distance-soft target, a small hard-candidate term, and KL distillation back to frozen v9 so specialization cannot discard the parent prior.

Artifact: `models/location/candidate_ranker_ethiopia/v10_v9_adapter_frozen/`. Trainer: `src/humanitarian_forecast/location/training/train_v9_ethiopia_adapter_v10.py`. The earlier stochastic-frozen-encoder attempt remains preserved at `models/location/candidate_ranker_ethiopia/v10_v9_adapter/` and is not overwritten.

Against an Ethiopia-only temperature-calibrated reproduction of v9 on the aligned chronology:

```text
                                  v9 parent       v10 adapter
validation distance-soft CE       3.11299          3.08088
validation mean top-1 error km     194.95           190.52
validation broad-area score          .38611           .39222

development distance-soft CE      3.27969          3.26152
development mean top-1 error km    208.72           204.48
development within 100 km            25.21%           27.27%
development broad-area score          .27030           .27844
```

The v10 adapter therefore improves its v9 parent on both chronological validation and the later development block. It is **not promoted yet**: the development block has been repeatedly inspected, and Ethiopia rolling-origin / prospective confirmation remains mandatory.

Recalibrating the propagation kernel specifically for Ethiopia revealed a stronger proper-score baseline than previously recorded (validation CE about 3.06470; development CE about 3.23668). A validation-only three-expert mixture of the strongest Ethiopia candidate-family experts selected 70.0% multiradius, 22.5% v10 adapter, and 7.5% propagation kernel. It reached validation CE 3.04434 and development mean/median error about 198.99 / 171.27 km, but development CE 3.23956 remains fractionally worse than the recalibrated kernel. Therefore the current strict probabilistic champion remains the Ethiopia-calibrated propagation kernel, while v10 is retained as a useful independent expert and the correct v9-preserving research direction.

Process lesson: after v9, too much research budget was spent on replacement representations before exhausting conservative v9-preserving Ethiopia adaptation. Continue architecture exploration only when it addresses a measured residual error or produces independent ensemble value.

##### Direct-v9 mainline supersession — 2026-08-24

The primary Ethiopia location program is now explicitly **v9 -> v10 -> v11**, not replacement-model exploration. Receipt: `reports/location/v10_direct_v9lineage_receipts.json`.

- Production weights: `models/location/candidate_ranker/v10/`, trained with the v9 architecture/objective on all **157,932** available labels, including all **2,669** Ethiopia labels through **2025-12-29**.
- Ethiopia production wrapper: `models/location/candidate_ranker_ethiopia/v10/`; tensor-state SHA-256 is identical to the global v10 checkpoint.
- Accepted Ethiopia calibration: weighted geometric median, temperature `0.8`, 50% model center + 50% causal `recent_8_mean`, selected by minimum validation mean subject to no validation regression at <=50 km and <=100 km. Frozen later-block mean/median/P90: **178.33 / 153.62 / 343.25 km**, versus v9 **181.85 / 158.58 / 357.31 km**; <=100 km improves **30.68% -> 31.71%**.
- Rejected: simply swapping in the 25/60-dimensional actor-transfer tensor; Ethiopia geomedian mean regressed to about **182.90 km**.
- Rejected as negligible: direct v9 Ethiopia weight fine-tune; later-block mean gain only about **0.10 km**.
- Production inference now reads country/conflict identity maps from all-data checkpoints instead of reconstructing the obsolete 85% maps.

##### v10-prized support expansion and final all-data package — 2026-08-24

The direct v9 line has now produced `models/location/candidate_ranker/v10_prized/`. The architecture remains `ConflictCandidateRanker`; the gain comes from preserving the sharp 32-candidate v9 view while adding a 64-candidate support view whose Ethiopia development oracle mean is **23.90 km** instead of **34.85 km**. A one-epoch late-layer Ethiopia adaptation is trained on all **2,669** Ethiopia labels from the existing all-data global v10 parent, while the global parent itself scores the expanded 64-candidate bank.

Historical aligned Ethiopia development receipt (`reports/location/v10_prized_receipts.json`): v9 parent mean/median/P90 **178.59 / 156.83 / 340.68 km** versus v10-prized **177.20 / 155.97 / 334.17 km**. <=100 km moves **30.89% -> 30.99%**. <=50 km and <=25 km regress (**10.43% -> 9.61%**, **3.51% -> 2.27%**), so the next objective must explicitly recover local-hit precision without giving back the mean/tail gain. The historical final 15% is still development data, not prospective evidence.

The final package uses a frozen 50/50 center blend: adapted 32-candidate geometric median (`T=1.0`) plus global 64-candidate weighted mean (`T=0.65`). Public inference snaps the internal center to a coarse >=0.2-degree humanitarian grid and never returns a ranked tactical candidate list.

**Budget rule:** alternate H3/GeoFusion/Hawkes/etc. branches remain archived research evidence. Do not allocate primary experimentation to them unless they are being used as a feature/teacher for the v9 lineage or can beat the current Ethiopia v9-lineage model under the same contract. The immediate optimization target is the measured candidate-support/ranking gap.

#### Rolling-origin Ethiopia location gate

A fixed 75-tree LambdaRank probe now evaluates aligned geographic feature families at three expanding origins. This prevents a later-regime gain from silently becoming a universal claim.

```text
Actor-transfer vs motion baseline
  mean rolling broad-score delta    +0.00274
  fold deltas                        0.00000, +0.00987, -0.00166
  mean <=100 km delta               +0.00405
  stable_accept                      false
  decision                           KEEP AS ENSEMBLE/DIVERSITY SIGNAL, NOT STANDALONE CHAMPION

Terrain vs actor-transfer
  mean rolling broad-score delta    -0.00160
  fold deltas                       +0.00208, -0.00011, -0.00678
  stable_accept                      false
  decision                           REJECT STANDALONE
```

The actor-transfer feature family remains useful because it improves the unseen Government–Fano subgroup and contributes non-zero weight to the proper-score global ensemble, but it does not satisfy the stricter standalone rolling-origin promotion rule. Terrain remains a preserved negative ablation.

#### Evaluation status

The historical final 15% has been repeatedly observed. It is a **development benchmark**, never an untouched-test claim. New promotion decisions require rolling-origin stability and then prospective frozen forecasts.

### Immutable Phase 2 restore point

After the population/OSM materialization, rolling-origin gates, proper-score broad-area ensemble, actor/terrain rolling evaluation, and transfer-objective ablations are recorded, create and preserve the annotated git tag:

```text
geo-supercharge-phase2-2026-08-24
```

This tag must remain immutable. Do not rewrite it or delete model artifacts produced before it. Raw/processed geospatial caches remain outside git under `/data/`, but their source/version manifests and the model/report artifacts needed to reproduce decisions are checkpointed.

### Immediate execution queue

- [x] Create immutable git restore tag `geo-supercharge-preflight-2026-08-23` at `1adf232`.
- [x] Audit promoted v9 and identify the candidate-oracle/ranking gap.
- [x] Confirm motion-v3 history exists but is not consumed by v9.
- [x] Register GeoBrain static-context compiler and manifest schema.
- [x] Add Copernicus DEM adapter and coarse terrain aggregation.
- [x] Add OSM/Geofabrik adapter for road/settlement aggregates.
- [x] Add WorldPop population adapter.
- [ ] Add WorldCover land-cover adapter.
- [x] Build motion-aware candidate dataset with the exact v9 examples/labels for apples-to-apples evaluation.
- [x] Expand the direct v9 support from 32 to 64 cutoff-safe historical candidates; Ethiopia development oracle mean falls 34.85 -> 23.90 km. Momentum/local-child support remains a later refinement.
- [x] Implement `GeoFusionCandidateRanker` (v1/v2 research challengers; neither promoted).
- [x] Test distance-soft and multiresolution supervision inside the v9 family; soft100 improved validation strongly but only weakly generalized, so it is preserved as a rejected primary recipe rather than promoted.
- [x] Add rolling-origin evaluator and experiment ledger.
- [x] Train first GeoFusion challenger on existing data before external-geo augmentation.
- [~] Add terrain/access/population/land-cover features one family at a time and run ablations. Terrain/access/population done; WorldCover remains.
- [x] Train tree diversity expert on the compiled GeoBrain table.
- [x] Build a validation-only proper-score broad-area ensemble after freezing individual expert outputs; keep rolling-origin/prospective confirmation as the promotion gate.
- [x] Preserve every trained version and produce `info.blt`/failure metadata with lineage, metrics, and rejection/promotion reason.

---

Build **The Humanitarian Forecast** into an autonomous, self-evaluating humanitarian conflict-risk forecasting system that continuously ingests public information, converts it into structured spatiotemporal state, produces prospective forecasts for Ethiopia, scores those forecasts against later verified outcomes, trains challengers on recent data, and promotes only models that demonstrate measurable prospective improvement.

Primary internal modeling target:

- **Top-1 mean geodesic error ≤ 25 km** on prospective chronological evaluation.
- Median error target ≤ 15 km.
- Improve 25 km / 50 km hit rates while keeping uncertainty calibrated.
- Never optimize against retrospectively revised predictions.

Public-facing output must remain a **coarse humanitarian risk map with uncertainty**, not a precise tactical targeting interface. The internal model may use fine spatial cells to improve accuracy, but the public map should aggregate risk into broader humanitarian zones/admin areas and should not expose inferred unit positions, tactical routes, weapons, or exact live strike coordinates.

---

# Core principles

- [ ] Every forecast must be **prospective and immutable after its cutoff**.
- [ ] Every raw source item must retain publication time, source, raw text, and provenance.
- [ ] AI-generated labels are **provisional supervision**, not ground truth.
- [ ] Later high-confidence sources can confirm, correct, or supersede provisional labels.
- [ ] Absence of ReliefWeb coverage must never be interpreted as absence of conflict.
- [ ] New reports update a persistent spatial world state rather than existing only as independent text rows.
- [ ] Weekly adaptation must use supervised/probabilistic training with replay rather than blindly applying generic RL.
- [ ] The live production model is never overwritten merely because retraining ran.
- [ ] Every retrained model is a **challenger** and must beat the champion through a fixed promotion gate.
- [ ] If no challenger improves the required metrics, keep the existing champion.
- [ ] Keep the validated v9 model untouched as a baseline/reference expert.
- [ ] Preserve per-example causal cutoffs everywhere in historical dataset generation.

---

# Target system architecture

```text
Telegram channels ──────┐
ReliefWeb ───────────────┼──► RAW EVENT BUS
Public corroboration ────┘
                              │
                              ▼
                    AI STRUCTURING LAYER
                    translation
                    semantic extraction
                    place resolution
                    deduplication
                    corroboration
                    confidence estimation
                              │
                              ▼
                    CANONICAL EVENT STORE
                              │
                              ▼
                    H3 FEATURE / WORLD STATE
                    1h / 6h / 24h / 3d / 7d / 30d / 90d
                              │
                 ┌────────────┴────────────┐
                 ▼                         ▼
           LIVE NOWCAST              DAILY SNAPSHOT
                                           │
                                           ▼
                                PROSPECTIVE FORECAST
                                           │
                                           ▼
                                    FORECAST LEDGER
                                           │
                                           ▼
                                  VERIFIED OUTCOMES
                                           │
                                           ▼
                                  WEEKLY CHALLENGERS
                                           │
                                           ▼
                                   PROMOTION GATE
                                           │
                                           ▼
                                      CHAMPION
```

---

# Phase 0 — Preserve current baseline and define the benchmark

Current validated reference:

- `models/location/candidate_ranker/v9/`
- Median historical test error ≈ 61.01 km.
- Mean historical test error ≈ 194.36 km.
- Current candidate-set oracle mean ≈ 36 km historically.

The candidate oracle result means that simply ranking the existing 32 historical candidates better cannot reliably reach a 25 km mean target. The spatial representation itself must expand beyond a fixed set of previous conflict coordinates.

Tasks:

- [ ] Preserve `models/location/candidate_ranker/v9/` exactly.
- [ ] Treat v9 as an ensemble expert and fallback benchmark.
- [ ] Create a benchmark specification that contains:
  - [ ] mean geodesic error
  - [ ] median geodesic error
  - [ ] p90 error
  - [ ] within-25-km rate
  - [ ] within-50-km rate
  - [ ] probability calibration metrics
  - [ ] uncertainty-region empirical coverage
  - [ ] chronological/prospective split definition
- [ ] Establish a simple persistence baseline.
- [ ] Establish historical-hotspot baseline.
- [ ] Establish v9 baseline.
- [ ] Require every future model report to compare against all three.

---

# Phase 1 — Delete/retire the old web stack and define the new application boundary

The old web implementation should not be extended. Replace it with a new active service designed specifically for Cloud Run.

Target services:

```text
forecast-web
telegram-ingest
normalize-events
corroboration-worker
feature-rollup
forecast-job
weekly-trainer
promotion-evaluator
```

Tasks:

- [ ] Remove the old active web implementation once the replacement service has a working vertical slice.
- [ ] Keep any useful static assets/templates only if explicitly migrated.
- [ ] Make the new web service stateless.
- [ ] Store durable state outside the container.
- [ ] Keep model artifacts/version metadata in structured model storage.
- [ ] Add health endpoints and model/version metadata endpoints.

Recommended cloud responsibilities:

- Cloud Run Service: frontend/API.
- Cloud Run Service: Telegram webhook if webhook mode is available.
- Pub/Sub: ingestion/event transport.
- Cloud Run Jobs: feature rollups, official daily forecast, weekly training, promotion evaluation.
- Cloud Scheduler: cron triggers.
- Cloud Storage: raw source archives, model artifacts, frozen snapshots.
- Postgres/PostGIS or BigQuery: normalized events and spatial state.
- BigQuery: forecast ledger/evaluation history if convenient.
- Secret Manager: Telegram/Groq/other API keys.

---

# Phase 2 — Canonical raw event ingestion

## 2.1 Telegram

Primary fast sensor: five selected Telegram channels about the war.

For every incoming post, persist an immutable raw record before doing any AI work.

Canonical raw fields:

```text
raw_event_id
source_type
source_name
source_message_id
publication_timestamp_utc
publication_timestamp_eat
raw_text
raw_media_metadata
raw_url_or_reference
language_if_known
ingested_at
content_hash
```

Tasks:

- [ ] Build one ingestion adapter per Telegram mode required.
- [ ] Prefer webhook delivery when technically possible.
- [ ] If authenticated channel polling/MTProto is required, isolate it behind the same canonical ingestion interface.
- [ ] Deduplicate exact reposts using content hashes.
- [ ] Keep repost relationships rather than deleting source provenance.
- [ ] Never overwrite raw posts after ingestion.
- [ ] Track source uptime and gaps.

## 2.2 ReliefWeb

ReliefWeb is a context and corroboration sensor, not a complete battle ledger.

Tasks:

- [ ] Continue historical database ingestion for training.
- [ ] Add incremental retrieval for newly published reports.
- [ ] Store publication time separately from the event time described in the report.
- [ ] Keep full text/body when allowed by the source pipeline.
- [ ] Extract humanitarian context even when the report is not a direct conflict-event report.
- [ ] Do not treat missing ReliefWeb reports as negative labels.

Useful ReliefWeb signal families:

- displacement
- humanitarian access
- civilian impact
- infrastructure disruption
- food/water/medical disruption
- regional deterioration
- clashes/attacks if explicitly reported
- delayed corroboration

## 2.3 Public corroboration sources

Web/public-search corroboration is a first-class sensor and **must not be removed from the architecture**. It should be selective rather than performed at maximum depth for every trivial event, because the autonomous call-allocation policy should learn where extra searches create forecast value.

The web layer should support both fixed adapters and open-ended public search. Maintain connectors/adapters for high-value structured sources where available, then allow the research/corroboration agent to search beyond those sources when novelty, contradiction, or uncertainty warrants it. Every retrieved item must carry retrieval time, publication time when available, URL/source identity, content hash, and the forecast cutoff under which it was legally observable.

Candidate web/public families to evaluate include:

```text
ReliefWeb and OCHA products
UCDP / ACLED-style public conflict datasets where licensing permits
UNHCR / IOM displacement reporting
WFP / FEWS NET food-security context
HDX humanitarian datasets
public government / regional statements
reputable local and international news
public humanitarian situation reports
weather / rainfall / drought products
road/access/infrastructure disruption notices
public geospatial/environmental datasets
search-engine discovery for novel local reports
```

Web/public-search corroboration should be **selective**, not performed for every trivial event.

Trigger deeper corroboration when one or more applies:

```text
high novelty
low source confidence
new geography
large predicted forecast impact
contradictory source claims
high humanitarian severity
```

- [ ] Store each corroboration result as another source object.
- [ ] Store publication time and retrieval time.
- [ ] Ensure features for a forecast include only information publicly available before that forecast cutoff.

---

# Phase 3 — AI supervision / structured event extraction

Groq-powered agents should transform raw unstructured reports into structured probabilistic event objects.

The AI is a **parser, reasoner, reconciler, and provisional labeler**. It is not automatically trusted as ground truth.

## 3.1 Structured output schema

Every parsed report should produce at least:

```text
event_candidate_id
raw_event_id
claimed_event_time_distribution
publication_time
language
translation_confidence

event_type_distribution
humanitarian_impact_distribution

location_candidates[]
location_probability[]
admin_region_candidates[]
location_confidence

civilian_harm_signal
displacement_signal
infrastructure_disruption_signal
humanitarian_access_signal

source_claim_strength
hedging_probability
rumor_probability
novelty_score

semantic_embedding
agent_version
prompt_version
created_at
```

## 3.2 Multi-agent iterative refinement

Do not blindly spend 50–60 calls on every post.

Use an adaptive call budget.

Simple high-confidence reports:

- ~3–8 calls.

Ambiguous/high-impact reports:

- potentially dozens of calls.

Possible agent roles:

1. Translation agent.
2. Event-type extraction agent.
3. Temporal interpretation agent.
4. Place-name candidate agent.
5. Geocoding/map resolution agent.
6. Deduplication agent.
7. Cross-source matching agent.
8. Contradiction detector.
9. Confidence calibration agent.
10. Final consensus/reconciliation agent.

## 3.3 Geographic resolution

The system may use map/geocoding tools to resolve ambiguous place names into spatial candidates.

Important design:

- [ ] Return a probability distribution over plausible cells/areas when the location is ambiguous.
- [ ] Do not force an exact coordinate when the source does not justify one.
- [ ] Preserve all candidate locations considered and the reasoning metadata required for later auditing.
- [ ] Convert resolved positions into the canonical H3 spatial representation.
- [ ] Keep human-facing output coarser than internal state.

## 3.4 Iterative event consensus

When new reports appear, determine whether they:

- describe the same event,
- corroborate an existing event,
- contradict an existing event,
- describe a new nearby event,
- provide a more precise time/location for a known event.

Maintain a canonical event object with version history rather than making each post a separate battle label.

Example:

```text
Telegram report A
Telegram report B
ReliefWeb report C
public report D
        │
        ▼
canonical_event_1042
        │
        ├── time distribution
        ├── H3 location distribution
        ├── event-type distribution
        ├── source set
        ├── confidence
        └── provenance/version history
```

---

# Phase 4 — Learn source reliability

Each source should develop its own reliability profile from later-confirmed outcomes.

Estimate quantities such as:

```text
P(report eventually confirmed | source, event_type, geography)
location error distribution by source
time error distribution by source
copy/repost rate
early-reporting advantage
correction frequency
rumor/false-positive frequency
```

Tasks:

- [ ] Track source-specific historical verification rate.
- [ ] Detect likely repost chains so copied reports are not treated as five independent confirmations.
- [ ] Weight consensus by source independence as well as source count.
- [ ] Feed source reliability into the event confidence model.
- [ ] Recalculate reliability as slow verified labels arrive.

---

# Phase 5 — Fast labels and slow labels

Create two supervision tiers.

## Fast provisional labels

Generated from:

- Telegram
- ReliefWeb
- public corroboration
- cross-source AI consensus
- geocoding/map resolution

These arrive quickly and power near-real-time adaptation.

## Slow high-confidence labels

Generated later from:

- UCDP
- trusted retrospective sources
- verified humanitarian reporting
- other curated historical datasets

Slow labels can correct the provisional event object.

Each training label should include a confidence/weight.

Example initial policy to validate empirically:

```text
verified retrospective label       1.00
strong independent multi-source     0.90–0.98
moderate corroborated AI label      0.70–0.90
single historically reliable source 0.40–0.70
ambiguous uncorroborated source      0.05–0.30
```

Tasks:

- [ ] Never assign all AI labels weight 1.0 by default.
- [ ] Learn calibration curves for AI-label confidence.
- [ ] When a provisional label is later corrected, record the correction instead of silently replacing history.
- [ ] Use correction history to evaluate the labeling agents themselves.

---

# Phase 6 — Replace crude event features with a persistent spatial world state

Current promoted v9 history has only 19 coarse event features. This is inadequate for the 25 km goal.

Move toward:

```text
X[timestamp, H3_cell, timescale, feature]
```

## 6.1 Spatial grid

Recommended starting representation:

- H3 resolution 5 for internal primary cells.
- Coarser res 3/4 aggregate context.
- Optionally allow continuous within-cell refinement.

Do not require the predicted event to lie on a previously observed conflict coordinate.

## 6.2 Time windows

Maintain rolling features over:

```text
1 hour
6 hours
24 hours
3 days
7 days
30 days
90 days
```

## 6.3 Feature families

### Conflict activity

- [ ] event counts
- [ ] event severity
- [ ] days/hours since last event
- [ ] active-day density
- [ ] acceleration/deceleration
- [ ] local event-type distribution
- [ ] local fatality/civilian-harm intensity

### Spatial propagation

- [ ] neighboring-cell activity
- [ ] distance-weighted nearby activity
- [ ] hotspot motion
- [ ] spatial spread/contraction
- [ ] directional propagation features
- [ ] local cluster density

### Source / information state

- [ ] number of independent sources
- [ ] channel diversity
- [ ] source reliability weighted consensus
- [ ] contradiction score
- [ ] report novelty
- [ ] report velocity
- [ ] missing-source flags
- [ ] ingestion outage flags

### AI semantics

- [ ] event-type probabilities
- [ ] displacement probabilities
- [ ] civilian-harm probabilities
- [ ] humanitarian-access probabilities
- [ ] infrastructure-disruption probabilities
- [ ] semantic embedding
- [ ] location confidence
- [ ] temporal confidence
- [ ] translation confidence
- [ ] hedging/rumor score

### ReliefWeb

Replace crude country/day keyword counts with spatially localized and semantic signals where possible.

- [ ] local report intensity
- [ ] displacement semantic intensity
- [ ] humanitarian-access intensity
- [ ] civilian-impact intensity
- [ ] infrastructure intensity
- [ ] report embeddings or compressed semantic vectors

### Historical conflict context

- [ ] conflict embedding
- [ ] country embedding
- [ ] long-term local intensity
- [ ] historical recurrence
- [ ] historical transitions
- [ ] seasonal features

### Geography / static context

Use only coarse civilian/humanitarian geographic features.

- [ ] elevation/ruggedness
- [ ] settlement/population density
- [ ] administrative boundaries
- [ ] coarse transport connectivity
- [ ] border proximity
- [ ] seasonal/weather context if validated as useful

Avoid constructing or displaying tactical military route/position features.

---

# Phase 7 — Model architecture research program

Do not jump immediately to one new production model. Run an ablation-driven research program.

## 7.1 Baselines to preserve

- v9 candidate ranker
- persistence
- historical hotspot
- simple H3 frequency model
- distance-decay model

## 7.2 Primary architecture candidate

Test a multiscale model composed of:

```text
H3 cell embeddings
+ static geographic features
+ dynamic world-state features
        │
        ▼
spatial graph/message passing or spatial attention
        │
        ▼
temporal Transformer
        │
        ├── cell probability head
        ├── within-cell offset head
        ├── event-time/horizon head
        └── uncertainty head
```

Potential ensemble experts:

- new H3 spatiotemporal model
- v9 historical-candidate model
- self-exciting/point-process model
- persistence expert
- historical-hotspot expert

Train an ensemble/gating model to learn when each expert is useful.

## 7.3 Output representation

Primary output should be a spatial probability distribution rather than only a point estimate.

Store:

```text
probability per H3 cell
optional continuous point estimate
calibrated uncertainty region
model disagreement
abstention/confidence signal
```

---

# Phase 8 — Training losses

Use supervised/probabilistic losses rather than generic RL as the main optimizer.

Candidate combined objective:

```text
L =
    λ1 * spatial cell cross entropy
  + λ2 * expected geodesic distance
  + λ3 * continuous offset regression
  + λ4 * calibration/probability loss
  + λ5 * uncertainty loss
```

Training examples should be weighted by label confidence.

Research tasks:

- [ ] Sweep objective weights.
- [ ] Test focal loss for sparse spatial cells.
- [ ] Test probability-distribution targets when the event label itself is spatially uncertain.
- [ ] Compare expected-distance optimization versus direct coordinate loss.
- [ ] Test mixture density/point-process heads.
- [ ] Measure whether semantic embeddings add prospective value.
- [ ] Measure whether Telegram-derived features improve beyond UCDP/ReliefWeb baseline.
- [ ] Perform source-family ablations.

---

# Phase 9 — Forecast timing and immutable snapshots

Maintain two distinct products.

## Live nowcast

Can update continuously as new data arrives.

Purpose:

- current humanitarian-risk state
- current model belief
- not used as a mutable substitute for historical official forecasts

## Official prospective forecast

Freeze a feature snapshot and produce a forecast at a fixed cutoff.

Recommended EAT schedule:

```text
continuous       ingest new reports
few minutes      AI normalize / deduplicate / corroborate
15 minutes       update rolling H3 feature state
hourly            produce live nowcast
23:55 EAT         freeze official daily feature snapshot
00:00 EAT         create next-day official forecast
T+1 ... T+7       accumulate provisional outcome evidence
later             attach high-confidence retrospective labels
```

Tasks:

- [ ] Official forecast row is immutable.
- [ ] Feature snapshot hash must be stored with each official forecast.
- [ ] Model version/hash must be stored.
- [ ] Data cutoff timestamp must be stored.
- [ ] Do not allow future reports to modify the stored official prediction.

---

# Phase 10 — Forecast ledger

Every official forecast must be logged before the outcome is known.

Canonical ledger fields:

```text
forecast_id
issued_at
cutoff_at
forecast_horizon
model_version
model_hash
feature_snapshot_id
feature_snapshot_hash
spatial_probability_distribution
point_estimate
uncertainty_region
abstention_state
source_coverage_summary
created_at
```

Later attach:

```text
provisional_outcome
verified_outcome
mean/point geodesic error
25km hit
50km hit
probability score
calibration score
label confidence
scored_at
```

This ledger becomes the single source of truth for whether the model is actually improving.

---

# Phase 11 — Iterative AI prediction/refinement before cutoff

The AI system can iteratively improve the world state and therefore improve the **live** prediction before the forecast cutoff.

Example:

```text
09:00 state → live prediction v1
10:23 new Telegram report
10:30 structured/corroborated event
10:31 world state update
10:32 live prediction v2
12:05 ReliefWeb corroboration
12:10 state update
12:11 live prediction v3
```

The final official forecast is frozen at cutoff.

Never permit:

```text
battle occurs
new report describes outcome
old official prediction is recomputed
recomputed value is scored
```

That would create retrospective leakage.

---

# Phase 12 — Weekly continual-learning loop

Use weekly supervised adaptation with replay.

Do not blindly train only on the last four weeks and evaluate on those same rows.

Recommended rolling 28-day structure:

```text
Days 1–21   recent adaptation pool
Days 22–28  untouched recent shadow evaluation
+ older Ethiopia replay
+ international replay
```

After the shadow week ages out, it becomes eligible for training.

Initial replay mix to test:

```text
40% recent Ethiopia
30% older Ethiopia
30% international history
```

Optimize this empirically.

Tasks every weekly cycle:

- [ ] Freeze current champion.
- [ ] Snapshot all newly eligible training labels.
- [ ] Build training pool with confidence weighting.
- [ ] Keep a recent untouched shadow window.
- [ ] Generate multiple challengers with different adaptation policies.
- [ ] Evaluate each challenger prospectively/chronologically.
- [ ] Run promotion gate.
- [ ] Promote only if gates pass.
- [ ] Otherwise retain champion.

Example challengers:

```text
A: recent-heavy adaptation
B: balanced replay
C: conservative low-LR fine-tune
D: same weights + uncertainty recalibration only
E: ensemble-weight/gating update only
```

## 12.1 Autonomous outcome-scored self-improvement controller

The weekly loop above is only the outer cadence. The intended research system is an **automatic reward-driven improvement engine** that continuously proposes, trains, scores, preserves, and selects new forecasting/labeling variants as verified future outcomes arrive. Do not reduce this to a hand-operated training script.

This is deliberately broader than ordinary gradient descent. Treat it as an AutoML / evolutionary / bandit / RL-style meta-optimization layer over the entire forecasting stack. The core predictor may still train with supervised/probabilistic losses because those are the statistically appropriate inner objectives; the **outer controller is outcome-reward driven**. Generic policy-gradient RL is not mandatory, but automatic reward-based search is a required capability.

Three nested learning loops must exist.

### Loop A — web/AI supervision improvement

Every provisional event produced by the web/Telegram/ReliefWeb + Groq pipeline is later compared against slower verified outcomes. Feed those corrections back into the labeling system.

The controller may automatically vary and evaluate:

```text
agent prompts / system instructions
agent role decomposition
translation strategy
geocoder candidate-generation strategy
number of reasoning/refinement passes
Groq model choice
temperature / sampling diversity
consensus/reconciliation policy
source-reliability weights
contradiction-handling rules
web-corroboration depth
web-search query templates
call budget per event
confidence-calibration mapping
```

Reward each configuration against later verified labels using proper metrics rather than human vibes:

```text
event-type log loss
spatial distance-soft cross-entropy
time-distribution log score
entity/actor extraction F1
deduplication precision/recall
confidence calibration error
correction rate
false-corroboration penalty
latency / API-cost penalty
```

Store every agent configuration as a versioned artifact with prompt hash, model name, sampling parameters, source set, cost, and retrospective score. A later correction must never erase the original provisional output.

### Loop B — forecasting-model self-improvement

When an official forecast matures and its outcome becomes observable, score the frozen forecast and add the result to the experiment/replay ledger. The controller then proposes new challengers automatically.

Search dimensions may include:

```text
feature families and interactions
history length / temporal windows
spatial neighborhood scales
actor-transfer features
propagation-kernel scales
H3/PRIO resolution
static-context encoders
semantic embeddings
model width/depth/heads
Transformer/CNN/tree/point-process experts
loss mixtures
distance-soft target radius
class weighting
replay composition
learning rate / weight decay
calibration temperatures
ensemble members / weights
gating policies
uncertainty heads
abstention thresholds
```

The outer reward must be multi-objective and calculated only from frozen chronological/prospective forecasts. Initial reward vector:

```text
+ proper broad-area probability score (primary)
+ distance-soft cross-entropy improvement
+ median/mean-error improvement
+ within-50/100/200 km coverage improvement
+ calibration improvement
+ unseen-conflict / cold-start improvement
+ rolling-origin stability
- p90/tail regression
- calibration regression
- geographic subgroup regression
- leakage/integrity failure (hard reject)
- excessive latency/compute when gains are negligible
```

Do **not** collapse all of these into a single opaque number at first. Maintain a Pareto frontier of challengers. Promotion can use hard non-regression constraints plus a primary proper-score objective.

### Loop C — adaptive source/search/call allocation

The information-acquisition layer is sequential and is a legitimate place for contextual-bandit/RL-style optimization. Given the current world state and an ambiguous report, the controller should learn whether another web search, another agent pass, another map/geocoder lookup, or no further call is worth the cost.

State may contain:

```text
current event uncertainty
source reliability
novelty
geographic ambiguity
forecast sensitivity to the event
contradiction count
time remaining before cutoff
current call/API spend
```

Actions may contain:

```text
stop and accept current event distribution
query another public source
run a targeted web search
ask another extraction/reconciliation agent
run map/geocoder resolution
request a higher-cost model
expand candidate places
```

Reward is the later improvement in verified event quality and downstream forecast proper score minus latency/API cost. This lets the system learn where 3 calls are enough and where 30–60 calls are justified.

### Automatic experiment lifecycle

Every cycle should behave approximately as:

```text
1. ingest new web / Telegram / ReliefWeb observations
2. update provisional canonical events and world state
3. freeze official forecast at cutoff
4. when outcome matures, attach verified/corrected label
5. compute forecast + labeling reward vectors
6. append experience to replay/experiment ledger
7. propose N challenger configurations automatically
8. train/evaluate challengers in isolated versioned directories
9. run rolling-origin + recent shadow + subgroup gates
10. update Pareto frontier / ensemble candidates
11. promote only challengers that pass all hard gates
12. retain champion if nothing wins
13. preserve every trained artifact, failed run, prompt version, and score
14. feed winning/losing experiment history back into the proposal policy
```

Candidate proposal algorithms to benchmark:

```text
random + successive halving baseline
Optuna/Bayesian optimization
ASHA/Hyperband
population-based training
evolutionary architecture/feature search
contextual bandits for source/call allocation
LLM research agent proposing experiments from the experiment ledger
learned surrogate model predicting which experiments are worth running
```

The experiment-proposal agent must never be allowed to rewrite historical scores or delete losing artifacts. It proposes new immutable experiments only. Compute already spent on a model is preserved even when the model loses.

### Reward integrity / anti-reward-hacking rules

The autonomous controller is allowed to optimize aggressively, but only against causal evaluation. Hard rules:

- [ ] Official forecasts are immutable once their cutoff passes.
- [ ] No post-outcome information may enter the corresponding feature snapshot.
- [ ] Hyperparameter/architecture search may inspect training + designated validation windows only.
- [ ] Repeatedly inspected historical final blocks are development data, not pristine test data.
- [ ] Promotion requires rolling-origin stability and ultimately prospective frozen forecasts.
- [ ] A model cannot win solely by improving one easy geography while collapsing Ethiopia or cold-start regimes.
- [ ] Proper scoring/calibration metrics must accompany hit-rate/distance metrics.
- [ ] Source-acquisition reward must include API/latency cost so the policy cannot win by making unlimited calls.
- [ ] The controller may archive/deprecate an experiment but must not delete its artifacts or overwrite the champion in place.

### Persistent memory for the autonomous researcher

Maintain a machine-readable experiment database. At minimum record:

```text
experiment_id
parent_experiment_ids
code_commit
data_snapshot_hash
feature_schema
model_config
agent/prompt versions
web/source configuration
training window
validation/rolling windows
prospective forecast IDs used for reward
full metric vector
compute time
API cost
status: failed / dominated / pareto / challenger / champion
reason for rejection or promotion
artifact paths
```

The proposal policy reads this history before generating the next batch. Repeating a previously failed configuration is allowed only when a relevant data/code regime has changed and the reason is recorded.

---

# Phase 13 — Promotion gate

Never auto-deploy just because training completed.

Initial gate policy to validate:

- [ ] recent prospective mean error improves by at least ~5%
- [ ] median error does not regress materially
- [ ] p90 does not materially degrade
- [ ] 25 km hit rate improves or remains stable
- [ ] probabilistic score improves
- [ ] uncertainty remains calibrated
- [ ] long-term rolling metrics worsen by <2%
- [ ] no leakage/data-integrity check fails
- [ ] source/feature pipeline health is acceptable

Promotion outcome:

```text
challenger passes all gates → promote
no challenger passes         → keep champion
```

Every model version must get an `info.blt` containing:

- training period
- shadow evaluation period
- exact metrics
- reference champion metrics
- label-confidence policy
- feature schema version
- dataset snapshot hashes
- model hash
- promotion decision
- calibration parameters
- source coverage
- notes

---

# Phase 14 — Adaptive uncertainty and abstention

Weights may update weekly, while uncertainty calibration can update more frequently.

Tasks:

- [ ] Calibrate spatial probability distribution prospectively.
- [ ] Track empirical 50%, 80%, 90% region coverage.
- [ ] Test adaptive conformal-style sequential calibration.
- [ ] Widen uncertainty when source coverage is poor.
- [ ] Widen uncertainty under model disagreement.
- [ ] Add abstention state when evidence is insufficient.

Possible abstention triggers:

```text
major source outage
extremely low source coverage
novel geography
high ensemble disagreement
out-of-distribution world state
contradictory reports
```

Public UI can display:

```text
LOW INFORMATION
FORECAST CONFIDENCE REDUCED
```

rather than manufacturing false precision.

---

# Phase 15 — Monthly/deeper retraining

Weekly fine-tuning handles local adaptation.

On a slower schedule perform a deeper retrain using all causally valid historical data.

Suggested monthly process:

1. Rebuild canonical historical event set.
2. Incorporate newly confirmed/corrected labels.
3. Recalculate source reliability.
4. Rebuild spatial state histories.
5. Train global/international model.
6. Fine-tune Ethiopia specialization.
7. Rebuild ensemble/gating weights.
8. Recalibrate uncertainty.
9. Compare against current champion using untouched rolling-origin windows.
10. Promote only through the same gate.

---

# Phase 16 — Evaluation methodology required for the 25 km target

Do not trust a single chronological split.

Use rolling-origin evaluation.

Example:

```text
train → month A
validate → month B
future test → month C

train → month B
validate → month C
future test → month D
```

Maintain:

- global historical evaluation
- Ethiopia historical evaluation
- prospective live evaluation
- recent 28-day adaptation evaluation
- source-ablation evaluation
- geography subgroup evaluation
- conflict subgroup evaluation

Metrics:

```text
mean geodesic error
median error
p75/p90/p95
within 10km
within 25km
within 50km
within 100km
negative log likelihood / probability score
Brier-style spatial probability score
calibration error
uncertainty-region coverage
abstention rate
conditional accuracy when not abstaining
```

The official claim of reaching 25 km should only be made from **prospective or genuinely untouched chronological evaluation**.

---

# Phase 17 — Ablation plan to reach 25 km

The path to 25 km should be scientific rather than speculative.

Run experiments in this order and keep an experiment ledger.

## Experiment family A — Spatial representation

- [ ] v9 candidates only
- [ ] H3 frequency baseline
- [ ] H3 res 4
- [ ] H3 res 5
- [ ] multiresolution H3
- [ ] H3 + continuous offset

## Experiment family B — Data sources

- [ ] UCDP only
- [ ] + ReliefWeb
- [ ] + Telegram structured events
- [ ] + public corroboration
- [ ] + source reliability

## Experiment family C — Semantic features

- [ ] keyword features
- [ ] event-class probabilities
- [ ] embeddings
- [ ] embeddings + confidence

## Experiment family D — Temporal windows

- [ ] event sequence only
- [ ] 24h/7d
- [ ] 1h/6h/24h/7d
- [ ] full 1h→90d multiscale

## Experiment family E — Architecture

- [ ] linear/logistic H3 baseline
- [ ] MLP
- [ ] temporal Transformer
- [ ] spatial graph model
- [ ] spatiotemporal graph Transformer
- [ ] point-process head
- [ ] ensemble

## Experiment family F — Supervision quality

- [ ] verified labels only
- [ ] AI provisional labels naively
- [ ] confidence-weighted AI labels
- [ ] source-reliability calibrated labels
- [ ] probabilistic spatial targets for ambiguous labels

## Experiment family G — Continual learning

- [ ] no weekly adaptation
- [ ] recent-only
- [ ] replay mixture
- [ ] conservative fine-tune
- [ ] adapter/LoRA-style specialization if appropriate
- [ ] gating-only update

Every experiment must save:

```text
config
seed
code commit
feature schema
training snapshot hash
validation snapshot hash
test snapshot hash
metrics
per-example predictions
```

---

# Phase 18 — Frontend design

Project name:

# THE HUMANITARIAN FORECAST

Visual direction:

**black / white / grayscale only**

Style references conceptually:

- Swiss/editorial typography
- Bloomberg/terminal information density
- geospatial command-center spatial hierarchy

Avoid:

- rounded SaaS cards
- gradients
- neon cyberpunk
- glassmorphism
- excessive animation

Use:

- sharp 1 px borders
- square controls
- large typography
- high negative space
- strong monospace metadata
- grayscale elevation/intensity

## Layout

Desktop target:

```text
┌─────────────────────────────────────────────────────────────────────┐
│ THE HUMANITARIAN FORECAST   24H FORECAST   AS OF <TIMESTAMP>       │
├──────────────────────────────────────────────────┬──────────────────┤
│                                                  │ NEXT 24 HOURS    │
│                                                  │                  │
│             LARGE 3D ETHIOPIA MAP                │ HIGHEST RISK     │
│                                                  │ REGION / ZONE    │
│        grayscale risk surfaces/contours          │ probability      │
│                                                  │                  │
│                                                  │ CONFIDENCE       │
│                                                  │                  │
│                                                  │ DRIVERS          │
│                                                  │                  │
│                                                  │ SOURCE COVERAGE  │
│                                                  │                  │
│                                                  │ MODEL VERSION    │
├──────────────────────────────────────────────────┤                  │
│ 7-DAY TIMELINE / NOW / FORECAST HORIZON          │ SUPPORT RELIEF   │
└──────────────────────────────────────────────────┴──────────────────┘
```

Rough proportions:

- ~75% map
- ~25% right information rail

## Map behavior

- [ ] Use a controllable 3D mapping stack suitable for grayscale custom styling.
- [ ] Start with Ethiopia framed at an oblique 3D camera pitch.
- [ ] Do not continuously spin the map.
- [ ] Use subtle camera transitions on region selection.
- [ ] Display risk as grayscale probability surfaces/contours/extrusions.
- [ ] Display uncertainty as an explicit contour/region.
- [ ] Public map should aggregate to humanitarian-risk zones rather than tactical points.

## Right rail

Show only interpretable high-level information:

```text
FORECAST
probability mass

CHANGE
change since previous official forecast

UNCERTAINTY
calibrated region / confidence

DRIVERS
high-level non-causal feature contributors

SOURCE COVERAGE
active primary feeds
independent corroboration count

MODEL
version
forecast timestamp
```

Use **drivers**, not causal “reasons.”

## Timeline

The timeline must allow users to view **what the model actually predicted at that historical time**.

Never regenerate past predictions with the current model and present them as historical forecasts.

Suggested control:

```text
7 DAYS AGO ───────── NOW ───────── +24H ───────── +72H
```

## Donation

Use a clearly named beneficiary/organization.

Preferred CTA:

```text
SUPPORT HUMANITARIAN RESPONSE →
```

Do not make the map feel monetized.

---

# Phase 19 — API contracts for frontend

Recommended read-only public API endpoints:

```text
GET /api/v1/status
GET /api/v1/forecast/latest
GET /api/v1/forecast/{forecast_id}
GET /api/v1/forecast/history?start=&end=
GET /api/v1/regions/{region_id}
GET /api/v1/model
```

Latest forecast should return approximately:

```json
{
  "forecast_id": "...",
  "issued_at": "...",
  "horizon": "24h",
  "model": "...",
  "public_risk_regions": [],
  "uncertainty": {},
  "confidence": "...",
  "drivers": [],
  "source_coverage": {},
  "data_cutoff": "..."
}
```

Do not expose internal raw tactical event resolution through the public endpoint.

---

# Phase 20 — Data/version schemas

Every important object needs an explicit schema version.

Minimum versioned schemas:

```text
raw-source/v1
normalized-report/v1
canonical-event/v1
world-state/v1
feature-snapshot/v1
forecast/v1
outcome-label/v1
model-info/v1
source-reliability/v1
```

A schema change should not silently alter historical records.

---

# Phase 21 — Model registry integration

Continue using:

```text
models/<subsystem>/<version>/
```

Each promoted/challenger model directory should contain:

```text
model artifact(s)
info.blt
training_metrics.json
feature_schema.json
calibration.json
```

`info.blt` must record whether the model is:

```text
research
challenger
validated
promoted
retired
```

Never overwrite an existing promoted version in place.

---

# Phase 22 — New main.py controller responsibilities

The existing controller should eventually orchestrate this full system without becoming tightly coupled to implementation details.

Desired future top-level capabilities:

```text
main.py ingest ...
main.py events ...
main.py features ...
main.py forecast ...
main.py evaluate ...
main.py train ...
main.py promote ...
main.py workflow realtime ...
main.py workflow weekly ...
main.py workflow monthly ...
main.py web ...
```

Future system registry entries should be added rather than hard-coding every component directly into the CLI.

The controller should orchestrate modules; it should not contain modeling logic itself.

---

# Phase 23 — Realtime workflow definition

Eventually define a workflow equivalent to:

```text
1. ingest raw reports
2. normalize with AI
3. deduplicate/canonicalize events
4. corroborate high-impact uncertain events
5. update source reliability metadata
6. update H3 world state
7. update live nowcast
8. write audit/provenance records
```

This path should be idempotent where possible.

---

# Phase 24 — Official daily workflow definition

```text
1. verify ingestion health
2. verify source coverage
3. freeze feature snapshot at cutoff
4. hash snapshot
5. load current champion
6. produce official probability forecast
7. calibrate uncertainty
8. run abstention rules
9. write immutable forecast ledger row
10. publish coarse public representation
```

If health checks fail severely, publish a low-information state instead of silently pretending the forecast is normal.

---

# Phase 25 — Weekly workflow definition

```text
1. lock current champion reference
2. collect newly eligible provisional/verified labels
3. update source-reliability estimates
4. select recent training window
5. select replay data
6. freeze untouched recent shadow window
7. train challenger A
8. train challenger B
9. train challenger C
10. recalibrate challengers
11. evaluate against champion
12. run leakage/integrity tests
13. run promotion gate
14. promote winner or retain champion
15. write model metadata and audit record
```

---

# Phase 26 — Monthly workflow definition

```text
1. rebuild complete historical normalized event corpus
2. merge corrected slow labels
3. rebuild source reliability
4. rebuild all causal feature histories
5. train global model
6. fine-tune Ethiopia model
7. train/fit ensemble
8. recalibrate
9. perform rolling-origin evaluation
10. compare against current champion
11. promotion gate
12. archive results
```

---

# Phase 27 — Monitoring

Track operational metrics separately from model metrics.

Operational:

- ingestion latency
- Telegram channel health
- ReliefWeb update latency
- AI parse failure rate
- geocoding failure rate
- duplicate rate
- queue depth
- feature-rollup latency
- forecast completion latency
- training success/failure

Model/data:

- source coverage
- missingness
- label-confidence distribution
- distribution shift
- ensemble disagreement
- abstention frequency
- prospective error
- calibration drift

Alert when model metrics degrade across multiple prospective windows.

---

# Phase 28 — Security and key handling

- [ ] Keep Groq/Telegram/API credentials in Secret Manager.
- [ ] Never ship secrets to frontend JavaScript.
- [ ] Log access to privileged admin/model-promotion operations.
- [ ] Make public forecast API read-only.
- [ ] Protect internal raw event endpoints.
- [ ] Keep raw source archives separate from public output.

---

# Phase 29 — Vertical-slice implementation order

When implementation begins, always create working vertical slices rather than building every subsystem in isolation.

## Slice 1 — One report to map

- [ ] ingest one Telegram/public report
- [ ] AI normalize it
- [ ] geocode to an H3 cell
- [ ] persist canonical event
- [ ] update simple H3 state
- [ ] expose coarse state through API
- [ ] render it on new black/white map

## Slice 2 — Prospective forecast ledger

- [ ] freeze state snapshot
- [ ] run baseline forecast
- [ ] save immutable forecast
- [ ] render forecast history in timeline

## Slice 3 — Outcome scoring

- [ ] add later verified event
- [ ] score frozen forecast
- [ ] display historical forecast versus outcome internally

## Slice 4 — Weekly challenger

- [ ] build eligible training set
- [ ] train one challenger
- [ ] evaluate against champion
- [ ] reject or promote automatically

## Slice 5 — Full multi-source AI supervision

- [ ] five Telegram sources
- [ ] ReliefWeb incremental ingestion
- [ ] selective web corroboration
- [ ] reliability-weighted event consensus

## Slice 6 — New H3 model

- [ ] H3 baseline
- [ ] temporal model
- [ ] spatial model
- [ ] ensemble
- [ ] uncertainty calibration

## Slice 7 — Cloud Run productionization

- [ ] deploy services/jobs
- [ ] Cloud Scheduler
- [ ] Pub/Sub
- [ ] durable database/storage
- [ ] monitoring
- [ ] health checks

---

# Phase 30 — Definition of success

The system is not considered complete merely because it can produce a map.

Success requires:

- [ ] live multi-source ingestion is stable
- [ ] AI supervision has measurable calibration against later verified labels
- [ ] official forecasts are immutable and auditable
- [ ] the model trains on world-state snapshots without temporal leakage
- [ ] weekly challengers can be trained and evaluated automatically
- [ ] failed challengers do not replace the champion
- [ ] uncertainty is empirically calibrated
- [ ] the frontend clearly communicates confidence and forecast age
- [ ] historical timeline shows the actual original forecasts
- [ ] production is deployable on Cloud Run
- [ ] prospective metrics substantially outperform v9/persistence/hotspot baselines
- [ ] the internal prospective mean-error benchmark reaches or approaches the 25 km target without sacrificing calibration or integrity

---

# Immediate next research sequence

The Ethiopia location objective is now **direct v9-lineage improvement first**. Do not restart broad replacement-model exploration before exhausting this queue.

1. [x] Retrain the v9 architecture on all available labels as production `v10`.
2. [x] Select and materialize Ethiopia-specific v10 inference calibration without changing the all-data weights.
3. [x] Write exact v9->v10 experiment receipts and preserve rejected direct variants.
4. [x] Expand the **v9 candidate support** to 64 while preserving the candidate-ranker architecture; Ethiopia development oracle mean improves **34.85 -> 23.90 km**.
5. [ ] Add local/child candidates around high-probability v9 candidates so the ~35 km candidate-oracle floor can fall materially.
6. [x] Train the v9 scorer on expanded support with 50/100/multiradius distance-aware labels. Soft100 is a validation win but not a strong later-block win; v10-prized instead uses the unmodified global v9 scorer on 64 support plus the adapted 32-support view.
7. [ ] Sweep v9-family capacity (history length, width, layers, candidate-interaction capacity) only after candidate-support ablations identify the bottleneck.
8. [ ] Add actor/motion/terrain/population/accessibility/ReliefWeb features **one family at a time into the v9 contract**; retain only features that improve Ethiopia under the fixed chronology.
9. [ ] Test warm-start/distillation from v9/v10 against fresh training; final winning recipe is retrained on **100% of available labels**.
10. [ ] Build rolling-origin Ethiopia evaluation for the winning v9-family recipe and keep prospective forecasts immutable.
11. [ ] Formalize forecast/outcome/feature snapshot schemas and the official daily cutoff/horizon.
12. [ ] Build the retrospective simulator and AI-event labeling evaluation needed for realtime ingestion.
13. [ ] Continue ReliefWeb/Telegram/public-web corroboration work and measure source/location-label error before scaling it.
14. [ ] Implement the experiment ledger + outcome reward vector + automatic challenger proposal engine offline.
15. [ ] Implement adaptive web/agent call allocation and evaluate it retrospectively against verified outcomes.
16. [ ] Run the autonomous researcher in shadow mode: propose/train/score **v9-lineage-first** challengers without production promotion.
17. [ ] Only after reward-integrity checks are strong, activate the realtime autonomous loop.
18. [ ] Enable automatic weekly challenger/promotion decisions under fixed Ethiopia gates.

H3, GeoFusion, Hawkes, and other replacement representations remain available as archived experiments or auxiliary teachers, but they are **not the default next step**. The goal remains a forecasting process that is causal, measurable, reversible, calibrated, and demonstrably improving while using all available labels for the final production retrain.
