# Model Card: Humanitarian Forecast

## Status

**Research system. Not approved for autonomous operational decisions.**

The repository now distinguishes validated reference models from all-data production retrains. A production retrain consumes every available label and therefore does not receive a fabricated new holdout score; its `info.blt` points back to the validated ancestor used to select the recipe.

## Validated next-location reference

Model directory:

```text
models/location/candidate_ranker/v9/
```

Promoted checkpoint:

```text
models/location/candidate_ranker/v9/candidate_ranker_calibrated.pt
```

Recipe: `conflict_candidate_transformer_spatial_rings_geometric_median_v9`.

### Data

- UCDP GED 26.1 georeferenced events
- ReliefWeb country/day context
- Telegram excluded from this location pipeline
- 157,932 chronologically ordered next-event examples
- 1,517 loaded UCDP conflicts; 120 countries represented in usable sequences
- 16 historical events per example
- 19 event-history features (`geo-v2`)
- up to 32 cutoff-safe conflict-history candidate locations
- 28 features per candidate, including spatial activity/fatality rings

### Fixed chronological protocol

- first 70%: phase-1 training
- next 15%: model/calibration selection
- first 85%: fresh phase-2 retraining for the validation-selected epoch count
- final 15%: untouched test evaluation
- fixed test size: 23,690 examples
- test interval: June 2023 through December 2025

### Architecture and objective

- 3-layer Transformer history encoder
- width 128, 4 attention heads, FFN width 384, dropout 0.1
- learned country and conflict embeddings
- candidate encoder + learned candidate bias
- 677,599 parameters
- training loss: candidate cross-entropy + `50 ×` probability-weighted center distance
- expected-candidate-distance weight: `0`
- selected epoch: 12
- output calibration: temperature `1.0`, weighted geometric median

### Untouched-test performance

| Metric | Result |
|---|---:|
| Median center error | **61.01 km** |
| Mean center error | **194.36 km** |
| P90 center error | **435.09 km** |
| Within 25 km | **34.39%** |
| Candidate oracle mean | **35.99 km** |

The candidate-oracle metric asks how close the best of 32 generated candidates was after seeing the target. It is a diagnostic of candidate coverage, not top-1 forecasting accuracy.

## Ethiopia production v10 — direct v9 lineage

The active production-weight artifact is `models/location/candidate_ranker/v10/`. It keeps the v9 candidate-ranker architecture and objective and retrains the recipe on **all 157,932 currently materialized labels** rather than permanently withholding the old final 15%. The corpus contains **2,669 Ethiopia examples through 2025-12-29**. Because every available label is used, this all-data checkpoint has no fabricated new untouched-test score; recipe evidence comes from the frozen historical experiments recorded in `reports/location/v10_direct_v9lineage_receipts.json`.

For Ethiopia, `models/location/candidate_ranker_ethiopia/v10/` is a calibration wrapper around the **same tensor weights**. Validation-only selection minimized mean error subject to no regression at the 50 km and 100 km thresholds on the selection block. It chose a weighted geometric median at temperature `0.8`, blended 50% with the causal `recent_8_mean` conflict-history reference. Applied unchanged to the later 968-event Ethiopia block, this reduced mean center error from **181.85 km to 178.33 km**, median from **158.58 km to 153.62 km**, and P90 from **357.31 km to 343.25 km**, while <=100 km improved from **30.68% to 31.71%**. <=25 km and <=50 km declined on the later block, so the next v9-lineage work must target candidate support/ranking rather than pretending calibration solved localization.

Two direct v9 continuations were rejected and preserved: the richer actor-transfer tensor improved global validation but regressed Ethiopia geometric-median mean (`182.90 km` vs `181.85 km`), and direct Ethiopia weight fine-tuning improved the later-block mean by only about `0.10 km`. These failures are receipts, not promoted models.

### Ethiopia v10-prized — direct v9 two-support artifact

`models/location/candidate_ranker/v10_prized/` is the current **internal v9-lineage development champion for Ethiopia mean/tail center error**. It does not replace v9's historical validation claim: the final 15% block has been repeatedly inspected and is development data, not a pristine prospective test.

The recipe deliberately keeps the `ConflictCandidateRanker` architecture. It combines two views of the same v9 family: (1) an Ethiopia-adapted 32-candidate model produced by one late-layer-only epoch on all **2,669 Ethiopia labels**, and (2) the all-data global v10 weights evaluated over a new **64-candidate, 28-feature** support bank. The 64-candidate bank lowers Ethiopia candidate-oracle mean from **34.85 km to 23.90 km** on the later block. The frozen center blend is 50% adapted-32 weighted geometric median at temperature `1.0` and 50% global-64 weighted mean at temperature `0.65`.

Aligned Ethiopia historical development receipts (`reports/location/v10_prized_receipts.json`):

| Metric | aligned v9 parent | v10-prized | Delta |
|---|---:|---:|---:|
| Mean center error | 178.59 km | **177.20 km** | **-1.40 km** |
| Median center error | 156.83 km | **155.97 km** | **-0.86 km** |
| P90 center error | 340.68 km | **334.17 km** | **-6.51 km** |
| Within 100 km | 30.89% | **30.99%** | **+0.10 pp** |
| Within 50 km | **10.43%** | 9.61% | -0.82 pp |
| Within 25 km | **3.51%** | 2.27% | -1.24 pp |

The result is therefore a **mean/tail improvement with a close-range tradeoff**, not a universal win. The production package is retrained/materialized with all historical labels and has `held_out_evaluation: false`; prospective and rolling-origin confirmation remain required before calling it a validated replacement for v9. The registered `location.v10_prized.predict` entry point emits only a coarse humanitarian zone and does not expose ranked tactical coordinates.

## Temporal risk references

Historical global and Ethiopia risk artifacts have been migrated to:

```text
models/risk/base/vN/
models/risk/ethiopia/vN/
```

Their legacy metrics are preserved in each version's `info.blt`. The validated reference versions used by the new production workflow are `models/risk/base/v2/` and `models/risk/ethiopia/v4/`.

### Ethiopia v6 spatial-risk challenger

Model directory:

```text
models/risk/ethiopia/v6/
```

v6 is a **coarse humanitarian early-warning** challenger, not a precise next-event location model. It preserves the v5 local temporal-risk predictor as a stable expert and adds cutoff-safe PRIO-GRID neighborhood histories. The spatial tensor contains the eight local channels plus first- and second-ring historical neighbor aggregates and historical active-neighbor counts. A separate spatial model is optimized for future intensity so the rare-event ranking and magnitude objectives do not fight over one checkpoint.

The escalation ensemble weight, checkpoint epochs, affine probability calibration, classification threshold, and split-conformal intensity radii are selected only on the chronological validation partition. The final 15% has been inspected during iterative research and is therefore reported as a **development holdout**, not as a pristine prospective test.

| Metric | v5 development | v6 development | Relative change |
|---|---:|---:|---:|
| Escalation average precision | 0.13436 | **0.14688** | **+9.32%** |
| Intensity MAE (log1p) | 0.16383 | **0.13684** | **-16.47%** |
| Brier score | 0.06023 | **0.06009** | lower is better |
| Log loss | 0.23368 | **0.23258** | lower is better |

The paired bootstrap 95% interval for the v6-v5 average-precision difference on the development block is **+0.0040 to +0.0253**. This is a development diagnostic, not a substitute for future prospective scoring. Promotion still requires rolling-origin and genuinely prospective confirmation.

v6 also stores validation split-conformal absolute residual radii for intensity at 80%, 90%, and 95% target coverage. These intervals describe historical model uncertainty; they are not guarantees about field conditions.

## Production all-data policy

`main.py workflow full` trains global models on all currently available labels, then fine-tunes Ethiopia-specialized copies. The default new version destinations are:

```text
models/risk/base/v3/
models/location/candidate_ranker/v10/
models/risk/ethiopia/v5/
models/location/candidate_ranker_ethiopia/v1/
```

These directories are created when the workflow is actually run. Production training includes data that formerly belonged to the evaluation period, but each historical training example remains causal: its features and candidate bank use only observations available by that example's own cutoff.

Because no data is withheld from an all-data production artifact, its `info.blt` contains `held_out_evaluation: false`. Generalization claims continue to come from the corresponding validated reference until a later-period evaluation becomes available.

## Required interpretation

- Forecasts are probabilistic research signals, not verified incidents.
- Candidate-set oracle metrics are not top-1 accuracy.
- Training loss is not a substitute for held-out geographic error.
- Exact model performance claims must identify the model version and evaluation protocol.
- Public/operational humanitarian decisions require independent corroboration and human review.
