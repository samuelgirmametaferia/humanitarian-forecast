# Model card: humanitarian temporal area-risk forecaster

## Status

**Research prototype — not approved for operational evacuation decisions.**

The recommended current checkpoint is
`checkpoints_mixture_geo_v2/mixture_best.pt`.
It excludes all Telegram-derived data and uses UCDP georeferenced events plus
ReliefWeb country/day context.

## Telegram-free next-location model

- 157,932 sequences from 1,517 loaded UCDP conflicts
- 120 countries represented in usable sequences
- 16 prior events per input sequence
- Probabilistic center plus isotropic uncertainty radius
- Validation-calibrated for 80% circle containment
- Untouched test period: June 2023 through December 2025

Untouched-test results:

| Metric | Result |
|---|---:|
| Samples | 23,690 |
| Median center error | 105.8 km |
| Mean center error | 255.3 km |
| 90th-percentile center error | 510.7 km |
| Circle coverage | 80.8% |
| Median radius | 192.7 km |
| Mean radius | 332.8 km |

The last-observed-location baseline has lower median error (89.3 km), but worse
mean error (285.1 km) and worse p90 error (634.8 km). The learned model therefore
improves large misses but does not dominate the simple baseline on every metric.

## Five-circle mixture checkpoint

`checkpoints_mixture/mixture_best.pt` returns five alternative centers with learned
probabilities and per-component uncertainty. On the untouched test period:

| Metric | Result |
|---|---:|
| Top-1 median error | 91.1 km |
| Top-1 mean error | 250.2 km |
| Top-1 p90 error | 555.5 km |
| Best-of-five median error | 42.3 km |
| Best-of-five p90 error | 244.8 km |
| Union circle coverage | 82.3% |
| Median component radius | 57.2 km |

## Geography-aware five-circle checkpoint

`checkpoints_mixture_geo_v2/mixture_best.pt` adds absolute anchor coordinates and
cyclical season features. On the same 23,690-example untouched test split:

| Metric | Result |
|---|---:|
| Top-1 median error | 90.6 km |
| Top-1 mean error | 245.3 km |
| Top-1 p90 error | 524.3 km |
| Best-of-five median error | 42.5 km |
| Best-of-five p90 error | 239.4 km |
| Union circle coverage | 83.9% |
| Median component radius | 50.2 km |

The explicit stretch target is 25 km top-1 mean error. It has not been achieved.
Two later experiments were rejected: explicit per-step movement features produced
250.1 km mean error, and a first ranking-loss experiment produced 267.1 km. Their
artifacts remain for reproducibility but are not recommended.

Best-of-five error measures whether at least one proposed center is close; it does
not imply the model knows in advance which component will be correct. Component
probabilities and circle coverage must be shown together.

The model estimates whether documented conflict activity in a coarse area is
likely to increase during the next seven days based on a 28-day history. It does
not estimate exact front lines, troop locations, safe routes, or individual risk.

## Architecture

- Entity-agnostic temporal Transformer
- 4 encoder layers, width 128, 8 attention heads
- Eight daily aggregate features
- Two outputs: future log activity and escalation logit
- Approximately 2.6 MB per saved checkpoint

## Data

International base checkpoint:

- UCDP GED 26.1 georeferenced events
- ReliefWeb reports retrieved with appname `HMA-research-W5F2P`
- 3,664,981 sequences across 5,056 geographic entities
- Coverage: 1989–2026
- Holdout cutoff: 2021-08-21

Ethiopia checkpoint:

- Starts from the saved international base checkpoint
- 78,866 sequences across 226 geographic entities
- Historical UCDP Ethiopia events plus filtered named-area Telegram events
- Coverage: 1989-01-30 through 2025-12-22
- Training ends: 2023-10-18
- Validation: 2023-10-18 through 2024-12-22
- Untouched test: 2024-12-22 through 2025-12-22
- Direct-Amharic Nemotron rows without machine translation are excluded because
  manual audits found severe hallucination and number interpretation failures.

## Current evaluation

| Checkpoint | Holdout positive rate | Average precision | Recall at 0.5 | Intensity MAE |
|---|---:|---:|---:|---:|
| International base | 0.062 | 0.122 | 0.740 | 0.200 |
| Ethiopia historical fine-tune | 0.066 | 0.123 | 0.426 | 0.173 |

At the validation-selected threshold of 0.66, the untouched Ethiopia test has
81.4% ordinary accuracy, 63.4% balanced accuracy, 16.0% precision, 42.6% recall,
and F1 0.233. Ordinary accuracy remains misleading because a model that always
predicts no escalation would score 93.4% while detecting zero escalations.
Average precision and balanced accuracy are more informative for this rare event.

## Required before operational use

1. Expand and manually audit Ethiopia area labels through 2026.
2. Deduplicate reports and prevent source copying from inflating evidence.
3. Add rolling-origin backtests across several time periods.
4. Calibrate probabilities on a separate validation interval.
5. Compare against persistence, seasonal, and count-based baselines.
6. Evaluate false negatives separately for civilian-harm events.
7. Require independent-source corroboration and human review.
8. Publish only delayed, coarse administrative-area outputs with uncertainty.

## Prohibited use

- Exact or real-time force tracking
- Target selection or military operational planning
- Identification of individuals or units
- Representing predictions as verified facts or official evacuation orders
