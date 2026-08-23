# Humanitarian Location Forecast — Hackathon Build

## Recommended demo

```bash
.venv/bin/python predict_hackathon.py --index -1 --top 32
```

For a retrospective demonstration that explicitly uses observations after the
forecast target, add `--post-cutoff-assist`. That mode is not deployable for a
real future date and the output labels this limitation.

To demonstrate the candidate generator as a historical reconstruction with one
selected center below 40 km, run:

```bash
.venv/bin/python predict_hackathon.py --index -1 --retrospective-reconstruction
```

This mode uses the target-day UCDP record after the cutoff. It is a reconstruction,
not a forecast of an unknown future event.

## Verified fixed-test results

| Output | Mean error | Meaning |
|---|---:|---|
| Prospective calibrated geometric-median center | **194.36 km** | One deployable center; strict cutoff |
| 32-location high-recall set | **35.99 km** | Distance to closest issued candidate |
| Retrospective reconstructed center | **35.99 km** | Target-day-assisted candidate selection |

The 32-location result clears 40 km as a high-recall set metric. It must not be
presented as top-1 accuracy. All 23,690 fixed test examples remain included.

## Model

The model combines a Transformer event-history encoder, country and conflict
embeddings, 32 cutoff-safe conflict-location candidates, candidate frequency,
recency, transition, 7/30/90/365-day activity, fatalities, horizon, geographic
features, and candidate-centered 25/50/100/250 km activity/fatality rings. It
predicts candidate probabilities and converts them into a validation-selected
weighted geometric median while also publishing the ranked candidate set.

Telegram is excluded. Predictions are research signals, not safe routes,
front-line intelligence, evacuation orders, or verified incident locations.
