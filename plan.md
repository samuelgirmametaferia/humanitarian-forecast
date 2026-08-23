# The Humanitarian Forecast — Execution Plan

## Objective

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

When work resumes, do **not** start by rewriting the model blindly.

Execute this order:

1. [ ] Formalize forecast/outcome/feature snapshot schemas.
2. [ ] Formalize official daily cutoff and horizon.
3. [ ] Build a retrospective simulator that reproduces the exact realtime information boundary.
4. [ ] Build canonical AI-event labeling evaluation using historical reports whose later outcomes are known.
5. [ ] Measure AI location-label error before using those labels at scale.
6. [ ] Build H3 historical baseline from existing UCDP data.
7. [ ] Compare H3 oracle/coverage against the v9 candidate oracle.
8. [ ] Add current ReliefWeb-derived features to the H3 representation.
9. [ ] Add simulated Telegram/AI event features from historical public reports where available.
10. [ ] Train first spatiotemporal H3 model.
11. [ ] Ensemble it with v9 and persistence.
12. [ ] Run rolling-origin Ethiopia evaluation.
13. [ ] Perform ablations until the largest remaining error sources are identified.
14. [ ] Only after the offline architecture is strong, activate the realtime autonomous loop.
15. [ ] Finally deploy weekly challenger/promotion automation.

The goal is not simply to make the system autonomous. The goal is to make autonomy operate on top of a forecasting process that is **causal, measurable, reversible, calibrated, and demonstrably improving**.
