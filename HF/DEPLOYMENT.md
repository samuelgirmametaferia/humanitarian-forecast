# Public deployment and secrets

The web client is safe to publish. It contains no private credentials. Keep that boundary: any variable beginning with `VITE_` is compiled into browser JavaScript and must be treated as public.

## Production configuration

Set these in the hosting provider's encrypted environment-variable store, never in a committed `.env` file:

| Variable | Scope | Purpose |
|---|---|---|
| `HF_DATABASE_URL` | server | PostgreSQL connection string for forecast snapshots |
| `HF_INGEST_HMAC_SECRET` | server + GitHub Actions | signs and verifies forecast ingestion; use a 32-byte random value |
| `HF_CRON_SECRET` | server + GitHub Actions | protects reconciliation runs; use a separate 32-byte random value |
| `CRON_SECRET` | Vercel | set to the same value as `HF_CRON_SECRET`; Vercel uses this exact name to sign cron requests |
| `HF_UPSTASH_REDIS_REST_URL` | server | shared production rate-limit store |
| `HF_UPSTASH_REDIS_REST_TOKEN` | server | Redis access token |
| `HF_ALLOWED_ORIGINS` | server | comma-separated deployed web origins |
| `HF_MODEL_DIR` | server | optional path to the parity-validated ONNX package |
| `HF_MODEL_REGISTRY_URL` | server | manifest URL of the model registry release; the API promotes new registry versions automatically and falls back to `HF_MODEL_DIR` on any failure |
| `GROQ_API_KEY` | GitHub Actions only | optional source-processing provider; never needed by the browser |
| `HF_PUBLIC_PREVIEW_CHANNELS` | GitHub Actions variable | comma-separated public Telegram channel names; no account or Telegram credential is used |
| `HF_RECONCILE_URL` | GitHub Actions only | deployed `/api/v1/learn/reconcile` URL for the scheduled reconciliation check |
| `HF_TRAINING_PAIRS_URL` | GitHub Actions only | deployed `/api/v1/learn/training-pairs` URL the reconcile workflow pulls stored feature payloads from |

Set `HF_PROVIDER_MODE=production` only after all required server values are present. The API fails closed when production configuration is incomplete.

The complete idempotent PostgreSQL schema is also available at [`docs/production-schema.sql`](docs/production-schema.sql). The API runs the same schema automatically when it first connects, so manually running the SQL is optional but useful for checking permissions before the first deploy.

## GitHub

Before making the repository public:

1. Run `git grep -nE '(api[_-]?key|secret|token|password)\s*[:=]\s*[^$<{]'` and review every match.
2. Add the ingest URL, HMAC secret, and any provider key through **Repository settings → Secrets and variables → Actions**.
3. Add the same server-only values to the deployment platform's encrypted environment settings.
4. Protect the default branch and require the frontend and backend test jobs.
5. If a real credential has ever been committed, rotate it before publishing; deleting the current file is not enough because Git history retains it.

The repository includes four workflows: CI validates the frontend and API on every pull request, the public-source workflow publishes forecasts every six hours, the reconcile workflow labels matured forecasts every third day, and the retrain workflow promotes a new model every Sunday. Scheduled workflows run on the default branch only.

The public-source workflow runs every six hours. It reads only anonymous `t.me/s/<channel>` previews, discovers Groq's current model inventory, excludes non-text model families, probes candidates, and caches the first working text model. A cached model is replaced automatically if its probe fails. Extracted rows retain source URL, publication time, retrieval time, city coordinates, confidence, and a hash of the source text. They are model input signals, never labels or verified events.

## Vercel

Import the repository with **Root Directory** set to `HF`. Vercel detects `vercel.json`, builds the Vite client, serves the FastAPI endpoint, and calls the protected reconciliation endpoint daily. Set `CRON_SECRET` and `HF_CRON_SECRET` to the same unique value. Then set `HF_PROVIDER_MODE=production` only after the database, ingest signing secret, Redis URL/token, and `HF_ALLOWED_ORIGINS` are in place.

The three-day job is a reconciliation and validation gate: it labels matured published forecasts with realized UCDP GED outcomes and appends them to the live training-pairs dataset (the `live-training-data` release). Labels come only from UCDP verified event data; public preview signals are model input features, never labels.

## Continuous retraining and the model registry

Every Sunday the `retrain` workflow fine-tunes the promoted model on the Ethiopia training rows plus the accumulated live-validated pairs, exports the ONNX serving package, validates PyTorch/ONNX parity, and publishes it to the GitHub Releases model registry. Promotion is automatic:

1. A versioned archive release (`model-retrain-*`) keeps every generation for rollback.
2. The `model-registry-current` release is repointed at the new package.
3. The API polls the registry manifest (at most every 10 minutes per instance), verifies member checksums, and serves the new version. The ingest and `/api/v1/models` paths force a registry check regardless of the poll window, so every published forecast reflects the current champion. A registry that is unreachable or fails validation leaves the current model in place.

Mechanical guards, not human gates, protect production: a package reaches the registry only if ONNX parity passes, the manifest is checksum-verified by the serving layer, and the retrained validation metric is finite and has not collapsed relative to its parent. The `base-dataset` release holds the deterministic v6 training set the retrain continues from.

The checked-in `.env.example` documents names only. `.env` and `.env.*` are ignored, except for that example.

## Public data provenance

Population totals and density shown in the demo snapshot are WorldPop Global 2 R2025A 2026 estimates at 100 m resolution, calculated over consistent windows of roughly 400 km² around each candidate site. Place context is a cached, small OpenStreetMap-derived extraction under ODbL. Terrain uses Mapterhorn tiles in the browser and Copernicus DEM in the offline feature pipeline. Preserve attributions whenever layers or exports are redistributed.
