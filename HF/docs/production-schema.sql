-- Humanitarian Forecaster production schema
-- PostgreSQL 14+. The API also runs this idempotently on first connection.

CREATE TABLE IF NOT EXISTS hf_forecasts (
    id text PRIMARY KEY,
    generated_at timestamptz NOT NULL,
    payload jsonb NOT NULL
);

CREATE INDEX IF NOT EXISTS hf_forecasts_generated_at_idx
    ON hf_forecasts (generated_at DESC);

CREATE TABLE IF NOT EXISTS hf_ingest_runs (
    run_id text PRIMARY KEY,
    body_sha256 char(64) NOT NULL,
    forecast_id text NOT NULL UNIQUE REFERENCES hf_forecasts(id),
    published_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS hf_history_events (
    id text PRIMARY KEY,
    occurred_at timestamptz NOT NULL,
    payload jsonb NOT NULL
);

CREATE INDEX IF NOT EXISTS hf_history_events_occurred_at_idx
    ON hf_history_events (occurred_at DESC);
