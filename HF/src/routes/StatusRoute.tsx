import type { ForecastSnapshot, ProjectHistoryEvent } from '../contracts/forecast'
import { formatDate, formatPercent } from '../lib/format'
import { useProjectData } from '../hooks/useProjectData'
import { ContentRoute, Definition } from './ContentRoute'

export function StatusRoute() {
  const { data, error, loading } = useProjectData()
  if (loading) return <main className="route-state" aria-live="polite"><h1>Loading status</h1></main>
  if (error || !data) return <main className="route-state" aria-live="assertive"><h1>Status unavailable</h1><p>{error ?? 'Status data could not be loaded.'}</p></main>
  const forecasts = [...data.forecasts].sort((a, b) => Date.parse(b.generatedAt) - Date.parse(a.generatedAt))
  const latest = forecasts[0]
  const evaluations = data.history.entries.filter((entry) => entry.kind === 'evaluation')
  const stale = latest ? data.history.entries.some((entry) => Date.parse(entry.occurredAt) - Date.parse(latest.generatedAt) > latest.horizonDays * 86_400_000) : false
  return <ContentRoute title="Status" intro="Service state, current model metadata, and recorded data freshness.">
    <section><h2>Service</h2><div className={`status-overview status-${data.health.status}`}><span className="status-dot" aria-hidden="true" /><strong>{data.health.status === 'ok' ? 'Available' : 'Degraded'}</strong></div><dl>
      <Definition term="Mode">{data.health.mode}</Definition><Definition term="Writes">{data.health.writesEnabled ? 'Enabled' : 'Disabled'}</Definition>
      {latest && <Definition term="Last prediction"><time dateTime={latest.generatedAt}>{formatDate(latest.generatedAt, true)}</time>{stale && <span className="stale-label">Older than its {latest.horizonDays}-day horizon</span>}</Definition>}
    </dl></section>
    {!latest ? <section><h2>Forecast</h2><p className="empty-state">No forecast snapshot is available.</p></section> : <>
      <section><h2>Current model</h2><dl><Definition term="Model">{latest.model.name}</Definition><Definition term="Version">{latest.model.version}</Definition><Definition term="Feature contract">{latest.model.featureContract}</Definition><Definition term="Candidate support">{latest.model.candidateCount}</Definition></dl></section>
      <section><h2>Source freshness</h2><div className="source-grid">{latest.sourceHealth.map((source) => <article key={source.id}><div><span className={`status-dot ${source.status}`} aria-hidden="true" /><h3>{source.label}</h3></div><strong>{source.status}</strong>{source.lastSuccessfulAt && <p>Last successful {formatDate(source.lastSuccessfulAt, true)}</p>}<p>{source.articlesProcessed.toLocaleString()} items processed</p></article>)}</div></section>
    </>}
    <PredictionHistory forecasts={forecasts} />
    <section><h2>Evaluation history</h2>{evaluations.length === 0 ? <p className="empty-state">No evaluation results have been recorded.</p> : <EvaluationTable entries={evaluations} />}</section>
  </ContentRoute>
}

function PredictionHistory({ forecasts }: { forecasts: ForecastSnapshot[] }) {
  return <section><h2>Prediction history</h2>{forecasts.length === 0 ? <p className="empty-state">No prediction history is available.</p> : <><p>{forecasts.length.toLocaleString()} recorded prediction {forecasts.length === 1 ? 'batch' : 'batches'}.</p>{forecasts.length > 1 && <PredictionPlot forecasts={forecasts} />}<div className="table-scroll"><table className="status-table"><thead><tr><th>Date</th><th>Model</th><th>Version</th><th>Zones</th></tr></thead><tbody>{forecasts.map((forecast) => <tr key={forecast.id}><td><time dateTime={forecast.generatedAt}>{formatDate(forecast.generatedAt)}</time></td><td>{forecast.model.name}</td><td>{forecast.model.version}</td><td>{forecast.zones.length}</td></tr>)}</tbody></table></div></>}</section>
}

function PredictionPlot({ forecasts }: { forecasts: ForecastSnapshot[] }) {
  const ordered = [...forecasts].reverse()
  const max = Math.max(...ordered.map((item) => item.zones.length), 1)
  return <figure className="prediction-chart"><figcaption>Zones returned per prediction batch</figcaption><svg viewBox="0 0 640 180" role="img" aria-label="Bar chart of zones returned by prediction batch">{ordered.map((item, index) => { const width = 520 / ordered.length; const height = item.zones.length / max * 120; return <g key={item.id}><title>{formatDate(item.generatedAt)}: {item.zones.length} zones</title><rect x={70 + index * width} y={140 - height} width={Math.max(4, width - 4)} height={height} tabIndex={0} /><text x={70 + index * width + width / 2} y="160">{index + 1}</text></g> })}<line x1="65" x2="600" y1="140" y2="140" /></svg></figure>
}

function EvaluationTable({ entries }: { entries: ProjectHistoryEvent[] }) {
  return <div className="table-scroll"><table className="status-table"><thead><tr><th>Date</th><th>Model</th><th>Metrics</th></tr></thead><tbody>{entries.map((entry) => <tr key={entry.id}><td>{formatDate(entry.occurredAt)}</td><td>{entry.modelName ?? 'Not recorded'} {entry.modelVersion ?? ''}</td><td>{Object.entries(entry.metrics).length === 0 ? 'No metrics recorded' : Object.entries(entry.metrics).map(([name, value]) => <span className="metric-value" key={name}>{name}: {value >= 0 && value <= 1 ? formatPercent(value) : value}</span>)}</td></tr>)}</tbody></table></div>
}
