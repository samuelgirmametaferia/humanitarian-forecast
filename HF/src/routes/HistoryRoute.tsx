import { ContentRoute } from './ContentRoute'
import type { ProjectHistoryEvent } from '../contracts/forecast'
import { formatDate } from '../lib/format'
import { useProjectData } from '../hooks/useProjectData'

const eventNames: Record<ProjectHistoryEvent['kind'], string> = {
  dataset_update: 'Dataset update',
  model_run: 'Model run',
  training: 'Training run',
  evaluation: 'Evaluation',
  prediction_batch: 'Prediction batch',
  model_version: 'Model version',
}

export function HistoryRoute() {
  const { data, error, loading } = useProjectData()
  if (loading) return <main className="route-state" aria-live="polite"><h1>Loading history</h1></main>
  if (error || !data) return <main className="route-state" aria-live="assertive"><h1>History unavailable</h1><p>{error ?? 'History could not be loaded.'}</p></main>
  const entries = [...data.history.entries].sort((a, b) => Date.parse(b.occurredAt) - Date.parse(a.occurredAt))
  return <ContentRoute title="History" intro="Recorded prediction, data, model, and evaluation activity.">
    <section className="history-summary"><h2>Recorded activity</h2><p>Evaluation reconciliation is configured on a {data.history.updateCycleDays}-day cycle. Entries appear only after work has occurred.</p></section>
    <section><h2>Timeline</h2>{entries.length === 0 ? <p className="empty-state">No lifecycle history has been recorded.</p> : <ol className="history-timeline">{entries.map((entry) => <li key={entry.id}>
      <div className="history-marker" aria-hidden="true" />
      <article><div className="history-entry-meta"><time dateTime={entry.occurredAt}>{formatDate(entry.occurredAt, true)}</time><span>{eventNames[entry.kind]}</span>{entry.mode === 'demo' && <span>Demonstration</span>}</div>
        <h3>{entry.modelName ? `${entry.modelName}${entry.modelVersion ? ` ${entry.modelVersion}` : ''}` : eventNames[entry.kind]}</h3>
        {entry.changes.length > 0 && <ul>{entry.changes.map((change) => <li key={change}>{change}</li>)}</ul>}
        {Object.keys(entry.metrics).length > 0 && <dl className="event-metrics">{Object.entries(entry.metrics).map(([name, value]) => <div key={name}><dt>{metricName(name)}</dt><dd>{formatMetric(value)}</dd></div>)}</dl>}
        {entry.reportUrl && <a href={entry.reportUrl} target="_blank" rel="noreferrer">Open report</a>}
      </article>
    </li>)}</ol>}</section>
  </ContentRoute>
}

function metricName(value: string) { return value.replace(/([a-z])([A-Z])/g, '$1 $2').replace(/^./, (letter) => letter.toUpperCase()) }
function formatMetric(value: number) { return new Intl.NumberFormat('en', { maximumFractionDigits: 3 }).format(value) }
