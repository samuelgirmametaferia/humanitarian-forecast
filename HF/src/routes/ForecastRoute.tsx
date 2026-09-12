import { EvidenceInspector } from '../features/forecast/EvidenceInspector'
import { ForecastHeader } from '../features/forecast/ForecastHeader'
import { ForecastTable } from '../features/forecast/ForecastTable'
import { ForecastMap } from '../features/map/ForecastMap'
import { LayerControls } from '../features/map/LayerControls'
import { useForecast } from '../hooks/useForecast'

export function ForecastRoute() {
  const { data, error, loading } = useForecast()
  if (loading) return <main className="route-state" aria-live="polite"><div className="loading-orbit" aria-hidden="true" /><h1>Loading forecast</h1></main>
  if (error || !data) return <main className="route-state" aria-live="assertive"><h1>Forecast unavailable</h1><p>{error ?? 'The forecast response could not be validated.'}</p><button type="button" onClick={() => window.location.reload()}>Try again</button></main>
  return <main id="main-content" tabIndex={-1} className="forecast-route">
    <ForecastHeader data={data} />
    <div className="forecast-workspace"><div className="map-stage" data-tour="map"><ForecastMap data={data} /><LayerControls /></div><EvidenceInspector data={data} /></div>
    <ForecastTable data={data} />
  </main>
}
