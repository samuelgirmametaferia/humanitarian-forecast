import { X } from 'lucide-react'
import type { ForecastSnapshot } from '../../contracts/forecast'
import { formatPercent } from '../../lib/format'
import { useForecastStore } from '../../lib/store'

export function EvidenceInspector({ data }: { data: ForecastSnapshot }) {
  const selectedId = useForecastStore((state) => state.selectedZoneId)
  const select = useForecastStore((state) => state.setSelectedZone)
  const zone = data.zones.find((item) => item.id === selectedId)
  if (!zone) return <aside className="inspector inspector-empty" aria-label="Forecast zone inspector" data-tour="inspector"><p>Select a zone to inspect its forecast data.</p></aside>
  return <aside className="inspector" aria-label="Selected forecast zone" data-tour="inspector">
    <div className="inspector-topline"><span>Rank {zone.rank} of {data.zones.length}</span><button type="button" onClick={() => select(null)} aria-label="Clear selected zone"><X aria-hidden="true" /></button></div>
    <h2>{zone.label}</h2>
    <div className="probability-reading"><strong>{formatPercent(zone.probability)}</strong><span>relative candidate score</span></div>
    <dl className="metrics-grid">
      <div><dt>Zone radius</dt><dd>{Math.min(zone.radiusKm, 20)} km</dd></div>
      <div><dt>Uncertainty</dt><dd>{zone.uncertainty === null ? 'Not available' : formatPercent(zone.uncertainty)}</dd></div>
      <div><dt>Site recency</dt><dd>{zone.daysSinceLastEvent} days</dd></div>
      <div><dt>People nearby</dt><dd>{zone.populationEstimate == null ? 'Not available' : zone.populationEstimate.toLocaleString()}</dd></div>
      <div><dt>Population density</dt><dd>{zone.populationDensityPerKm2 == null ? 'Not available' : `${Math.round(zone.populationDensityPerKm2).toLocaleString()} / km²`}</dd></div>
      <div><dt>Elevation</dt><dd>{zone.elevationM.toLocaleString()} m</dd></div>
      <div><dt>Ruggedness</dt><dd>{Math.round(zone.ruggednessM)} m</dd></div>
    </dl>
    {zone.adminArea && <p className="inspector-kind">{zone.position.latitude.toFixed(2)}°N, {zone.position.longitude.toFixed(2)}°E · {zone.adminArea}</p>}
    {zone.populationSource && <section className="inspector-section data-source"><h3>Population context</h3><p>{zone.populationYear} estimate across {Math.round(zone.populationAreaKm2 ?? 0).toLocaleString()} km². Source: {zone.populationSource}. Population estimates are not official government statistics.</p></section>}
    <section className="inspector-section"><h3>Model drivers</h3><ul>{zone.drivers.map((driver) => <li key={driver}>{driver}</li>)}</ul></section>
    <p className="inspector-kind">{zone.siteType}</p>
  </aside>
}
