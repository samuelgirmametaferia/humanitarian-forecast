import { useMemo, useState } from 'react'
import type { ForecastSnapshot } from '../../contracts/forecast'
import { useForecastStore } from '../../lib/store'

type SortKey = 'rank' | 'probability' | 'uncertainty' | 'daysSinceLastEvent'

export function ForecastTable({ data }: { data: ForecastSnapshot }) {
  const open = useForecastStore((state) => state.tableOpen)
  const select = useForecastStore((state) => state.setSelectedZone)
  const [sort, setSort] = useState<SortKey>('rank')
  const rows = useMemo(() => [...data.zones].sort((a, b) => {
    if (sort === 'rank') return a.rank - b.rank
    const aValue = a[sort] ?? -1
    const bValue = b[sort] ?? -1
    return bValue - aValue
  }), [data.zones, sort])
  if (!open) return null
  return (
    <section className="forecast-table-wrap" aria-labelledby="table-title">
      <div className="panel-heading"><div><h2 id="table-title">Forecast data table</h2><p>Complete keyboard-accessible equivalent to the visible map zones.</p></div><button type="button" onClick={() => useForecastStore.getState().setTableOpen(false)}>Close table</button></div>
      <div className="table-scroll"><table><caption>{data.zones.length} spatially diverse candidate zones for the {data.horizonDays}-day horizon</caption><thead><tr>
        <th><button onClick={() => setSort('rank')}>Rank</button></th><th>Site</th><th><button onClick={() => setSort('probability')}>Probability</button></th><th><button onClick={() => setSort('uncertainty')}>Uncertainty</button></th><th>Radius</th><th><button onClick={() => setSort('daysSinceLastEvent')}>Site recency</button></th><th>Population</th><th>Density</th>
      </tr></thead><tbody>{rows.map((zone) => <tr key={zone.id}><td>{zone.rank}</td><th scope="row"><button className="table-zone-link" onClick={() => select(zone.id)}>{zone.label}</button><small>{zone.position.latitude.toFixed(2)}°N, {zone.position.longitude.toFixed(2)}°E</small></th><td>{(zone.probability * 100).toFixed(1)}%</td><td>{zone.uncertainty === null ? 'Not available' : `${(zone.uncertainty * 100).toFixed(0)}%`}</td><td>{zone.radiusKm} km</td><td>{zone.daysSinceLastEvent} days</td><td>{zone.populationEstimate?.toLocaleString() ?? 'Not available'}</td><td>{zone.populationDensityPerKm2 == null ? 'Not available' : `${Math.round(zone.populationDensityPerKm2).toLocaleString()} / km²`}</td></tr>)}</tbody></table></div>
      <p className="table-note">Candidate probabilities are model allocation across a finite support set. They are not calibrated event-occurrence probabilities for every place in Ethiopia and should not be summed as “chance of conflict.”</p>
    </section>
  )
}
