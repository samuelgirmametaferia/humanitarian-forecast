import type { ForecastSnapshot } from '../../contracts/forecast'
import { useForecastStore } from '../../lib/store'

export function FallbackMap({ data, globeAvailable = true }: { data: ForecastSnapshot; globeAvailable?: boolean }) {
  const selected = useForecastStore((state) => state.selectedZoneId)
  const select = useForecastStore((state) => state.setSelectedZone)
  const setMapMode = useForecastStore((state) => state.setMapMode)
  const showLabels = useForecastStore((state) => state.showLabels)
  const reduced = useForecastStore((state) => state.visualDensity === 'reduced')
  const project = (lon: number, lat: number) => ({ x: (lon - 32.5) / 16 * 680 + 20, y: 520 - (lat - 3) / 12 * 480 })
  const zones = reduced ? data.zones.slice(0, 5) : data.zones

  return <div className="fallback-map" role="region" aria-label="Accessible regional forecast map">
    <MapSwitch mode="accessible" globeAvailable={globeAvailable} onGlobe={() => setMapMode('globe')} onAccessible={() => setMapMode('accessible')} />
    <svg viewBox="0 0 720 560" aria-label="Selectable forecast zones">
      <defs><pattern id="uncertainty-hatch" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(35)"><line x1="0" y1="0" x2="0" y2="8" /></pattern></defs>
      <path className="fallback-land" d="M278 15L510 60 665 195 620 390 430 535 210 488 42 328 85 155Z" />
      <path className="fallback-ethiopia" d="M270 120L458 105 582 220 515 350 342 408 185 314 188 196Z" />
      {zones.map((zone) => { const point = project(zone.position.longitude, zone.position.latitude); return <g key={zone.id} className={selected === zone.id ? 'is-selected' : ''} role="button" tabIndex={0} aria-label={`Select ${zone.label}, rank ${zone.rank}`} onClick={() => select(zone.id)} onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); select(zone.id) } }}>
        {zone.uncertainty !== null && <circle className="fallback-uncertainty" cx={point.x} cy={point.y} r={28 + zone.uncertainty * 18} />}
        <circle className="fallback-zone" cx={point.x} cy={point.y} r={12 + zone.probability * 45} />
        {showLabels && <text x={point.x} y={point.y + 4}>{zone.rank}</text>}
      </g> })}
    </svg>
    {!globeAvailable && <p className="fallback-caption">3D rendering is unavailable in this browser.</p>}
  </div>
}

export function MapSwitch({ mode, globeAvailable, onGlobe, onAccessible }: { mode: 'globe' | 'accessible'; globeAvailable: boolean; onGlobe: () => void; onAccessible: () => void }) {
  return <div className="map-switch" aria-label="Map view"><button type="button" aria-pressed={mode === 'globe'} disabled={!globeAvailable} onClick={onGlobe}>3D Globe</button><button type="button" aria-pressed={mode === 'accessible'} onClick={onAccessible}>Accessible Map</button></div>
}
