import type { LayerId } from '../../contracts/forecast'
import { useForecastStore } from '../../lib/store'

const labels: Record<LayerId, string> = {
  probability: 'Forecast zones',
  uncertainty: 'Uncertainty',
  observations: 'Observed events',
  signals: 'Source signals',
  terrain: 'Terrain',
  exposure: 'Population heatmap',
  populationDots: 'Population dots',
  motion: 'Signal motion',
  places: 'Cities & forecast sites',
  administrative: 'Boundaries',
}

export function LayerControls() {
  const layers = useForecastStore((state) => state.layers)
  const toggle = useForecastStore((state) => state.toggleLayer)
  const layerIds = Object.keys(labels) as LayerId[]
  return <section className="layer-panel" aria-labelledby="layer-title" data-tour="layers">
    <div className="panel-heading"><h2 id="layer-title">Layers</h2></div>
    <div className="layer-list">{layerIds.map((id) => <label key={id} className={`layer-control layer-${id}`}>
      <input type="checkbox" checked={layers[id]} onChange={() => toggle(id)} /><span className="layer-symbol" aria-hidden="true" /><strong>{labels[id]}</strong>
    </label>)}</div>
    {(layers.exposure || layers.populationDots) && <div className="population-key" aria-label="Population map legend"><span>Population density</span><div className="population-ramp" aria-hidden="true" /><small>lower density</small><small>higher density</small><p>{layers.populationDots ? 'Bright dot clusters scale with city population; WorldPop zone dots appear when density estimates are published.' : 'The heatmap shows relative population concentration.'}</p></div>}
  </section>
}
