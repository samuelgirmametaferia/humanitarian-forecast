import type { ForecastSnapshot, ForecastZone } from '../contracts/forecast'
import { DEFAULT_MODEL_ID, MODEL_CATALOG, PUBLISHED_MODEL_IDS } from './model-catalog'

type ActualZone = [latitude: number, longitude: number, probability: number, radiusKm?: number, elevationM?: number, ruggednessM?: number, recency?: number]

const outputs: Record<string, ActualZone[]> = {
  [DEFAULT_MODEL_ID]: [
    [10.321, 38.2312, .2372, 20, 2607, 147.2, 1],
    [10.8417, 37.161, .0425, 20, 1910, 34.8, 7],
    [11.5, 38.5, .0321, 20, 1634, 118.8, 1],
    [11.5936, 37.3908, .0299, 20, 1930, 15.3, 52],
    [10.45, 37.5667, .0249, 20, 2665, 82.9, 7],
  ],
  'location/theswarm/v1': [
    [11.4992, 38.6031, .721331, 25],
    [10.4157, 37.5659, .277853, 25],
    [10.968, 37.1899, .000341, 25],
    [10.4861, 38.2732, .000247, 25],
    [10.5754, 37.2041, .000227, 25],
  ],
  'location/candidate_ranker/v10_prized': [
    [10.75, 37.75, 1, 25],
  ],
  'location/candidate_ranker/v9': [
    [10.84175, 37.16101, .176917, 20],
    [10.32104, 38.23116, .103014, 20],
    [10.55, 37.48333, .079742, 20],
    [10.45, 37.56667, .059756, 20],
    [10.7, 37.26667, .059634, 20],
  ],
}

function zones(rows: ActualZone[], codeName: string): ForecastZone[] {
  return rows.map(([latitude, longitude, probability, radiusKm = 20, elevationM = 0, ruggednessM = 0, recency = 0], index) => ({
    id: `${codeName.toLowerCase()}-${index + 1}`,
    rank: index + 1,
    label: `${latitude.toFixed(2)}°N · ${longitude.toFixed(2)}°E`,
    position: { latitude, longitude },
    probability,
    uncertainty: null,
    radiusKm,
    siteType: 'conflict-frequency site',
    daysSinceLastEvent: recency,
    elevationM,
    ruggednessM,
    populationExposureBand: null,
    populationEstimate: null,
    populationDensityPerKm2: null,
    populationAreaKm2: null,
    populationYear: null,
    populationSource: null,
    adminArea: null,
    drivers: [`${codeName} archived-input inference`, 'same Ethiopia feature row', 'model-specific spatial ranking'],
  }))
}

export function forecastForModel(base: ForecastSnapshot, selectedId: string): ForecastSnapshot {
  const modelId = PUBLISHED_MODEL_IDS.has(selectedId) ? selectedId : DEFAULT_MODEL_ID
  const entry = MODEL_CATALOG.find((item) => item.id === modelId)!
  return {
    ...base,
    id: `scenario:${entry.codeName.toLowerCase()}:2025-12-28`,
    conflict: 'Ethiopia: Government/Amhara',
    observationCutoff: '2025-12-28T23:59:59Z',
    horizonDays: 1,
    model: { ...base.model, name: entry.codeName, version: modelId.split('/').at(-1) ?? base.model.version },
    zones: zones(outputs[modelId]!, entry.codeName),
    warning: `Model-specific ${entry.codeName} output generated from the same archived Ethiopia feature row. Research signal, not a current incident report, tactical forecast, safe route, or evacuation order.`,
  }
}
