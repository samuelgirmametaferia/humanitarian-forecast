import modelForecast from '../data/model-forecast.json'
import {
  forecastSnapshotSchema,
  healthSchema,
  projectHistorySchema,
  type ForecastSnapshot,
  type Health,
  type ProjectHistory,
} from '../contracts/forecast'

const apiBase = import.meta.env.VITE_API_BASE_URL || '/api'
const providerMode = import.meta.env.VITE_PROVIDER_MODE || 'production'

async function request(path: string, signal?: AbortSignal) {
  const response = await fetch(`${apiBase}${path}`, { signal, headers: { Accept: 'application/json' } })
  if (!response.ok) throw new Error(`Forecast service returned ${response.status}`)
  return response.json() as Promise<unknown>
}

export async function getLatestForecast(signal?: AbortSignal): Promise<ForecastSnapshot> {
  if (providerMode === 'model-scenario') return forecastSnapshotSchema.parse(modelForecast)
  return forecastSnapshotSchema.parse(await request('/v1/forecast/latest', signal))
}

export async function getForecastHistory(signal?: AbortSignal): Promise<ForecastSnapshot[]> {
  if (providerMode === 'model-scenario') return [forecastSnapshotSchema.parse(modelForecast)]
  return forecastSnapshotSchema.array().parse(await request('/v1/forecast/history', signal))
}

export async function getHealth(signal?: AbortSignal): Promise<Health> {
  if (providerMode === 'model-scenario') return healthSchema.parse({ status: 'degraded', mode: 'demo', writesEnabled: false })
  return healthSchema.parse(await request('/v1/health', signal))
}

export async function getProjectHistory(signal?: AbortSignal): Promise<ProjectHistory> {
  if (providerMode === 'model-scenario') return projectHistorySchema.parse({ entries: [], updateCycleDays: 3 })
  return projectHistorySchema.parse(await request('/v1/history', signal))
}

export function isDemoMode() {
  return providerMode === 'model-scenario'
}
