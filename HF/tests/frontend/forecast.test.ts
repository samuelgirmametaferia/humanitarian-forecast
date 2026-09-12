import demoForecast from '../../src/data/demo-forecast.json'
import invalidForecast from '../fixtures/forecast.invalid.json'
import { describe, expect, it } from 'vitest'
import { forecastSnapshotSchema, healthSchema, projectHistorySchema } from '../../src/contracts/forecast'

describe('forecast contract', () => {
  it('accepts the bundled demo snapshot', () => {
    expect(forecastSnapshotSchema.parse(demoForecast).country).toBe('Ethiopia')
  })

  it('rejects unexpected and incomplete fields', () => {
    expect(() => forecastSnapshotSchema.parse(invalidForecast)).toThrow()
  })

  it('validates health and an empty lifecycle history', () => {
    expect(healthSchema.parse({ status: 'ok', mode: 'demo', writesEnabled: false }).status).toBe('ok')
    expect(projectHistorySchema.parse({ entries: [], updateCycleDays: 3 }).entries).toEqual([])
  })

  it('rejects invented lifecycle event fields', () => {
    expect(() => projectHistorySchema.parse({ entries: [{ id: 'event-1', kind: 'training', occurredAt: '2026-09-12T00:00:00Z', mode: 'production', modelName: null, modelVersion: null, sourceSnapshotId: null, metrics: {}, changes: [], reportUrl: null, future: true }], updateCycleDays: 3 })).toThrow()
  })
})
