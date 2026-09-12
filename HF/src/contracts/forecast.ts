import { z } from 'zod'

const positionSchema = z.object({
  latitude: z.number().min(-90).max(90),
  longitude: z.number().min(-180).max(180),
}).strict()

export const forecastSnapshotSchema = z.object({
  schemaVersion: z.literal('forecast-snapshot.v1'),
  id: z.string().min(8).max(80),
  mode: z.enum(['demo', 'production']),
  country: z.literal('Ethiopia'),
  conflict: z.string().min(1).max(100),
  generatedAt: z.iso.datetime(),
  observationCutoff: z.iso.datetime(),
  horizonDays: z.number().int().min(1).max(30),
  model: z.object({
    name: z.string().max(120),
    version: z.string().max(40),
    artifactSha256: z.string(),
    featureContract: z.literal('feature-contract.v1'),
    candidateCount: z.number().int().positive().max(512),
    validationTop1Within20Km: z.number().min(0).max(1),
    validationDiverseTop5Within20Km: z.number().min(0).max(1),
    developmentTop1Within20Km: z.number().min(0).max(1),
    candidateOracleWithin20Km: z.number().min(0).max(1),
  }).strict(),
  zones: z.array(z.object({
    id: z.string(),
    rank: z.number().int().min(1).max(10),
    label: z.string().min(1).max(80),
    position: positionSchema,
    probability: z.number().min(0).max(1),
    uncertainty: z.number().min(0).max(1).nullable(),
    radiusKm: z.number().min(10).max(100),
    siteType: z.enum(['conflict-frequency site', 'recent cross-conflict event']),
    daysSinceLastEvent: z.number().int().nonnegative(),
    elevationM: z.number().int(),
    ruggednessM: z.number().nonnegative(),
    populationExposureBand: z.enum(['lower', 'moderate', 'higher']).nullable(),
    populationEstimate: z.number().int().nonnegative().nullable().optional(),
    populationDensityPerKm2: z.number().nonnegative().nullable().optional(),
    populationAreaKm2: z.number().positive().nullable().optional(),
    populationYear: z.number().int().min(2015).max(2030).nullable().optional(),
    populationSource: z.string().max(120).nullable().optional(),
    adminArea: z.string().max(120).nullable().optional(),
    drivers: z.array(z.string()).min(1).max(5),
  }).strict()).min(1).max(10),
  observations: z.array(z.object({
    id: z.string(),
    kind: z.enum(['observed event', 'source signal']),
    position: positionSchema,
    occurredAt: z.iso.datetime(),
    precision: z.enum(['exact', 'named place', 'regional']),
    summary: z.string().min(1).max(240),
  }).strict()).max(500),
  sourceHealth: z.array(z.object({
    id: z.string(),
    label: z.string(),
    status: z.enum(['healthy', 'delayed', 'unavailable']),
    lastSuccessfulAt: z.iso.datetime().nullable(),
    articlesProcessed: z.number().int().nonnegative(),
  }).strict()).min(1).max(12),
  warning: z.string().min(20).max(500),
}).strict()

export type ForecastSnapshot = z.infer<typeof forecastSnapshotSchema>
export type ForecastZone = ForecastSnapshot['zones'][number]
export type ForecastObservation = ForecastSnapshot['observations'][number]
export const healthSchema = z.object({
  status: z.enum(['ok', 'degraded']),
  mode: z.enum(['demo', 'production']),
  writesEnabled: z.boolean(),
}).strict()

export const historyEventSchema = z.object({
  id: z.string(),
  kind: z.enum(['dataset_update', 'model_run', 'training', 'evaluation', 'prediction_batch', 'model_version']),
  occurredAt: z.iso.datetime(),
  mode: z.enum(['demo', 'production']),
  modelName: z.string().nullable(),
  modelVersion: z.string().nullable(),
  sourceSnapshotId: z.string().nullable(),
  metrics: z.record(z.string(), z.number()),
  changes: z.array(z.string()),
  reportUrl: z.string().nullable(),
}).strict()

export const projectHistorySchema = z.object({
  entries: z.array(historyEventSchema),
  updateCycleDays: z.literal(3),
}).strict()

export type Health = z.infer<typeof healthSchema>
export type ProjectHistory = z.infer<typeof projectHistorySchema>
export type ProjectHistoryEvent = z.infer<typeof historyEventSchema>
export type LayerId = 'probability' | 'uncertainty' | 'observations' | 'signals' | 'terrain' | 'exposure' | 'populationDots' | 'motion' | 'places' | 'administrative'
