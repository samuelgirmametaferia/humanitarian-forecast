export type ModelCatalogEntry = {
  id: string
  codeName: string
  family: 'Location' | 'Risk'
  status: 'live' | 'candidate' | 'reference' | 'retired'
}

export const MODEL_CATALOG: ModelCatalogEntry[] = [
  { id: 'location/candidate_ranker/v10', codeName: 'IRONWOOD', family: 'Location', status: 'live' },
  { id: 'location/candidate_ranker/v10_prized', codeName: 'CITADEL', family: 'Location', status: 'live' },
  { id: 'location/candidate_ranker/v9', codeName: 'MERIDIAN', family: 'Location', status: 'reference' },
  { id: 'location/candidate_ranker_ethiopia/v10', codeName: 'SABLE', family: 'Location', status: 'live' },
  { id: 'location/candidate_ranker_ethiopia/v10_prized_adapted', codeName: 'LANTERN', family: 'Location', status: 'live' },
  { id: 'location/geostate/ethiopia_pretrained_m_v1', codeName: 'STRATA', family: 'Location', status: 'retired' },
  { id: 'location/geostate/ethiopia_v1', codeName: 'BEDROCK', family: 'Location', status: 'retired' },
  { id: 'location/h3_lambdarank/ethiopia_r4_global_transfer_v1', codeName: 'TIDE', family: 'Location', status: 'candidate' },
  { id: 'location/h3_lambdarank/ethiopia_r4_v1', codeName: 'DRIFT', family: 'Location', status: 'retired' },
  { id: 'location/h3_lambdarank_memory/ethiopia_r4_v1', codeName: 'MEMORY', family: 'Location', status: 'candidate' },
  { id: 'location/marked_hawkes/ethiopia_v1', codeName: 'EMBER', family: 'Location', status: 'candidate' },
  { id: 'location/reliefweb_spatial_specialist/content_v1', codeName: 'SIGNAL', family: 'Location', status: 'candidate' },
  { id: 'location/reliefweb_spatial_specialist/count_v1', codeName: 'PULSE', family: 'Location', status: 'candidate' },
  { id: 'location/reliefweb_spatial_specialist/full_v1', codeName: 'CHORUS', family: 'Location', status: 'candidate' },
  { id: 'location/shape_analogue/ethiopia_v1', codeName: 'MIRROR', family: 'Location', status: 'candidate' },
  { id: 'location/swarm/v1', codeName: 'FLOCK', family: 'Location', status: 'candidate' },
  { id: 'location/theswarm/fine_v1', codeName: 'KESTREL', family: 'Location', status: 'reference' },
  { id: 'location/theswarm/fine_v2', codeName: 'NIGHTJAR', family: 'Location', status: 'live' },
  { id: 'location/theswarm/v1', codeName: 'SWARM', family: 'Location', status: 'live' },
  { id: 'risk/base/v1', codeName: 'ORIGIN', family: 'Risk', status: 'reference' },
  { id: 'risk/base/v2', codeName: 'HORIZON', family: 'Risk', status: 'reference' },
  { id: 'risk/ethiopia/v1', codeName: 'ROOT', family: 'Risk', status: 'reference' },
  { id: 'risk/ethiopia/v2', codeName: 'SEED', family: 'Risk', status: 'reference' },
  { id: 'risk/ethiopia/v3', codeName: 'BRANCH', family: 'Risk', status: 'reference' },
  { id: 'risk/ethiopia/v4', codeName: 'CANOPY', family: 'Risk', status: 'reference' },
  { id: 'risk/ethiopia/v5', codeName: 'FORGE', family: 'Risk', status: 'candidate' },
  { id: 'risk/ethiopia/v6', codeName: 'RIDGELINE', family: 'Risk', status: 'candidate' },
  { id: 'risk/ethiopia/v7_accessibility', codeName: 'SWITCHBACK', family: 'Risk', status: 'retired' },
  { id: 'risk/ethiopia/v7_cold_start_transfer', codeName: 'NORTHSTAR', family: 'Risk', status: 'candidate' },
  { id: 'risk/ethiopia/v7_context_combined', codeName: 'CONFLUENCE', family: 'Risk', status: 'candidate' },
  { id: 'risk/ethiopia/v7_context_fusion', codeName: 'FUSE', family: 'Risk', status: 'retired' },
  { id: 'risk/ethiopia/v7_population', codeName: 'DENSITY', family: 'Risk', status: 'retired' },
  { id: 'risk/ethiopia/v7_transfer', codeName: 'TRANSFER', family: 'Risk', status: 'retired' },
]

// The live registry model: whatever version the serving API is currently
// running (the weekly retrain promotes automatically). Selected by default.
export const LIVE_MODEL_ID = 'live'

export const DEFAULT_MODEL_ID = 'location/theswarm/fine_v2'

export const PUBLISHED_MODEL_IDS = new Set([
  DEFAULT_MODEL_ID,
  'location/theswarm/v1',
  'location/candidate_ranker/v10_prized',
  'location/candidate_ranker/v9',
])
