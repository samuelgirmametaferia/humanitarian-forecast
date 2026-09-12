import type { ForecastSnapshot, ForecastZone } from '../../contracts/forecast'

export type GeoJsonFeature = {
  type: 'Feature'
  id?: string
  properties: Record<string, string | number | boolean>
  geometry: { type: 'Point'; coordinates: [number, number] } | { type: 'Polygon'; coordinates: [number, number][][] } | { type: 'LineString'; coordinates: [number, number][] }
}

export type GeoJsonCollection = { type: 'FeatureCollection'; features: GeoJsonFeature[] }

const ADVISORY_RADIUS_KM = 20
const EARTH_RADIUS_KM = 6371.0088

function circle(zone: ForecastZone, points = 96, radiusKm = Math.min(zone.radiusKm, ADVISORY_RADIUS_KM)): GeoJsonFeature {
  const latitude = zone.position.latitude * Math.PI / 180
  const longitude = zone.position.longitude * Math.PI / 180
  const angularRadius = radiusKm / EARTH_RADIUS_KM
  const ring: [number, number][] = Array.from({ length: points + 1 }, (_, index) => {
    const bearing = index / points * Math.PI * 2
    const pointLatitude = Math.asin(
      Math.sin(latitude) * Math.cos(angularRadius)
      + Math.cos(latitude) * Math.sin(angularRadius) * Math.cos(bearing),
    )
    const pointLongitude = longitude + Math.atan2(
      Math.sin(bearing) * Math.sin(angularRadius) * Math.cos(latitude),
      Math.cos(angularRadius) - Math.sin(latitude) * Math.sin(pointLatitude),
    )
    return [pointLongitude * 180 / Math.PI, pointLatitude * 180 / Math.PI]
  })
  return {
    type: 'Feature', id: zone.id,
    properties: {
      id: zone.id, rank: zone.rank, label: zone.label, probability: zone.probability,
      uncertainty: zone.uncertainty ?? -1, height: 6000 + zone.probability * 80000,
    },
    geometry: { type: 'Polygon', coordinates: [ring] },
  }
}

export function exposureGeoJson(data: ForecastSnapshot): GeoJsonCollection {
  const exposureValue = { lower: 1, moderate: 2, higher: 3 } as const
  return {
    type: 'FeatureCollection',
    features: data.zones.map((zone) => {
      const feature = circle(zone, 40)
      feature.properties.exposure = zone.populationExposureBand ? exposureValue[zone.populationExposureBand] : 0
      feature.properties.density = zone.populationDensityPerKm2 ?? 0
      feature.properties.population = zone.populationEstimate ?? 0
      return feature
    }),
  }
}

export function forecastGeoJson(data: ForecastSnapshot): GeoJsonCollection {
  return { type: 'FeatureCollection', features: data.zones.map((zone) => circle(zone)) }
}

export function forecastGaussianGeoJson(data: ForecastSnapshot): GeoJsonCollection {
  const bands = 18
  return {
    type: 'FeatureCollection',
    features: data.zones.flatMap((zone) => Array.from({ length: bands }, (_, index) => {
      const radialFraction = 1 - index / bands * .92
      const gaussianWeight = Math.exp(-.5 * (radialFraction / .36) ** 2)
      const feature = circle(zone, 64, Math.min(zone.radiusKm, ADVISORY_RADIUS_KM) * radialFraction)
      feature.id = `${zone.id}:gaussian:${index}`
      feature.properties.zoneId = zone.id
      feature.properties.gaussianWeight = gaussianWeight
      feature.properties.localLikelihood = zone.probability * gaussianWeight
      feature.properties.height = 500 + zone.probability * gaussianWeight * 240000
      return feature
    })),
  }
}

export function zonePointsGeoJson(data: ForecastSnapshot): GeoJsonCollection {
  return {
    type: 'FeatureCollection',
    features: data.zones.map((zone) => ({
      type: 'Feature', id: zone.id,
      properties: {
        id: zone.id, label: zone.label, rank: zone.rank,
        population: zone.populationEstimate ?? 0,
        density: zone.populationDensityPerKm2 ?? 0,
      },
      geometry: { type: 'Point', coordinates: [zone.position.longitude, zone.position.latitude] },
    })),
  }
}

export function populationDensityDotsGeoJson(data: ForecastSnapshot): GeoJsonCollection {
  return {
    type: 'FeatureCollection',
    features: data.zones.flatMap((zone) => {
      const density = zone.populationDensityPerKm2 ?? 0
      if (density <= 0) return []
      const dotCount = Math.max(4, Math.min(42, Math.round(Math.sqrt(density) * 1.45)))
      const latitudeScale = Math.cos(zone.position.latitude * Math.PI / 180)
      return Array.from({ length: dotCount }, (_, index) => {
        // A deterministic sunflower layout reads as density without implying
        // household-level precision.
        const fraction = Math.sqrt((index + .5) / dotCount)
        const angle = index * 2.399963229728653
        const radiusKm = Math.min(zone.radiusKm, ADVISORY_RADIUS_KM) * .82 * fraction
        const latitude = zone.position.latitude + Math.sin(angle) * radiusKm / 110.574
        const longitude = zone.position.longitude + Math.cos(angle) * radiusKm / (111.32 * latitudeScale)
        return {
          type: 'Feature' as const,
          id: `${zone.id}:population:${index}`,
          properties: {
            zoneId: zone.id,
            label: zone.label,
            density,
            population: zone.populationEstimate ?? 0,
          },
          geometry: { type: 'Point' as const, coordinates: [longitude, latitude] as [number, number] },
        }
      })
    }),
  }
}

export function observationsGeoJson(data: ForecastSnapshot): GeoJsonCollection {
  return {
    type: 'FeatureCollection',
    features: data.observations.map((item) => ({
      type: 'Feature', id: item.id,
      properties: { id: item.id, kind: item.kind, summary: item.summary, precision: item.precision },
      geometry: { type: 'Point', coordinates: [item.position.longitude, item.position.latitude] },
    })),
  }
}

export function motionGeoJson(data: ForecastSnapshot): GeoJsonCollection {
  return {
    type: 'FeatureCollection',
    features: data.observations.map((observation) => {
      const nearest = data.zones.reduce((best, zone) => {
        const distance = (zone.position.latitude - observation.position.latitude) ** 2 + (zone.position.longitude - observation.position.longitude) ** 2
        const bestDistance = (best.position.latitude - observation.position.latitude) ** 2 + (best.position.longitude - observation.position.longitude) ** 2
        return distance < bestDistance ? zone : best
      }, data.zones[0]!)
      return {
        type: 'Feature', id: `motion:${observation.id}`,
        properties: { id: observation.id, basis: 'public signal direction; not tracked movement' },
        geometry: { type: 'LineString', coordinates: [
          [observation.position.longitude, observation.position.latitude],
          [nearest.position.longitude, nearest.position.latitude],
        ] },
      }
    }),
  }
}
