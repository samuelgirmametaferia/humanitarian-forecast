import { useCallback, useEffect, useRef, useState } from 'react'
import { LocateFixed } from 'lucide-react'
import * as maplibregl from 'maplibre-gl'
import type { Map as MapLibreMap, MapLayerMouseEvent } from 'maplibre-gl'
import type { ForecastSnapshot } from '../../contracts/forecast'
import { useForecastStore } from '../../lib/store'
import { FallbackMap, MapSwitch } from './FallbackMap'
import { exposureGeoJson, forecastGaussianGeoJson, forecastGeoJson, motionGeoJson, observationsGeoJson, zonePointsGeoJson } from './geo'
import 'maplibre-gl/dist/maplibre-gl.css'

const TERRAIN_SOURCE = 'elevation-dem'
const ETHIOPIA_VIEW = { center: [39.1, 9.6] as [number, number], zoom: 5.05, pitch: 58, bearing: -12 }

const baseStyle: maplibregl.StyleSpecification = {
  version: 8,
  sources: {
    countries: { type: 'geojson', data: '/horn-context.geojson' },
    regions: { type: 'geojson', data: '/ethiopia-regions.geojson' },
    [TERRAIN_SOURCE]: {
      type: 'raster-dem',
      url: 'https://tiles.mapterhorn.com/tilejson.json',
      encoding: 'terrarium',
      tileSize: 512,
      attribution: 'Elevation © Mapterhorn',
    },
  },
  layers: [
    { id: 'background', type: 'background', paint: { 'background-color': '#071820' } },
    {
      id: 'countries-fill', type: 'fill', source: 'countries',
      paint: {
        'fill-color': ['case', ['==', ['get', 'shapeGroup'], 'ETH'], '#496f45', '#233b3b'],
        'fill-opacity': ['case', ['==', ['get', 'shapeGroup'], 'ETH'], .78, .58],
      },
    },
    { id: 'regional-fill', type: 'fill', source: 'regions', paint: {
      'fill-color': ['match', ['get', 'shapeISO'],
        'ET-AA', '#4f7650', 'ET-AF', '#8a7849', 'ET-AM', '#547950', 'ET-BE', '#34665d',
        'ET-DD', '#637b55', 'ET-GA', '#3c6b61', 'ET-HA', '#78764a', 'ET-OR', '#3f7158',
        'ET-SN', '#697747', 'ET-SO', '#776f48', 'ET-TI', '#566f50', '#476b54'],
      'fill-opacity': .54,
    } },
    { id: 'terrain-hillshade', type: 'hillshade', source: TERRAIN_SOURCE, paint: {
      'hillshade-shadow-color': '#071820', 'hillshade-highlight-color': '#e9d9a7',
      'hillshade-accent-color': '#8d6c42', 'hillshade-exaggeration': .48,
    } },
    { id: 'regional-line', type: 'line', source: 'regions', paint: { 'line-color': '#a9bd8b', 'line-opacity': .5, 'line-width': 1 } },
    { id: 'countries-line', type: 'line', source: 'countries', paint: {
      'line-color': ['case', ['==', ['get', 'shapeGroup'], 'ETH'], '#f0d697', '#718987'],
      'line-opacity': ['case', ['==', ['get', 'shapeGroup'], 'ETH'], .95, .52],
      'line-width': ['case', ['==', ['get', 'shapeGroup'], 'ETH'], 2.4, 1],
    } },
  ],
  sky: { 'sky-color': '#071820', 'horizon-color': '#28494b', 'fog-color': '#101d1b', 'sky-horizon-blend': 0.72, 'horizon-fog-blend': 0.76, 'fog-ground-blend': 0.5 },
}

function supportsWebGl() {
  try {
    const canvas = document.createElement('canvas')
    return Boolean(canvas.getContext('webgl2') || canvas.getContext('webgl'))
  } catch {
    return false
  }
}

export function ForecastMap({ data }: { data: ForecastSnapshot }) {
  const container = useRef<HTMLDivElement>(null)
  const mapRef = useRef<MapLibreMap | null>(null)
  const hoverId = useRef<string | null>(null)
  const [globeAvailable, setGlobeAvailable] = useState(supportsWebGl)
  const layers = useForecastStore((state) => state.layers)
  const mapMode = useForecastStore((state) => state.mapMode)
  const setMapMode = useForecastStore((state) => state.setMapMode)
  const reducedMotion = useForecastStore((state) => state.reducedMotion)
  const visualDensity = useForecastStore((state) => state.visualDensity)
  const showLabels = useForecastStore((state) => state.showLabels)
  const selectedZoneId = useForecastStore((state) => state.selectedZoneId)
  const selectZone = useForecastStore((state) => state.setSelectedZone)

  const focusEthiopia = useCallback((animate = true) => {
    const map = mapRef.current
    if (!map) return
    map.resize()
    if (animate && !reducedMotion) {
      map.flyTo({ ...ETHIOPIA_VIEW, duration: 2200, curve: 1.25, essential: true })
    } else {
      map.jumpTo(ETHIOPIA_VIEW)
    }
  }, [reducedMotion])

  useEffect(() => {
    if (!container.current || mapMode !== 'globe' || !globeAvailable || mapRef.current) return
    const map = new maplibregl.Map({
      container: container.current,
      style: import.meta.env.VITE_MAP_STYLE_URL || baseStyle,
      center: reducedMotion ? ETHIOPIA_VIEW.center : [18, 5],
      zoom: reducedMotion ? ETHIOPIA_VIEW.zoom : 1.15,
      pitch: reducedMotion ? ETHIOPIA_VIEW.pitch : 0,
      bearing: reducedMotion ? ETHIOPIA_VIEW.bearing : 0,
      attributionControl: false,
    })
    mapRef.current = map
    map.on('style.load', () => {
      map.setProjection({ type: 'globe' })
      if (map.getSource(TERRAIN_SOURCE)) map.setTerrain({ source: TERRAIN_SOURCE, exaggeration: 1.35 })
      map.addSource('forecast-zones', { type: 'geojson', data: forecastGeoJson(data), promoteId: 'id' })
      map.addSource('gaussian-zones', { type: 'geojson', data: forecastGaussianGeoJson(data) })
      map.addSource('zone-points', { type: 'geojson', data: zonePointsGeoJson(data), promoteId: 'id' })
      map.addSource('observations', { type: 'geojson', data: observationsGeoJson(data), cluster: true, clusterRadius: 30, clusterMaxZoom: 9 })
      map.addSource('motion', { type: 'geojson', data: motionGeoJson(data), lineMetrics: true })
      map.addSource('exposure', { type: 'geojson', data: exposureGeoJson(data) })
      map.addSource('cities', { type: 'geojson', data: '/ethiopia-cities.geojson', attribution: 'Cities © OpenStreetMap contributors, ODbL' })
      map.addLayer({
        id: 'exposure-bands', type: 'fill', source: 'exposure',
        paint: {
          'fill-color': ['interpolate', ['linear'], ['get', 'density'], 0, '#132d35', 100, '#276d6e', 300, '#d4a848', 800, '#f36f3f'],
          'fill-opacity': .62,
          'fill-outline-color': 'rgba(255,255,255,.42)',
        },
        layout: { visibility: useForecastStore.getState().layers.exposure ? 'visible' : 'none' },
      })
      map.addLayer({
        id: 'city-population-heat', type: 'heatmap', source: 'cities', maxzoom: 8,
        paint: {
          'heatmap-weight': ['interpolate', ['linear'], ['coalesce', ['get', 'population'], 15000], 0, .08, 250000, .7, 3500000, 1],
          'heatmap-intensity': ['interpolate', ['linear'], ['zoom'], 3, .45, 7, 1.1],
          'heatmap-radius': ['interpolate', ['linear'], ['zoom'], 3, 12, 7, 34],
          'heatmap-opacity': .46,
          'heatmap-color': ['interpolate', ['linear'], ['heatmap-density'], 0, 'rgba(11,30,34,0)', .25, 'rgba(46,147,133,.34)', .55, 'rgba(232,196,84,.62)', 1, 'rgba(255,105,64,.9)'],
        },
      })
      map.addLayer({
        id: 'probability-relief', type: 'fill-extrusion', source: 'gaussian-zones',
        paint: {
          'fill-extrusion-color': ['interpolate', ['linear'], ['get', 'localLikelihood'], 0, '#1a5661', 0.025, '#2bbf9b', 0.08, '#d7c34b', 0.18, '#ff7043'],
          'fill-extrusion-height': ['get', 'height'],
          'fill-extrusion-base': 250,
          'fill-extrusion-opacity': 0.82,
          'fill-extrusion-vertical-gradient': true,
        },
      })
      map.addLayer({
        id: 'probability-contours', type: 'line', source: 'gaussian-zones',
        paint: {
          'line-color': ['interpolate', ['linear'], ['get', 'localLikelihood'], 0, '#48a9b4', .04, '#51d4a8', .1, '#ead35c', .18, '#ff7043'],
          'line-width': ['interpolate', ['linear'], ['zoom'], 4, .45, 8, 1.35],
          'line-opacity': .68,
        },
      })
      map.addLayer({
        id: 'zone-interaction-outline', type: 'line', source: 'forecast-zones',
        paint: {
          'line-color': '#ffffff',
          'line-width': ['case', ['boolean', ['feature-state', 'selected'], false], 3, ['boolean', ['feature-state', 'hover'], false], 2, 0],
          'line-opacity': ['case', ['boolean', ['feature-state', 'selected'], false], 1, ['boolean', ['feature-state', 'hover'], false], .72, 0],
          'line-width-transition': { duration: reducedMotion ? 0 : 140 },
          'line-opacity-transition': { duration: reducedMotion ? 0 : 140 },
        },
      })
      map.addLayer({
        id: 'zone-hit-targets', type: 'circle', source: 'zone-points',
        paint: { 'circle-radius': 34, 'circle-color': 'rgba(0,0,0,0)', 'circle-stroke-width': 0 },
      })
      map.addLayer({
        id: 'zone-labels', type: 'symbol', source: 'zone-points', minzoom: 4.7,
        layout: {
          'text-field': ['concat', ['to-string', ['get', 'rank']], '  ', ['get', 'label']],
          'text-size': 12, 'text-offset': [0, 2.1], 'text-anchor': 'top',
          'text-allow-overlap': false, visibility: useForecastStore.getState().showLabels ? 'visible' : 'none',
        },
        paint: { 'text-color': '#f4f1e7', 'text-halo-color': '#0b1715', 'text-halo-width': 1.5 },
      })
      map.addLayer({
        id: 'uncertainty-outline', type: 'line', source: 'forecast-zones',
        paint: {
          'line-color': '#c98500',
          'line-width': ['case', ['<', ['get', 'uncertainty'], 0], 0, ['interpolate', ['linear'], ['get', 'uncertainty'], 0, 1, 1, 5]],
          'line-opacity': ['case', ['<', ['get', 'uncertainty'], 0], 0, ['interpolate', ['linear'], ['get', 'uncertainty'], 0, .2, 1, .9]],
          'line-dasharray': [1.2, 1.4],
        },
      })
      map.addLayer({
        id: 'source-signals', type: 'circle', source: 'observations',
        filter: ['==', ['get', 'kind'], 'source signal'],
        paint: { 'circle-radius': 8, 'circle-color': 'rgba(201,133,0,.15)', 'circle-stroke-color': '#c98500', 'circle-stroke-width': 2 },
      })
      map.addLayer({
        id: 'motion-traces', type: 'line', source: 'motion',
        paint: {
          'line-width': ['interpolate', ['linear'], ['zoom'], 4, 1, 8, 3],
          'line-opacity': .72,
          'line-gradient': ['interpolate', ['linear'], ['line-progress'], 0, 'rgba(233,217,167,0)', .55, '#e9d9a7', 1, '#ff7043'],
          'line-dasharray': [2, 2],
        },
      })
      map.addLayer({
        id: 'observed-events', type: 'circle', source: 'observations',
        filter: ['==', ['get', 'kind'], 'observed event'],
        paint: {
          'circle-radius': 6,
          'circle-color': '#d03b3b',
          'circle-stroke-color': '#111814',
          'circle-stroke-width': 2,
        },
      })
      map.addLayer({
        id: 'city-points', type: 'circle', source: 'cities', minzoom: 4,
        paint: {
          'circle-radius': ['interpolate', ['linear'], ['coalesce', ['get', 'population'], 15000], 0, 2.5, 100000, 4.5, 500000, 7, 3500000, 12],
          'circle-color': '#f4df9a', 'circle-opacity': .9,
          'circle-stroke-color': '#14231f', 'circle-stroke-width': 1,
        },
      })
      map.addLayer({
        id: 'city-labels', type: 'symbol', source: 'cities', minzoom: 4.6,
        layout: {
          'text-field': ['get', 'name'], 'text-size': ['interpolate', ['linear'], ['zoom'], 5, 10, 8, 13],
          'text-offset': [0, 1.1], 'text-anchor': 'top', 'text-allow-overlap': false,
        },
        paint: { 'text-color': '#fff5d5', 'text-halo-color': '#101b18', 'text-halo-width': 1.5 },
      })
      map.on('click', 'zone-hit-targets', (event: MapLayerMouseEvent) => {
        const id = event.features?.[0]?.properties?.id as string | undefined
        if (id) selectZone(id)
      })
      map.on('mouseenter', 'zone-hit-targets', () => { map.getCanvas().style.cursor = 'pointer' })
      map.on('mousemove', 'zone-hit-targets', (event: MapLayerMouseEvent) => {
        const id = event.features?.[0]?.properties?.id as string | undefined
        if (!id || id === hoverId.current) return
        if (hoverId.current) map.setFeatureState({ source: 'forecast-zones', id: hoverId.current }, { hover: false })
        hoverId.current = id
        map.setFeatureState({ source: 'forecast-zones', id }, { hover: true })
      })
      map.on('mouseleave', 'zone-hit-targets', () => {
        map.getCanvas().style.cursor = ''
        if (hoverId.current) map.setFeatureState({ source: 'forecast-zones', id: hoverId.current }, { hover: false })
        hoverId.current = null
      })
      window.requestAnimationFrame(() => {
        map.resize()
        if (!reducedMotion) map.flyTo({ ...ETHIOPIA_VIEW, duration: 3000, curve: 1.3, essential: true })
      })
    })
    const cameraFallback = window.setTimeout(() => {
      if (map.getZoom() < 4) map.jumpTo(ETHIOPIA_VIEW)
    }, 3800)
    map.on('webglcontextlost', () => { setGlobeAvailable(false); setMapMode('accessible') })
    map.addControl(new maplibregl.NavigationControl({ showCompass: true, visualizePitch: true }), 'top-right')
    map.addControl(new maplibregl.AttributionControl({ compact: true }), 'bottom-right')
    return () => {
      window.clearTimeout(cameraFallback)
      map.remove()
      mapRef.current = null
    }
  }, [data, globeAvailable, mapMode, reducedMotion, selectZone, setMapMode])

  useEffect(() => {
    const map = mapRef.current
    if (!map?.isStyleLoaded()) return
    const terrainVisible = layers.terrain && Boolean(map.getSource(TERRAIN_SOURCE))
    map.setTerrain(terrainVisible ? { source: TERRAIN_SOURCE, exaggeration: 1.35 } : null)
    if (map.getLayer('terrain-hillshade')) map.setLayoutProperty('terrain-hillshade', 'visibility', terrainVisible ? 'visible' : 'none')
    const visibility: [string, boolean][] = [
      ['probability-relief', layers.probability],
      ['probability-contours', layers.probability],
      ['uncertainty-outline', layers.uncertainty],
      ['observed-events', layers.observations],
      ['source-signals', layers.signals],
      ['exposure-bands', layers.exposure],
      ['city-population-heat', layers.exposure],
      ['motion-traces', layers.motion],
      ['zone-labels', layers.places && showLabels],
      ['city-points', layers.places],
      ['city-labels', layers.places && showLabels],
      ['regional-fill', layers.administrative],
      ['regional-line', layers.administrative],
      ['countries-line', layers.administrative],
    ]
    for (const [id, visible] of visibility) if (map.getLayer(id)) map.setLayoutProperty(id, 'visibility', visible ? 'visible' : 'none')
  }, [layers, showLabels])

  useEffect(() => {
    const map = mapRef.current
    if (!map?.isStyleLoaded()) return
    for (const zone of data.zones) map.setFeatureState({ source: 'forecast-zones', id: zone.id }, { selected: zone.id === selectedZoneId })
    const zone = data.zones.find((item) => item.id === selectedZoneId)
    if (zone) map.easeTo({ center: [zone.position.longitude, zone.position.latitude], zoom: Math.max(map.getZoom(), 6.5), pitch: 64, duration: reducedMotion ? 0 : 720, essential: false })
  }, [data.zones, reducedMotion, selectedZoneId])

  useEffect(() => {
    const map = mapRef.current
    if (!map?.isStyleLoaded()) return
    map.setPaintProperty('probability-relief', 'fill-extrusion-opacity', visualDensity === 'reduced' ? .62 : .82)
  }, [visualDensity])

  useEffect(() => {
    const map = mapRef.current
    if (!map?.isStyleLoaded() || !map.getLayer('zone-labels')) return
    map.setLayoutProperty('zone-labels', 'visibility', showLabels && layers.places ? 'visible' : 'none')
    if (map.getLayer('city-labels')) map.setLayoutProperty('city-labels', 'visibility', showLabels && layers.places ? 'visible' : 'none')
  }, [layers.places, showLabels])

  if (mapMode === 'accessible' || !globeAvailable) return <FallbackMap data={data} globeAvailable={globeAvailable} />
  return (
    <div className="map-canvas-wrap">
      <div ref={container} className="map-canvas" role="application" aria-label="Interactive globe centered on Ethiopia. Use View data for a keyboard-accessible table." />
      <button type="button" className="map-focus" onClick={() => focusEthiopia()} aria-label="Focus map on Ethiopia"><LocateFixed aria-hidden="true" /><span>Focus Ethiopia</span></button>
      <MapSwitch mode="globe" globeAvailable={globeAvailable} onGlobe={() => setMapMode('globe')} onAccessible={() => setMapMode('accessible')} />
    </div>
  )
}
