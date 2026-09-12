import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { LayerId } from '../contracts/forecast'
import { DEFAULT_MODEL_ID } from '../data/model-catalog'

export type ThemePreference = 'system' | 'light' | 'dark'
export type MapMode = 'globe' | 'accessible'
export type VisualDensity = 'standard' | 'reduced'
export type InformationDensity = 'comfortable' | 'compact'

const DEFAULT_LAYERS: Record<LayerId, boolean> = {
  probability: true,
  uncertainty: true,
  observations: true,
  signals: true,
  terrain: true,
  exposure: false,
  populationDots: true,
  motion: true,
  places: true,
  administrative: true,
}

type ForecastState = {
  layers: Record<LayerId, boolean>
  selectedZoneId: string | null
  tableOpen: boolean
  theme: ThemePreference
  reducedMotion: boolean
  mapMode: MapMode
  showLabels: boolean
  visualDensity: VisualDensity
  informationDensity: InformationDensity
  tutorialOpen: boolean
  tutorialSeen: boolean
  selectedModelId: string
  setSelectedZone: (id: string | null) => void
  toggleLayer: (id: LayerId) => void
  setTableOpen: (open: boolean) => void
  setTheme: (theme: ThemePreference) => void
  setReducedMotion: (reducedMotion: boolean) => void
  setMapMode: (mapMode: MapMode) => void
  setShowLabels: (showLabels: boolean) => void
  setVisualDensity: (visualDensity: VisualDensity) => void
  setInformationDensity: (informationDensity: InformationDensity) => void
  startTutorial: () => void
  finishTutorial: () => void
  setSelectedModel: (id: string) => void
}

export const useForecastStore = create<ForecastState>()(persist((set) => ({
  layers: DEFAULT_LAYERS,
  selectedZoneId: null,
  tableOpen: false,
  theme: 'system',
  reducedMotion: false,
  mapMode: 'globe',
  showLabels: true,
  visualDensity: 'standard',
  informationDensity: 'comfortable',
  tutorialOpen: true,
  tutorialSeen: false,
  selectedModelId: DEFAULT_MODEL_ID,
  setSelectedZone: (selectedZoneId) => set({ selectedZoneId }),
  toggleLayer: (id) => set((state) => ({ layers: { ...state.layers, [id]: !state.layers[id] } })),
  setTableOpen: (tableOpen) => set({ tableOpen }),
  setTheme: (theme) => set({ theme }),
  setReducedMotion: (reducedMotion) => set({ reducedMotion }),
  setMapMode: (mapMode) => set({ mapMode }),
  setShowLabels: (showLabels) => set({ showLabels }),
  setVisualDensity: (visualDensity) => set({ visualDensity }),
  setInformationDensity: (informationDensity) => set({ informationDensity }),
  startTutorial: () => set({ tutorialOpen: true }),
  finishTutorial: () => set({ tutorialOpen: false, tutorialSeen: true }),
  setSelectedModel: (selectedModelId) => set({ selectedModelId }),
}), {
  name: 'hf-preferences',
  partialize: ({ layers, theme, reducedMotion, mapMode, showLabels, visualDensity, informationDensity, tutorialSeen, selectedModelId }) => ({
    layers, theme, reducedMotion, mapMode, showLabels, visualDensity, informationDensity,
    tutorialSeen, selectedModelId,
  }),
  merge: (persisted, current) => {
    const saved = persisted as Partial<ForecastState>
    return { ...current, ...saved, layers: { ...current.layers, ...saved.layers }, tutorialOpen: !saved.tutorialSeen }
  },
}))
