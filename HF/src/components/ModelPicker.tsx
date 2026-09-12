import { Check, ChevronDown, Cpu, X } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { getRegistryModels } from '../lib/api'
import { LIVE_MODEL_ID, MODEL_CATALOG, PUBLISHED_MODEL_IDS } from '../data/model-catalog'
import type { RegistryModels } from '../contracts/forecast'
import { useForecastStore } from '../lib/store'

function formatPublishedAt(entry: RegistryModels['active']): string {
  if (!entry?.publishedAt) return 'published by the weekly retrain'
  return `published ${entry.publishedAt.slice(0, 10)}`
}

export function ModelPicker() {
  const [open, setOpen] = useState(false)
  const [registry, setRegistry] = useState<RegistryModels | null>(null)
  const dialog = useRef<HTMLDivElement>(null)
  const selectedId = useForecastStore((state) => state.selectedModelId)
  const selectModel = useForecastStore((state) => state.setSelectedModel)
  const effectiveId = selectedId === LIVE_MODEL_ID || PUBLISHED_MODEL_IDS.has(selectedId) ? selectedId : LIVE_MODEL_ID
  const isLive = effectiveId === LIVE_MODEL_ID
  const selected = MODEL_CATALOG.find((model) => model.id === effectiveId)

  useEffect(() => {
    if (!open) return
    const controller = new AbortController()
    getRegistryModels(controller.signal).then((models) => { if (models) setRegistry(models) })
    const close = (event: KeyboardEvent) => { if (event.key === 'Escape') setOpen(false) }
    document.addEventListener('keydown', close)
    return () => { controller.abort(); document.removeEventListener('keydown', close) }
  }, [open])

  const active = registry?.active ?? null
  const liveLabel = active
    ? `${active.version} · top-1 within 20 km ${(active.validationTop1Within20Km ?? 0).toFixed(3)}`
    : 'newest retrained model, promoted automatically'
  const systemCount = MODEL_CATALOG.length + 1 + (registry?.history.length ?? 0)

  return <div className="model-picker">
    <button type="button" className="model-trigger" onClick={() => setOpen(!open)} aria-label={`Select forecast model: ${isLive ? 'LIVE' : selected?.codeName}`} aria-expanded={open} aria-haspopup="dialog"><Cpu aria-hidden="true" /><span><small>Model</small><strong>{isLive ? 'LIVE' : selected?.codeName}</strong></span><ChevronDown aria-hidden="true" /></button>
    {open && <div className="model-menu" role="dialog" aria-modal="false" aria-label="Select forecast model" ref={dialog}>
      <header><div><span>Model registry</span><strong>{systemCount} systems</strong></div><button type="button" onClick={() => setOpen(false)} aria-label="Close model registry"><X aria-hidden="true" /></button></header>
      <p className="model-menu-note">The live registry model is retrained weekly and promoted automatically. Four catalog models have comparable archived Ethiopia outputs; the rest remain catalog entries until an inference adapter is connected.</p>
      <div className="model-list">
        <button type="button" className={isLive ? 'selected' : ''} onClick={() => { selectModel(LIVE_MODEL_ID); setOpen(false) }}>
          <span className="model-status status-live" aria-hidden="true" /><span><strong>Live registry model</strong><small>{liveLabel}</small></span><em>ready</em>{isLive && <Check aria-hidden="true" />}
        </button>
        {(registry?.history ?? []).map((entry) => (
          <button type="button" key={entry.version} disabled>
            <span className="model-status status-retired" aria-hidden="true" /><span><strong>{entry.version}</strong><small>{formatPublishedAt(entry)} · top-1 within 20 km {(entry.validationTop1Within20Km ?? 0).toFixed(3)}</small></span><em>archived</em>
          </button>
        ))}
        {MODEL_CATALOG.map((model) => {
          const disabled = !PUBLISHED_MODEL_IDS.has(model.id)
          return <button type="button" key={model.id} disabled={disabled} className={model.id === effectiveId ? 'selected' : ''} onClick={() => { selectModel(model.id); setOpen(false) }}>
            <span className={`model-status status-${model.status}`} aria-hidden="true" /><span><strong>{model.codeName}</strong><small>{model.family} · {model.id}</small></span><em>{disabled ? 'catalog' : 'ready'}</em>{model.id === effectiveId && <Check aria-hidden="true" />}
          </button>
        })}
      </div>
    </div>}
  </div>
}
