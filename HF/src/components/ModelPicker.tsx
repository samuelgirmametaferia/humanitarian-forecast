import { Check, ChevronDown, Cpu, X } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { DEFAULT_MODEL_ID, MODEL_CATALOG, PUBLISHED_MODEL_IDS } from '../data/model-catalog'
import { useForecastStore } from '../lib/store'

export function ModelPicker() {
  const [open, setOpen] = useState(false)
  const dialog = useRef<HTMLDivElement>(null)
  const selectedId = useForecastStore((state) => state.selectedModelId)
  const selectModel = useForecastStore((state) => state.setSelectedModel)
  const effectiveId = PUBLISHED_MODEL_IDS.has(selectedId) ? selectedId : DEFAULT_MODEL_ID
  const selected = MODEL_CATALOG.find((model) => model.id === effectiveId)!

  useEffect(() => {
    if (!open) return
    const close = (event: KeyboardEvent) => { if (event.key === 'Escape') setOpen(false) }
    document.addEventListener('keydown', close)
    return () => document.removeEventListener('keydown', close)
  }, [open])

  return <div className="model-picker">
    <button type="button" className="model-trigger" onClick={() => setOpen(!open)} aria-label={`Select forecast model: ${selected.codeName}`} aria-expanded={open} aria-haspopup="dialog"><Cpu aria-hidden="true" /><span><small>Model</small><strong>{selected.codeName}</strong></span><ChevronDown aria-hidden="true" /></button>
    {open && <div className="model-menu" role="dialog" aria-modal="false" aria-label="Select forecast model" ref={dialog}>
      <header><div><span>Model registry</span><strong>33 systems</strong></div><button type="button" onClick={() => setOpen(false)} aria-label="Close model registry"><X aria-hidden="true" /></button></header>
      <p className="model-menu-note">Four models have comparable Ethiopia outputs for the same archived input. The rest remain visible as catalog entries until an inference adapter is connected.</p>
      <div className="model-list">{MODEL_CATALOG.map((model) => {
        const disabled = !PUBLISHED_MODEL_IDS.has(model.id)
        return <button type="button" key={model.id} disabled={disabled} className={model.id === selected.id ? 'selected' : ''} onClick={() => { selectModel(model.id); setOpen(false) }}>
          <span className={`model-status status-${model.status}`} aria-hidden="true" /><span><strong>{model.codeName}</strong><small>{model.family} · {model.id}</small></span><em>{disabled ? 'catalog' : 'ready'}</em>{model.id === selected.id && <Check aria-hidden="true" />}
        </button>
      })}</div>
    </div>}
  </div>
}
