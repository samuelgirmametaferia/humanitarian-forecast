import { ArrowDown, ArrowLeft, ArrowRight, X } from 'lucide-react'
import { useState } from 'react'
import { useForecastStore } from '../lib/store'

const steps = [
  {
    eyebrow: 'Why this exists',
    title: 'See pressure building before it becomes a surprise.',
    body: 'Humanitarian teams work with partial information and very little time. HF brings public signals, geography, and tested forecasts into one place so people can decide where to look closer.',
    target: 'mission',
  },
  {
    eyebrow: 'The forecast',
    title: 'Start with the places, not the model.',
    body: 'Each raised site is a specific 20 km watch area. Height and color show its relative share of the current candidate forecast. Click near a site—you do not need to hit the edge.',
    target: 'map',
  },
  {
    eyebrow: 'Forecast lens',
    title: 'Compare a model, then compare the place.',
    body: 'The model control switches between published model outputs. Each code name is a separate forecast run, so the sites and probability distribution can change—not just the label.',
    target: 'model',
  },
  {
    eyebrow: 'People and places',
    title: 'See the city context around a forecast.',
    body: 'Use Layers to show population density, population-scaled city circles, named cities, observed reports, terrain, and uncertainty. Population is context for review, not a measure of need.',
    target: 'layers',
  },
  {
    eyebrow: 'Read the detail',
    title: 'Treat every forecast as a lead to verify.',
    body: 'Select a site to see recency, terrain, population, density, and the signals that shaped its rank. HF is decision support—not an incident report or an evacuation order.',
    target: 'inspector',
  },
]

export function OnboardingTour() {
  const open = useForecastStore((state) => state.tutorialOpen)
  const finish = useForecastStore((state) => state.finishTutorial)
  const reducedMotion = useForecastStore((state) => state.reducedMotion)
  const [step, setStep] = useState(0)
  if (!open) return null
  const item = steps[step]!
  const direction = step === 0 ? 'down' : step === steps.length - 1 ? 'left' : 'right'
  const Arrow = direction === 'down' ? ArrowDown : direction === 'left' ? ArrowLeft : ArrowRight
  const close = () => { setStep(0); finish() }
  return <div className={`tour-overlay tour-step-${step}`} role="dialog" aria-modal="true" aria-labelledby="tour-title">
    <div className={`tour-card tour-point-${direction}`}>
      <button className="tour-close" type="button" onClick={close} aria-label="Close tutorial"><X aria-hidden="true" /></button>
      <span className="tour-count">{String(step + 1).padStart(2, '0')} / {String(steps.length).padStart(2, '0')}</span>
      <p className="tour-eyebrow">{item.eyebrow}</p>
      <h2 id="tour-title">{item.title}</h2>
      <p>{item.body}</p>
      <div className="tour-actions">
        <button type="button" className="tour-skip" onClick={close}>Skip</button>
        <button type="button" className="tour-next" onClick={() => step === steps.length - 1 ? close() : setStep(step + 1)}>{step === steps.length - 1 ? 'Enter HF' : 'Next'} <Arrow aria-hidden="true" /></button>
      </div>
      {!reducedMotion && <span className="tour-arrow" aria-hidden="true"><Arrow /></span>}
    </div>
  </div>
}
