import { ContentRoute } from './ContentRoute'
import { useNavigate } from 'react-router-dom'
import { useForecastStore, type InformationDensity, type MapMode, type ThemePreference, type VisualDensity } from '../lib/store'

export function SettingsRoute() {
  const store = useForecastStore()
  const navigate = useNavigate()
  return <ContentRoute title="Settings">
    <section className="settings-section"><h2>Appearance</h2>
      <SettingSelect label="Theme" value={store.theme} onChange={(value) => store.setTheme(value as ThemePreference)} options={[['system', 'System'], ['light', 'Light'], ['dark', 'Dark']]} />
      <SettingToggle label="Reduced motion" checked={store.reducedMotion} onChange={store.setReducedMotion} />
      <SettingSelect label="Information density" value={store.informationDensity} onChange={(value) => store.setInformationDensity(value as InformationDensity)} options={[['comfortable', 'Comfortable'], ['compact', 'Compact']]} />
    </section>
    <section className="settings-section"><h2>Map</h2>
      <SettingSelect label="Default map" value={store.mapMode} onChange={(value) => store.setMapMode(value as MapMode)} options={[['globe', '3D Globe'], ['accessible', 'Accessible Map']]} />
      <SettingToggle label="Zone labels" checked={store.showLabels} onChange={store.setShowLabels} />
      <SettingSelect label="Visualization density" value={store.visualDensity} onChange={(value) => store.setVisualDensity(value as VisualDensity)} options={[['standard', 'Standard'], ['reduced', 'Reduced']]} />
    </section>
    <section className="settings-section"><h2>Guide</h2><p>Replay the short introduction to HF and its map.</p><button className="settings-action" type="button" onClick={() => { store.startTutorial(); void navigate('/') }}>Replay tutorial</button></section>
    <section className="settings-section"><h2>Documents</h2><p><a href="/privacy" target="_blank" rel="noreferrer">Privacy Policy</a></p><p><a href="/terms" target="_blank" rel="noreferrer">Terms of Service</a></p></section>
  </ContentRoute>
}

function SettingSelect({ label, value, options, onChange }: { label: string; value: string; options: [string, string][]; onChange: (value: string) => void }) {
  return <label className="setting-row"><span>{label}</span><select value={value} onChange={(event) => onChange(event.target.value)}>{options.map(([key, text]) => <option key={key} value={key}>{text}</option>)}</select></label>
}

function SettingToggle({ label, checked, onChange }: { label: string; checked: boolean; onChange: (value: boolean) => void }) {
  return <label className="setting-row"><span>{label}</span><input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} /></label>
}
