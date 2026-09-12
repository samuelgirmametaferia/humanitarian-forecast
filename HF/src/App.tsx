import { Suspense, lazy } from 'react'
import { Navigate, Route, Routes } from 'react-router-dom'
import { NavigationRail } from './components/NavigationRail'
import { OnboardingTour } from './components/OnboardingTour'
import { ForecastRoute } from './routes/ForecastRoute'

const HistoryRoute = lazy(() => import('./routes/HistoryRoute').then((module) => ({ default: module.HistoryRoute })))
const MethodologyRoute = lazy(() => import('./routes/MethodologyRoute').then((module) => ({ default: module.MethodologyRoute })))
const StatusRoute = lazy(() => import('./routes/StatusRoute').then((module) => ({ default: module.StatusRoute })))
const SettingsRoute = lazy(() => import('./routes/SettingsRoute').then((module) => ({ default: module.SettingsRoute })))
const AboutRoute = lazy(() => import('./routes/AboutRoute').then((module) => ({ default: module.AboutRoute })))
const PrivacyRoute = lazy(() => import('./routes/PrivacyRoute').then((module) => ({ default: module.PrivacyRoute })))
const TermsRoute = lazy(() => import('./routes/TermsRoute').then((module) => ({ default: module.TermsRoute })))

export default function App() {
  const focusMain = (attempt = 0) => {
    const main = document.getElementById('main-content')
    if (main) main.focus()
    else if (attempt < 10) window.setTimeout(() => focusMain(attempt + 1), 100)
  }
  return <div className="app-shell">
    <a className="skip-link" href="#main-content" onClick={(event) => { event.preventDefault(); focusMain() }}>Skip to main content</a>
    <NavigationRail />
    <OnboardingTour />
    <Suspense fallback={<main className="route-state" aria-live="polite">Loading…</main>}>
      <Routes>
        <Route path="/" element={<ForecastRoute />} />
        <Route path="/history" element={<HistoryRoute />} />
        <Route path="/methodology" element={<MethodologyRoute />} />
        <Route path="/status" element={<StatusRoute />} />
        <Route path="/settings" element={<SettingsRoute />} />
        <Route path="/about" element={<AboutRoute />} />
        <Route path="/privacy" element={<PrivacyRoute />} />
        <Route path="/terms" element={<TermsRoute />} />
        <Route path="/policy" element={<Navigate to="/privacy" replace />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Suspense>
  </div>
}
