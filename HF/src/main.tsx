import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { MotionConfig } from 'motion/react'
import App from './App'
import { useForecastStore } from './lib/store'
import '@fontsource-variable/public-sans'
import '@fontsource-variable/newsreader'
import './styles/index.css'

function Root() {
  const theme = useForecastStore((state) => state.theme)
  const reducedMotion = useForecastStore((state) => state.reducedMotion)
  const informationDensity = useForecastStore((state) => state.informationDensity)
  React.useEffect(() => {
    if (theme === 'system') document.documentElement.removeAttribute('data-theme')
    else document.documentElement.dataset.theme = theme
    document.documentElement.dataset.density = informationDensity
    document.documentElement.dataset.reducedMotion = String(reducedMotion)
  }, [informationDensity, reducedMotion, theme])
  return <MotionConfig reducedMotion={reducedMotion ? 'always' : 'user'}><BrowserRouter><App /></BrowserRouter></MotionConfig>
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode><Root /></React.StrictMode>,
)
