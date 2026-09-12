import { useEffect, useState } from 'react'
import { getForecastHistory, getHealth, getProjectHistory } from '../lib/api'
import type { ForecastSnapshot, Health, ProjectHistory } from '../contracts/forecast'

type ProjectData = {
  forecasts: ForecastSnapshot[]
  health: Health
  history: ProjectHistory
}

export function useProjectData() {
  const [data, setData] = useState<ProjectData | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const controller = new AbortController()
    Promise.all([
      getForecastHistory(controller.signal),
      getHealth(controller.signal),
      getProjectHistory(controller.signal),
    ]).then(([forecasts, health, history]) => setData({ forecasts, health, history }))
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : 'Project data unavailable')
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false)
      })
    return () => controller.abort()
  }, [])

  return { data, error, loading }
}
