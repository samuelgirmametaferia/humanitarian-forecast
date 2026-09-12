import { useEffect, useState } from 'react'
import { getLatestForecast } from '../lib/api'
import type { ForecastSnapshot } from '../contracts/forecast'
import { useForecastStore } from '../lib/store'
import { forecastForModel } from '../data/model-forecasts'

export function useForecast() {
  const selectedModelId = useForecastStore((state) => state.selectedModelId)
  const [data, setData] = useState<ForecastSnapshot | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const controller = new AbortController()
    void getLatestForecast(controller.signal)
      .then(setData)
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : 'Forecast unavailable')
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false)
      })
    return () => controller.abort()
  }, [])

  return { data: data ? forecastForModel(data, selectedModelId) : null, error, loading }
}
