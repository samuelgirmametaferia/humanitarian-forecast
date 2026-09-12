import { Table2 } from 'lucide-react'
import type { ForecastSnapshot } from '../../contracts/forecast'
import { formatDate } from '../../lib/format'
import { useForecastStore } from '../../lib/store'
import { ModelPicker } from '../../components/ModelPicker'

export function ForecastHeader({ data }: { data: ForecastSnapshot }) {
  const setTable = useForecastStore((state) => state.setTableOpen)
  return <header className="forecast-header">
    <div><h1>Humanitarian Forecaster</h1><p>Spatial early warning · Ethiopia</p></div>
    <div className="forecast-meta"><div className="forecast-updated"><span>Last update</span><time dateTime={data.generatedAt}>{formatDate(data.generatedAt, true)}</time></div><ModelPicker /><button type="button" onClick={() => setTable(true)}><Table2 aria-hidden="true" />View data</button></div>
  </header>
}
