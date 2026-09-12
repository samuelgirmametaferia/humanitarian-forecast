export function formatDate(value: string, includeTime = false) {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return 'Unknown'
  return new Intl.DateTimeFormat('en', {
    day: 'numeric',
    month: 'short',
    year: 'numeric',
    ...(includeTime ? { hour: '2-digit', minute: '2-digit', timeZoneName: 'short' } : {}),
    timeZone: 'UTC',
  }).format(date)
}

export function formatPercent(value: number, digits = 1) {
  return new Intl.NumberFormat('en', {
    style: 'percent',
    maximumFractionDigits: digits,
  }).format(value)
}

export function ageInHours(value: string) {
  return Math.max(0, (Date.now() - new Date(value).getTime()) / 3_600_000)
}
