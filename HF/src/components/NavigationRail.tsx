import { Activity, BookOpen, Clock3, Info, Map, Settings } from 'lucide-react'
import { NavLink } from 'react-router-dom'

const links = [
  { to: '/', label: 'Forecast', icon: Map },
  { to: '/history', label: 'History', icon: Clock3 },
  { to: '/methodology', label: 'Methodology', icon: BookOpen },
  { to: '/status', label: 'Status', icon: Activity },
  { to: '/settings', label: 'Settings', icon: Settings },
  { to: '/about', label: 'About', icon: Info },
]

export function NavigationRail() {
  return <nav className="navigation-rail" aria-label="Primary navigation" data-tour="mission">
    <NavLink className="brand-mark" to="/" aria-label="HF home"><span>HF</span></NavLink>
    <div className="nav-links">{links.map(({ to, label, icon: Icon }) => <NavLink key={to} to={to} end={to === '/'} title={label}><Icon aria-hidden="true" /><span>{label}</span></NavLink>)}</div>
    <div className="nav-legal">
      <a href="/privacy" target="_blank" rel="noreferrer">Privacy</a>
      <a href="/terms" target="_blank" rel="noreferrer">Terms</a>
    </div>
  </nav>
}
