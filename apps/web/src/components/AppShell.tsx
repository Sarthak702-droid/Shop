import { useEffect, useState, type ReactNode } from 'react'
import { NavLink, useLocation, useNavigate } from 'react-router-dom'
import { getReviewer, getToken, setToken } from '../lib/api'
import { workerLabel } from '../lib/format'
import { useApiResource } from '../lib/hooks'
import type { EventListResponse, ReadinessResponse } from '../lib/types'
import {
  IconCamera,
  IconChart,
  IconCpu,
  IconGrid,
  IconModel,
  IconQueue,
  IconSettings,
  IconUpload,
  IconWifiOff,
} from './Icons'
import { Modal, StatusPill } from './Common'

const NAV = [
  { to: '/', label: 'Overview', icon: IconGrid, end: true },
  { to: '/review', label: 'Review queue', icon: IconQueue, badge: true },
  { to: '/cameras', label: 'Cameras', icon: IconCamera },
  { to: '/insights', label: 'Research insights', icon: IconChart },
  { to: '/models', label: 'Models', icon: IconModel },
  { to: '/settings', label: 'Settings', icon: IconSettings },
]

function pageTitle(pathname: string): string {
  if (pathname.startsWith('/review')) return 'Review queue'
  if (pathname.startsWith('/upload')) return 'Add footage'
  if (pathname.startsWith('/jobs')) return 'Analysis job'
  if (pathname.startsWith('/cameras')) return 'Cameras'
  if (pathname.startsWith('/models')) return 'Models'
  if (pathname.startsWith('/insights')) return 'Research insights'
  if (pathname.startsWith('/settings')) return 'Settings'
  return 'Overview'
}

function workerTone(status: string): 'teal' | 'review' | 'muted' {
  if (status === 'idle' || status === 'running') return 'teal'
  if (status === 'blocked' || status === 'not_started' || status === 'stale') return 'review'
  return 'muted'
}

export function AppShell({ children }: { children: ReactNode }) {
  const location = useLocation()
  const navigate = useNavigate()
  const readiness = useApiResource<ReadinessResponse>('/health/ready', { pollMs: 12_000 })
  const events = useApiResource<EventListResponse>('/events?limit=1', { pollMs: 20_000 })
  const [authOpen, setAuthOpen] = useState(false)
  const [tokenDraft, setTokenDraft] = useState('')
  const [tokenError, setTokenError] = useState('')
  const worker = readiness.data?.worker

  useEffect(() => {
    const open = () => setAuthOpen(true)
    window.addEventListener('shop-auth-required', open)
    return () => window.removeEventListener('shop-auth-required', open)
  }, [])

  useEffect(() => {
    if (readiness.error && 'status' in readiness.error && readiness.error.status === 401) {
      setAuthOpen(true)
    }
  }, [readiness.error])

  const unreviewed = events.data?.unreviewed_count ?? 0
  const workerStatus = worker?.status ?? 'unknown'
  const reviewer = getReviewer() || 'Local reviewer'
  const title = pageTitle(location.pathname)

  const applyToken = () => {
    if (!tokenDraft.trim()) {
      setTokenError('Enter the local API token configured for this host.')
      return
    }
    setToken(tokenDraft)
    setAuthOpen(false)
    setTokenError('')
    window.location.reload()
  }

  return (
    <>
      <a className="skip-link" href="#main-content">Skip to main content</a>
      <div className="app-shell">
        <aside className="sidebar" aria-label="Primary navigation">
          <div className="brand-card">
            <img src="/shop-logo.png" alt="Shop" className="brand-logo" />
          </div>
          <nav className="nav-list">
            {NAV.map(({ to, label, icon: Icon, end, badge }) => (
              <NavLink
                key={to}
                to={to}
                end={end}
                className={({ isActive }) => `nav-link${isActive ? ' nav-link--active' : ''}`}
              >
                <Icon />
                <span>{label}</span>
                {badge && unreviewed > 0 && (
                  <span className="nav-badge" aria-label={`${unreviewed} events awaiting review`}>
                    {unreviewed > 99 ? '99+' : unreviewed}
                  </span>
                )}
              </NavLink>
            ))}
          </nav>

          <div className="sidebar-footer">
            <div className="local-mode">
              <span className={`status-dot status-dot--${workerTone(workerStatus)}`} aria-hidden="true" />
              <div>
                <strong>Local processing</strong>
                <span>{worker ? workerLabel(workerStatus) : 'Checking status…'}</span>
              </div>
            </div>
            <div className="cpu-chip"><IconCpu /> CPU only · no cloud inference</div>
            {!readiness.data?.ready && readiness.data && (
              <p className="sidebar-degraded">Readiness is degraded. Review the status panel.</p>
            )}
          </div>
        </aside>

        <div className="main-shell">
          <header className="topbar">
            <div className="breadcrumb" aria-label="Breadcrumb">
              <span>Workspace</span>
              <span className="breadcrumb__sep" aria-hidden="true">/</span>
              <strong>{title}</strong>
            </div>
            <div className="topbar__actions">
              <span className="today-label">
                {new Intl.DateTimeFormat(undefined, { weekday: 'short', month: 'short', day: 'numeric' }).format(new Date())}
              </span>
              <StatusPill tone={readiness.data?.ready ? 'teal' : 'review'} dot>
                {readiness.data?.ready ? 'System ready' : 'Status needs attention'}
              </StatusPill>
              <div className="reviewer-chip" title="Server-side reviewer identity is checked against the local allowlist">
                <span className="reviewer-avatar" aria-hidden="true">{reviewer.slice(0, 2).toUpperCase()}</span>
                <span>{reviewer}</span>
              </div>
              {!location.pathname.startsWith('/upload') && (
                <button className="button button--primary button--small" type="button" onClick={() => navigate('/upload')}>
                  <IconUpload /> Add footage
                </button>
              )}
            </div>
          </header>

          {readiness.error && (
            <div className="shell-banner shell-banner--review" role="status">
              <IconWifiOff />
              <span>
                The local API could not report readiness. Review actions may be unavailable.
                {readiness.error instanceof Error ? ` ${readiness.error.message}` : ''}
              </span>
              <button type="button" className="text-button" onClick={readiness.reload}>Retry</button>
            </div>
          )}

          {readiness.data && !readiness.data.ready && (
            <div className="shell-banner shell-banner--review" role="status">
              <IconWifiOff />
              <span>
                {workerStatus === 'not_started' || workerStatus === 'stale' || workerStatus === 'stopped'
                  ? 'The local analysis worker is not available. Uploads can be validated, but analysis jobs will remain queued.'
                  : 'One or more local readiness checks are failing. Open Overview for dependency detail.'}
              </span>
              <button type="button" className="text-button" onClick={() => navigate('/')}>View status</button>
            </div>
          )}

          <main id="main-content" className="workspace" tabIndex={-1}>
            {children}
          </main>
        </div>
      </div>

      {authOpen && (
        <Modal
          title="Local access token required"
          onClose={() => setAuthOpen(false)}
          footer={
            <>
              <button type="button" className="button button--ghost" onClick={() => setAuthOpen(false)}>Not now</button>
              <button type="button" className="button button--primary" onClick={applyToken}>Use token</button>
            </>
          }
        >
          <p className="modal-intro">
            This Shop instance is configured with a local bearer token. The token stays in this browser's
            local storage and is sent only to this same-origin API.
          </p>
          <div className="field">
            <label htmlFor="api-token">Bearer token</label>
            <input
              id="api-token"
              type="password"
              value={tokenDraft}
              onChange={(event) => setTokenDraft(event.target.value)}
              autoComplete="off"
              placeholder={getToken() ? 'Token is already stored' : 'Enter SHOP_API_TOKEN'}
            />
            {tokenError && <p className="field__error">{tokenError}</p>}
          </div>
          <p className="text-muted small">
            The server reads the token from the <code>SHOP_API_TOKEN</code> environment variable. It is never
            displayed by the API.
          </p>
        </Modal>
      )}
    </>
  )
}
