import { useEffect, useId, useRef, type ReactNode } from 'react'
import { ApiError, errorMessage } from '../lib/api'
import { IconAlert, IconCheck, IconClose, IconInfo, IconRefresh } from './Icons'

export function Spinner({ label = 'Loading' }: { label?: string }) {
  return (
    <span className="spinner-wrap" role="status">
      <span className="spinner" aria-hidden="true" />
      <span className="sr-only">{label}</span>
    </span>
  )
}

type Tone = 'neutral' | 'teal' | 'review' | 'danger' | 'muted' | 'success'

export function StatusPill({
  children,
  tone = 'neutral',
  dot = false,
}: {
  children: ReactNode
  tone?: Tone
  dot?: boolean
}) {
  return (
    <span className={`pill pill--${tone}`}>
      {dot && <span className={`status-dot status-dot--${tone}`} aria-hidden="true" />}
      {children}
    </span>
  )
}

export function InlineNotice({
  tone = 'info',
  title,
  children,
  action,
}: {
  tone?: 'info' | 'review' | 'danger' | 'success' | 'neutral'
  title?: string
  children: ReactNode
  action?: ReactNode
}) {
  const Icon = tone === 'success' ? IconCheck : tone === 'danger' || tone === 'review' ? IconAlert : IconInfo
  return (
    <div className={`notice notice--${tone}`} role={tone === 'danger' ? 'alert' : 'status'}>
      <Icon className="notice__icon" />
      <div className="notice__body">
        {title && <p className="notice__title">{title}</p>}
        <div className="notice__copy">{children}</div>
      </div>
      {action && <div className="notice__action">{action}</div>}
    </div>
  )
}

export function PageHeader({
  eyebrow,
  title,
  subtitle,
  actions,
}: {
  eyebrow?: string
  title: string
  subtitle?: string
  actions?: ReactNode
}) {
  return (
    <header className="page-header">
      <div>
        {eyebrow && <p className="eyebrow">{eyebrow}</p>}
        <h1>{title}</h1>
        {subtitle && <p className="page-subtitle">{subtitle}</p>}
      </div>
      {actions && <div className="page-actions">{actions}</div>}
    </header>
  )
}

export function SectionHeading({
  title,
  detail,
  action,
}: {
  title: string
  detail?: string
  action?: ReactNode
}) {
  return (
    <div className="section-heading">
      <div>
        <h2>{title}</h2>
        {detail && <p>{detail}</p>}
      </div>
      {action}
    </div>
  )
}

export function MetricCard({
  label,
  value,
  detail,
  tone = 'neutral',
  icon,
}: {
  label: string
  value: ReactNode
  detail?: string
  tone?: 'neutral' | 'review' | 'teal'
  icon?: ReactNode
}) {
  return (
    <div className={`metric-card metric-card--${tone}`}>
      <div className="metric-card__label">
        {icon}
        <span>{label}</span>
      </div>
      <div className="metric-card__value">{value}</div>
      {detail && <p className="metric-card__detail">{detail}</p>}
    </div>
  )
}

export function LoadingPanel({ label = 'Loading local data…', rows = 3 }: { label?: string; rows?: number }) {
  return (
    <div className="panel loading-panel" aria-busy="true" aria-label={label}>
      <div className="skeleton skeleton--title" />
      {Array.from({ length: rows }).map((_, index) => (
        <div className="skeleton-row" key={index}>
          <div className="skeleton skeleton--thumb" />
          <div className="skeleton-stack">
            <div className="skeleton skeleton--line" />
            <div className="skeleton skeleton--line skeleton--short" />
          </div>
        </div>
      ))}
      <span className="sr-only">{label}</span>
    </div>
  )
}

export function ErrorPanel({
  error,
  onRetry,
  title = 'Local API unavailable',
}: {
  error: unknown
  onRetry?: () => void
  title?: string
}) {
  const code = error instanceof ApiError ? error.code : null
  const requestId = error instanceof ApiError ? error.requestId : null
  // A malformed or missing resource will not succeed on retry; offering the button only adds noise.
  const permanent =
    error instanceof ApiError && !error.retryable && [400, 404, 409, 410, 422].includes(error.status)
  return (
    <div className="panel error-panel" role="alert">
      <div className="error-panel__icon">
        <IconAlert />
      </div>
      <div className="error-panel__body">
        <h2>{title}</h2>
        <p>{errorMessage(error)}</p>
        <div className="error-panel__meta">
          {code && <span>Code: {code}</span>}
          {requestId && <span>Request: {requestId}</span>}
        </div>
        {onRetry && !permanent && (
          <button className="button button--secondary button--small" type="button" onClick={onRetry}>
            <IconRefresh /> Retry
          </button>
        )}
      </div>
    </div>
  )
}

export function EmptyState({
  icon,
  title,
  children,
  action,
}: {
  icon?: ReactNode
  title: string
  children?: ReactNode
  action?: ReactNode
}) {
  return (
    <div className="empty-state">
      {icon && <div className="empty-state__icon">{icon}</div>}
      <h2>{title}</h2>
      {children && <div className="empty-state__copy">{children}</div>}
      {action && <div className="empty-state__action">{action}</div>}
    </div>
  )
}

export function Field({
  label,
  htmlFor,
  hint,
  error,
  children,
  optional,
}: {
  label: string
  htmlFor: string
  hint?: string
  error?: string
  children: ReactNode
  optional?: boolean
}) {
  const hintId = `${htmlFor}-hint`
  const errorId = `${htmlFor}-error`
  return (
    <div className={`field ${error ? 'field--error' : ''}`}>
      <label htmlFor={htmlFor}>
        {label} {optional && <span className="field__optional">optional</span>}
      </label>
      {children}
      {hint && <p className="field__hint" id={hintId}>{hint}</p>}
      {error && <p className="field__error" id={errorId}>{error}</p>}
    </div>
  )
}

export function Modal({
  title,
  children,
  onClose,
  footer,
  wide = false,
}: {
  title: string
  children: ReactNode
  onClose: () => void
  footer?: ReactNode
  wide?: boolean
}) {
  const ref = useRef<HTMLDivElement>(null)
  const titleId = useId()
  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null
    const node = ref.current
    const focusable = node?.querySelector<HTMLElement>(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])',
    )
    focusable?.focus()
    const handleKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
      if (event.key !== 'Tab' || !node) return
      const items = Array.from(
        node.querySelectorAll<HTMLElement>(
          'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
        ),
      )
      if (items.length === 0) return
      const first = items[0]
      const last = items[items.length - 1]
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault()
        last.focus()
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault()
        first.focus()
      }
    }
    document.addEventListener('keydown', handleKey)
    const originalOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      document.removeEventListener('keydown', handleKey)
      document.body.style.overflow = originalOverflow
      previous?.focus()
    }
  }, [onClose])

  return (
    <div className="modal-backdrop" onMouseDown={(event) => {
      if (event.target === event.currentTarget) onClose()
    }}>
      <div
        className={`modal ${wide ? 'modal--wide' : ''}`}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        ref={ref}
      >
        <div className="modal__header">
          <h2 id={titleId}>{title}</h2>
          <button className="icon-button" type="button" onClick={onClose} aria-label="Close dialog">
            <IconClose />
          </button>
        </div>
        <div className="modal__body">{children}</div>
        {footer && <div className="modal__footer">{footer}</div>}
      </div>
    </div>
  )
}

export function DataRow({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="data-row">
      <dt>{label}</dt>
      <dd>{children}</dd>
    </div>
  )
}
