import { Link } from 'react-router-dom'
import type { EventSummary } from '../lib/types'
import {
  eventTypeLabel,
  formatDuration,
  formatMediaTime,
  formatTime,
  reasonLabel,
  reviewStateLabel,
} from '../lib/format'
import { useBlobUrl } from '../lib/hooks'
import { IconFilm, IconPlay } from './Icons'
import { StatusPill } from './Common'
import { ACTION_COLORS } from './Charts'

function reviewTone(state: EventSummary['review_state']): 'review' | 'success' | 'muted' | 'teal' {
  if (state === 'unreviewed') return 'review'
  if (state === 'benign') return 'success'
  if (state === 'review_needed') return 'review'
  return 'muted'
}

export function EvidenceThumbnail({ event, compact = false }: { event: EventSummary; compact?: boolean }) {
  const url = event.evidence.thumbnail_url
  const blob = useBlobUrl(url, { enabled: Boolean(url) })
  if (compact) {
    return (
      <div className="event-thumb event-thumb--compact">
        {blob.url ? <img src={blob.url} alt="" /> : <IconFilm />}
      </div>
    )
  }
  return (
    <div className={`event-thumb${blob.url ? ' event-thumb--loaded' : ''}`}>
      {blob.url ? (
        <>
          <img src={blob.url} alt="Sampled evidence thumbnail for this activity segment" />
          <span className="event-thumb__play"><IconPlay /></span>
        </>
      ) : (
        <div className="event-thumb__fallback">
          <IconFilm />
          <span>{blob.loading ? 'Loading preview' : 'No thumbnail'}</span>
        </div>
      )}
    </div>
  )
}

export function EventRow({
  event,
  selected = false,
  to,
}: {
  event: EventSummary
  selected?: boolean
  to?: string
}) {
  const sourceLabel = event.source?.label ?? 'Local source'
  const duration = event.media_interval_ms.end - event.media_interval_ms.start
  return (
    <Link
      to={to ?? `/review/${event.event_id}`}
      className={`event-row${selected ? ' event-row--selected' : ''}`}
      aria-current={selected ? 'true' : undefined}
    >
      <EvidenceThumbnail event={event} compact />
      <div className="event-row__body">
        <div className="event-row__top">
          <strong>{sourceLabel}</strong>
          <span className="event-row__time">{formatTime(event.created_at)}</span>
        </div>
        <p className="event-row__activity">{eventTypeLabel(event.event_type)}</p>
        {event.actions && event.actions.length > 0 ? (
          <p className="event-row__reason chip-row">
            {[...new Set(event.actions.map((a) => a.label))].map((label) => (
              <span className="chip chip--action" key={label}>
                <span className="chart-legend__swatch" style={{ background: ACTION_COLORS[label] }} aria-hidden="true" />
                {reasonLabel(label)}
              </span>
            ))}
          </p>
        ) : (
          <p className="event-row__reason">
            {event.reasons.slice(0, 2).map(reasonLabel).join(' · ') || 'No reason codes recorded'}
          </p>
        )}
        <div className="event-row__meta">
          <span>Media {formatMediaTime(event.media_interval_ms.start)}–{formatMediaTime(event.media_interval_ms.end)}</span>
          <span>{formatDuration(duration)}</span>
        </div>
        <div className="event-row__pills">
          <StatusPill tone={reviewTone(event.review_state)} dot>
            {reviewStateLabel(event.review_state)}
          </StatusPill>
          {event.event_type === 'shelf_interaction_observed' && <StatusPill tone="muted">Research model</StatusPill>}
          {Array.isArray(event.quality?.warnings) && event.quality.warnings.includes('insufficient_person_visibility') && (
            <StatusPill tone="review">Low visibility</StatusPill>
          )}
          {event.evidence.status !== 'available' && (
            <StatusPill tone="muted">{event.evidence.status === 'pending' ? 'Evidence pending' : 'Evidence unavailable'}</StatusPill>
          )}
        </div>
      </div>
      <span className="event-row__chevron" aria-hidden="true">›</span>
    </Link>
  )
}
