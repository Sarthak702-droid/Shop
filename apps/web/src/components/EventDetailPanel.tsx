import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  ApiError,
  apiDelete,
  apiPost,
  errorMessage,
  getReviewer,
  newIdempotencyKey,
} from '../lib/api'
import {
  eventTypeLabel,
  evidenceStatusLabel,
  formatBytes,
  formatDateTime,
  formatDuration,
  formatMediaTime,
  reasonLabel,
  reviewDecisionLabel,
  reviewStateLabel,
} from '../lib/format'
import { useApiResource, useBlobUrl } from '../lib/hooks'
import type { EventSummary, ReviewDecision } from '../lib/types'
import {
  DataRow,
  EmptyState,
  ErrorPanel,
  InlineNotice,
  LoadingPanel,
  Modal,
  StatusPill,
} from './Common'
import {
  IconAlert,
  IconArrowLeft,
  IconCheck,
  IconClose,
  IconFilm,
  IconInfo,
  IconPlay,
  IconShield,
  IconTrash,
} from './Icons'

const DECISIONS: Array<{
  value: ReviewDecision
  label: string
  description: string
  defaultReason: string
}> = [
  {
    value: 'benign',
    label: 'Mark benign',
    description: 'After context review, the observable activity is consistent with normal activity.',
    defaultReason: 'activity_benign',
  },
  {
    value: 'review_needed',
    label: 'Needs review',
    description: 'Context is incomplete, ambiguous or needs another reviewer.',
    defaultReason: 'context_insufficient',
  },
  {
    value: 'inconclusive',
    label: 'Inconclusive',
    description: 'The available evidence does not support a reliable review outcome.',
    defaultReason: 'context_insufficient',
  },
]

const REASONS = [
  'activity_benign',
  'context_insufficient',
  'occlusion',
  'product_unclear',
  'customer_owned_item',
  'equipment_fault',
  'other',
] as const

const DECISION_TONE: Record<ReviewDecision, 'success' | 'review' | 'muted'> = {
  benign: 'success',
  review_needed: 'review',
  inconclusive: 'muted',
}

function latestReview(event: EventSummary) {
  if (!event.reviews?.length) return null
  return [...event.reviews].sort(
    (a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime(),
  ).at(-1) ?? null
}

function EvidencePlayer({ event }: { event: EventSummary }) {
  const [requested, setRequested] = useState(false)
  const [reloadKey, setReloadKey] = useState(0)
  const thumbnail = useBlobUrl(event.evidence.thumbnail_url, { enabled: Boolean(event.evidence.thumbnail_url) })
  const media = useBlobUrl(event.evidence.media_url, {
    enabled: requested && Boolean(event.evidence.media_url),
    version: reloadKey,
  })

  const asset = event.evidence.asset
  const clipStart = asset?.media_interval_ms?.start ?? event.media_interval_ms.start
  const clipEnd = asset?.media_interval_ms?.end ?? event.media_interval_ms.end
  const clipSpan = Math.max(1, clipEnd - clipStart)
  const eventStartPct = Math.max(0, Math.min(100, ((event.media_interval_ms.start - clipStart) / clipSpan) * 100))
  const eventEndPct = Math.max(eventStartPct, Math.min(100, ((event.media_interval_ms.end - clipStart) / clipSpan) * 100))

  if (event.evidence.status !== 'available') {
    return (
      <div className="evidence-unavailable">
        <IconFilm />
        <div>
          <strong>{evidenceStatusLabel(event.evidence.status)}</strong>
          <p>{event.evidence.reason ?? 'No evidence clip is attached to this event.'}</p>
          <p className="small text-muted">
            The event remains reviewable, but visual context may be insufficient. Required evidence is listed below.
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="evidence-player">
      <div className="evidence-stage">
        {requested && media.url ? (
          <video controls playsInline preload="metadata" src={media.url} className="evidence-video">
            Your browser cannot play this local WebM evidence clip.
          </video>
        ) : (
          <div className="evidence-poster">
            {thumbnail.url ? (
              <img src={thumbnail.url} alt="Sampled evidence context for this event" />
            ) : (
              <div className="evidence-poster__placeholder"><IconFilm /></div>
            )}
            <button
              className="evidence-load"
              type="button"
              disabled={media.loading || !event.evidence.media_url}
              onClick={() => setRequested(true)}
            >
              <span className="evidence-load__circle"><IconPlay /></span>
              <span>{media.loading ? 'Loading local evidence…' : 'Load evidence clip'}</span>
            </button>
            <span className="evidence-stage__badge">Local evidence · no cloud video</span>
          </div>
        )}
        {requested && media.error && (
          <div className="evidence-media-error">
            <IconAlert />
            <div>
              <strong>Evidence media could not load</strong>
              <p>{errorMessage(media.error)}</p>
              <button className="text-button" type="button" onClick={() => setReloadKey((value) => value + 1)}>
                Retry clip
              </button>
            </div>
          </div>
        )}
      </div>
      <div className="evidence-timeline" aria-label="Evidence timeline: context, event, context">
        <div className="timeline-labels">
          <span>Before {formatDuration((event.media_interval_ms.start - clipStart))}</span>
          <strong>Observed interval {formatDuration(event.media_interval_ms.end - event.media_interval_ms.start)}</strong>
          <span>After {formatDuration(Math.max(0, clipEnd - event.media_interval_ms.end))}</span>
        </div>
        <div className="timeline-track">
          <span className="timeline-context" style={{ left: 0, width: `${eventStartPct}%` }} />
          <span className="timeline-event" style={{ left: `${eventStartPct}%`, width: `${Math.max(1, eventEndPct - eventStartPct)}%` }} />
          <span className="timeline-context" style={{ left: `${eventEndPct}%`, right: 0 }} />
        </div>
        <div className="timeline-labels timeline-labels--meta">
          <span>clip {formatMediaTime(clipStart)}–{formatMediaTime(clipEnd)}</span>
          <span>{asset ? `${formatDuration(asset.duration_ms)} · ${asset.fps} fps · ${formatBytes(asset.bytes)}` : 'asset metadata loading'}</span>
        </div>
      </div>
    </div>
  )
}

function ReviewComposer({
  event,
  onRecorded,
}: {
  event: EventSummary
  onRecorded: () => void
}) {
  const previous = latestReview(event)
  const [decision, setDecision] = useState<ReviewDecision>(
    previous?.decision ?? 'benign',
  )
  const [reasons, setReasons] = useState<string[]>(
    previous?.reason_codes.length ? [...previous.reason_codes] : ['activity_benign'],
  )
  const [notes, setNotes] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<unknown>(null)
  const [saved, setSaved] = useState(false)
  const reviewer = getReviewer() || 'Local reviewer (server default)'

  useEffect(() => {
    setDecision(previous?.decision ?? 'benign')
    setReasons(previous?.reason_codes.length ? [...previous.reason_codes] : ['activity_benign'])
  }, [event.event_id, previous?.review_id])

  const applyDecision = (value: ReviewDecision) => {
    const option = DECISIONS.find((item) => item.value === value)
    setDecision(value)
    setSaved(false)
    if (option) {
      setReasons((current) => Array.from(new Set([option.defaultReason, ...current])))
    }
  }

  const toggleReason = (reason: string) => {
    setReasons((current) =>
      current.includes(reason) ? current.filter((item) => item !== reason) : [...current, reason],
    )
  }

  const submit = async () => {
    setError(null)
    setSaved(false)
    if (reasons.length === 0 && !notes.trim()) {
      setError(new Error('Choose at least one reason or add a short note.'))
      return
    }
    setSubmitting(true)
    try {
      await apiPost(
        `/events/${event.event_id}/reviews`,
        {
          decision,
          reason_codes: reasons,
          notes: notes.slice(0, 2000),
          supersedes_review_id: previous?.review_id ?? null,
        },
        { 'Idempotency-Key': newIdempotencyKey() },
      )
      setNotes('')
      setSaved(true)
      onRecorded()
      window.dispatchEvent(new CustomEvent('shop-queue-changed'))
    } catch (caught) {
      setError(caught)
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="review-composer" id="review-actions">
      <div className="review-composer__heading">
        <div>
          <h3>{previous ? 'Record a correction' : 'Record review outcome'}</h3>
          <p>
            Signed as <strong>{reviewer}</strong>. A new record is appended; earlier decisions remain visible.
          </p>
        </div>
        {previous && <StatusPill tone="muted">Supersedes {previous.review_id.slice(0, 8)}…</StatusPill>}
      </div>

      <div className="decision-grid" role="group" aria-label="Review decision">
        {DECISIONS.map((option) => (
          <button
            type="button"
            key={option.value}
            className={`decision-card decision-card--${option.value}${decision === option.value ? ' decision-card--selected' : ''}`}
            aria-pressed={decision === option.value}
            onClick={() => applyDecision(option.value)}
          >
            <span className="decision-card__mark" aria-hidden="true">
              {decision === option.value && <IconCheck />}
            </span>
            <span><strong>{option.label}</strong><small>{option.description}</small></span>
          </button>
        ))}
      </div>

      <div className="review-form-grid">
        <fieldset className="reason-fieldset">
          <legend>Context reasons</legend>
          <p className="field__hint">These describe review context, not suspicion. At least one is required unless a note is supplied.</p>
          <div className="checkbox-grid">
            {REASONS.map((reason) => (
              <label className="checkbox-row" key={reason}>
                <input
                  type="checkbox"
                  checked={reasons.includes(reason)}
                  onChange={() => toggleReason(reason)}
                />
                <span>{reasonLabel(reason)}</span>
              </label>
            ))}
          </div>
        </fieldset>
        <div className="field notes-field">
          <label htmlFor="review-notes">Reviewer note <span className="field__optional">optional · 2000 characters</span></label>
          <textarea
            id="review-notes"
            rows={5}
            value={notes}
            maxLength={2000}
            onChange={(event) => setNotes(event.target.value)}
            placeholder="Describe what the evidence shows and what remains unclear."
          />
          <div className="char-count">{notes.length} / 2000</div>
        </div>
      </div>

      {Boolean(error) && (
        <InlineNotice tone="danger" title="Review was not recorded">
          {errorMessage(error)}
        </InlineNotice>
      )}
      {saved && (
        <InlineNotice tone="success" title="Review recorded">
          The event's current review state is now <strong>{reviewStateLabel(decision)}</strong>. This is an operational
          review outcome, not a finding of guilt.
        </InlineNotice>
      )}

      <div className="review-composer__actions">
        <p>Operational review only · no enforcement action is available.</p>
        <button className="button button--primary" type="button" onClick={() => void submit()} disabled={submitting}>
          {submitting ? 'Recording locally…' : previous ? 'Append correction' : `Save · ${reviewDecisionLabel(decision)}`}
        </button>
      </div>
    </div>
  )
}

export function EventDetailPanel({
  eventId,
  onBack,
  onDeleted,
}: {
  eventId: string
  onBack?: () => void
  onDeleted?: () => void
}) {
  const navigate = useNavigate()
  const resource = useApiResource<{ event: EventSummary }>(`/events/${eventId}`)
  const [deleteOpen, setDeleteOpen] = useState(false)
  const [deleting, setDeleting] = useState(false)
  const [deleteError, setDeleteError] = useState<unknown>(null)
  const event = resource.data?.event
  const currentReview = event ? latestReview(event) : null

  useEffect(() => {
    const onChanged = () => resource.reload()
    window.addEventListener('shop-queue-changed', onChanged)
    return () => window.removeEventListener('shop-queue-changed', onChanged)
  }, [resource])

  const warnings = useMemo(() => {
    const values = event?.quality.warnings
    return Array.isArray(values) ? values.map(String) : []
  }, [event])

  const deleteEvent = async () => {
    if (!event) return
    setDeleting(true)
    setDeleteError(null)
    try {
      await apiDelete(`/events/${event.event_id}`)
      setDeleteOpen(false)
      window.dispatchEvent(new CustomEvent('shop-queue-changed'))
      onDeleted?.()
      navigate('/review')
    } catch (caught) {
      setDeleteError(caught)
    } finally {
      setDeleting(false)
    }
  }

  if (resource.loading && !event) return <div className="panel"><LoadingPanel label="Loading event evidence…" rows={4} /></div>
  if (resource.error && !event) return <div className="panel"><ErrorPanel error={resource.error} onRetry={resource.reload} title="Event could not load" /></div>
  if (!event) {
    return (
      <div className="panel">
        <EmptyState icon={<IconInfo />} title="Select an activity segment">
          <p>Choose a row in the queue to inspect its local evidence and record a context review.</p>
        </EmptyState>
      </div>
    )
  }

  return (
    <div className="event-detail">
      {onBack && (
        <button className="back-button" type="button" onClick={onBack}>
          <IconArrowLeft /> Back to queue
        </button>
      )}
      <div className="event-detail__header">
        <div>
          <div className="event-detail__eyebrow">
            <span>{event.source?.label ?? 'Local source'}</span>
            <span aria-hidden="true">·</span>
            <span>{formatDateTime(event.created_at)}</span>
          </div>
          <h2>{eventTypeLabel(event.event_type)}</h2>
          <div className="event-detail__pills">
            <StatusPill tone={event.review_state === 'unreviewed' ? 'review' : currentReview ? DECISION_TONE[currentReview.decision] : 'muted'} dot>
              {reviewStateLabel(event.review_state)}
            </StatusPill>
            <StatusPill tone="muted">Human review queue</StatusPill>
            {warnings.length > 0 && <StatusPill tone="review">{warnings.length} quality warning{warnings.length === 1 ? '' : 's'}</StatusPill>}
          </div>
        </div>
        <div className="event-detail__actions">
          <button className="button button--ghost button--small" type="button" onClick={resource.reload} disabled={resource.refreshing}>
            {resource.refreshing ? 'Refreshing…' : 'Refresh'}
          </button>
          <button className="button button--danger-ghost button--small" type="button" onClick={() => setDeleteOpen(true)}>
            <IconTrash /> Delete
          </button>
        </div>
      </div>

      <InlineNotice tone="neutral" title="Observed activity, not a conclusion">
        The baseline reports a person track and its context. It does not establish intent, ownership, payment or
        whether a theft occurred. Reviewers may find a benign explanation or mark the evidence inconclusive.
      </InlineNotice>

      <EvidencePlayer event={event} />

      <div className="detail-grid">
        <section className="panel detail-panel">
          <h3>Observable reasons</h3>
          <ul className="reason-list">
            {event.reasons.length === 0 && <li>No reason codes were recorded.</li>}
            {event.reasons.map((reason) => (
              <li key={reason}><span className="reason-dot" aria-hidden="true" />{reasonLabel(reason)}</li>
            ))}
          </ul>
          <h3 className="detail-subheading">Quality and context</h3>
          <dl className="data-grid data-grid--compact">
            <DataRow label="Event media interval">
              {formatMediaTime(event.media_interval_ms.start)} – {formatMediaTime(event.media_interval_ms.end)}
            </DataRow>
            <DataRow label="Observed duration">
              {formatDuration(event.media_interval_ms.end - event.media_interval_ms.start)}
            </DataRow>
            <DataRow label="Observed samples">
              {typeof event.quality.observed_samples === 'number' ? event.quality.observed_samples : '—'}
            </DataRow>
            <DataRow label="Observed fraction">
              {typeof event.quality.observed_fraction === 'number' ? `${Math.round(event.quality.observed_fraction * 100)}%` : '—'}
            </DataRow>
            <DataRow label="Max persons in scene">
              {typeof event.quality.crowded_max_persons === 'number' ? event.quality.crowded_max_persons : '—'}
            </DataRow>
            <DataRow label="Session-local tracks">
              {event.track_ids.length ? event.track_ids.join(', ') : 'No stable track ID available'}
            </DataRow>
          </dl>
          {warnings.length > 0 && (
            <div className="warning-list">
              {warnings.map((warning) => (
                <div key={warning}><IconAlert /><span>{reasonLabel(warning)}</span></div>
              ))}
            </div>
          )}
          <div className="evidence-requirements">
            <h3 className="detail-subheading">Evidence provenance</h3>
            <div className="req-list">
              {event.required_evidence.map((item) => (
                <span key={item} className={`req-chip ${event.available_evidence.includes(item) ? 'req-chip--present' : 'req-chip--missing'}`}>
                  {event.available_evidence.includes(item) ? <IconCheck /> : <IconClose />}
                  {reasonLabel(item)}
                </span>
              ))}
            </div>
            {event.evidence.asset && (
              <dl className="data-grid data-grid--compact">
                <DataRow label="Clip integrity">{event.evidence.asset.sha256.slice(0, 20)}…</DataRow>
                <DataRow label="Clip details">
                  {event.evidence.asset.width}×{event.evidence.asset.height} · {event.evidence.asset.fps} fps · {event.evidence.asset.encoder}
                </DataRow>
                <DataRow label="Retain until">{formatDateTime(event.evidence.asset.retain_until)}</DataRow>
                <DataRow label="Model bundle"><code>{event.model_bundle}</code></DataRow>
                <DataRow label="Policy">{event.policy_version}</DataRow>
              </dl>
            )}
          </div>
        </section>

        <section className="panel detail-panel">
          <div className="score-panel">
            <div>
              <span className="score-panel__label">Activity review score</span>
              <strong className="score-panel__value">Not scored</strong>
            </div>
            <IconShield />
          </div>
          <p className="score-explanation">
            The baseline policy does not calculate an activity probability. The stored score is
            <code> null</code>, not zero. No uncalibrated number is presented as risk.
          </p>
          <div className="review-history">
            <div className="history-heading">
              <h3>Review history</h3>
              <span>{event.reviews?.length ?? 0} record{(event.reviews?.length ?? 0) === 1 ? '' : 's'}</span>
            </div>
            {event.reviews?.length ? (
              <ol className="history-list">
                {[...event.reviews].reverse().map((review) => (
                  <li key={review.review_id}>
                    <div className="history-dot" aria-hidden="true" />
                    <div>
                      <div className="history-top">
                        <StatusPill tone={DECISION_TONE[review.decision]}>{reviewDecisionLabel(review.decision)}</StatusPill>
                        <time>{formatDateTime(review.created_at)}</time>
                      </div>
                      <p className="history-actor">{review.actor}</p>
                      {review.reason_codes.length > 0 && (
                        <p className="history-reasons">{review.reason_codes.map(reasonLabel).join(' · ')}</p>
                      )}
                      {review.notes && <p className="history-notes">{review.notes}</p>}
                      {review.supersedes_review_id && (
                        <p className="history-supersedes">Supersedes {review.supersedes_review_id.slice(0, 8)}…</p>
                      )}
                    </div>
                  </li>
                ))}
              </ol>
            ) : (
              <p className="muted-copy">No review decision has been recorded. The event remains in the queue.</p>
            )}
          </div>
          <ReviewComposer event={event} onRecorded={resource.reload} />
        </section>
      </div>

      {deleteOpen && (
        <Modal
          title="Delete this event and its evidence?"
          onClose={() => setDeleteOpen(false)}
          footer={
            <>
              <button type="button" className="button button--ghost" onClick={() => setDeleteOpen(false)}>Cancel</button>
              <button type="button" className="button button--danger" onClick={() => void deleteEvent()} disabled={deleting}>
                {deleting ? 'Deleting locally…' : 'Delete event'}
              </button>
            </>
          }
        >
          <p>
            This removes the local evidence clip and hides the event from the review queue. The audit record and
            review metadata remain as a tombstone. This action cannot be undone from the interface.
          </p>
          <p className="small text-muted">
            To preserve an independent review record, record a decision before deletion.
          </p>
          {Boolean(deleteError) && <InlineNotice tone="danger">{errorMessage(deleteError)}</InlineNotice>}
        </Modal>
      )}
    </div>
  )
}
