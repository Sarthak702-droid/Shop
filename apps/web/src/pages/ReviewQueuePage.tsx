import { useEffect, useMemo, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { apiGet } from '../lib/api'
import { reasonLabel } from '../lib/format'
import { useApiResource } from '../lib/hooks'
import type { EventListResponse, EventSummary, MetricsResponse } from '../lib/types'
import {
  EmptyState,
  ErrorPanel,
  InlineNotice,
  LoadingPanel,
  MetricCard,
  PageHeader,
  StatusPill,
} from '../components/Common'
import { EventDetailPanel } from '../components/EventDetailPanel'
import { EventRow } from '../components/EventList'
import { IconCheck, IconFilm, IconQueue, IconSearch } from '../components/Icons'

type QueueFilter = 'unreviewed' | 'reviewed' | 'all'

const FILTERS: Array<{ value: QueueFilter; label: string }> = [
  { value: 'unreviewed', label: 'Needs review' },
  { value: 'reviewed', label: 'Reviewed' },
  { value: 'all', label: 'All segments' },
]

const PAGE_SIZE = 30

function filterPath(filter: QueueFilter): string {
  const params = new URLSearchParams({ limit: String(PAGE_SIZE) })
  if (filter === 'unreviewed') params.set('reviewed', 'false')
  if (filter === 'reviewed') params.set('reviewed', 'true')
  return `/events?${params.toString()}`
}

function matchesQuery(event: EventSummary, query: string): boolean {
  if (!query) return true
  const haystack = [
    event.source?.label ?? '',
    event.event_id,
    event.event_type,
    event.review_state,
    event.job_id,
    ...event.reasons,
    ...event.reasons.map(reasonLabel),
  ]
    .join(' ')
    .toLowerCase()
  return haystack.includes(query.toLowerCase())
}

export function ReviewQueuePage() {
  const params = useParams<{ eventId: string }>()
  const navigate = useNavigate()
  const [filter, setFilter] = useState<QueueFilter>(() => (params.eventId ? 'all' : 'unreviewed'))
  const [query, setQuery] = useState('')
  const [older, setOlder] = useState<EventSummary[]>([])
  const [nextCursor, setNextCursor] = useState<string | null>(null)
  const [loadingMore, setLoadingMore] = useState(false)
  const [loadMoreError, setLoadMoreError] = useState<unknown>(null)
  const path = filterPath(filter)
  const feed = useApiResource<EventListResponse>(path, { pollMs: 20_000 })
  const metrics = useApiResource<MetricsResponse>('/metrics', { pollMs: 20_000 })

  useEffect(() => {
    setOlder([])
    setNextCursor(null)
    setLoadMoreError(null)
  }, [path])

  useEffect(() => {
    const onChanged = () => feed.reload()
    window.addEventListener('shop-queue-changed', onChanged)
    return () => window.removeEventListener('shop-queue-changed', onChanged)
  }, [feed])

  const items = useMemo(() => {
    const seen = new Set<string>()
    const combined: EventSummary[] = []
    for (const event of [...(feed.data?.items ?? []), ...older]) {
      if (seen.has(event.event_id)) continue
      seen.add(event.event_id)
      combined.push(event)
    }
    return combined.filter((event) => matchesQuery(event, query.trim()))
  }, [feed.data, older, query])

  const availableCursor = nextCursor ?? feed.data?.next_cursor ?? null
  const pending = feed.data?.unreviewed_count ?? metrics.data?.summary.events_by_review_state.unreviewed ?? 0
  const reviewedToday = metrics.data?.summary.reviews_today ?? 0
  const totalEvents = metrics.data?.summary.events_total ?? 0
  const selectedId = params.eventId ?? null

  const loadMore = async () => {
    if (!availableCursor || loadingMore) return
    setLoadingMore(true)
    setLoadMoreError(null)
    try {
      const separator = path.includes('?') ? '&' : '?'
      const result = await apiGet<EventListResponse>(
        `${path}${separator}cursor=${encodeURIComponent(availableCursor)}`,
      )
      setOlder((current) => [...current, ...result.items])
      setNextCursor(result.next_cursor)
    } catch (caught) {
      setLoadMoreError(caught)
    } finally {
      setLoadingMore(false)
    }
  }

  return (
    <div className="page-stack queue-page">
      <PageHeader
        eyebrow="Human review"
        title="Review queue"
        subtitle="Review observable activity with context. Every record asks for a human decision; none states that theft occurred."
      />

      <div className="provenance-line">
        <span className="provenance-tag">Actual measured result</span>
        <span>Counts reflect events in the local SQLite database. No incidents are fabricated for demonstration.</span>
      </div>

      <section className="queue-stats" aria-label="Queue summary">
        <MetricCard
          label="Needs review"
          value={pending}
          detail="Events with no recorded review outcome."
          tone={pending > 0 ? 'review' : 'neutral'}
          icon={<IconQueue />}
        />
        <MetricCard
          label="Reviewed today"
          value={reviewedToday}
          detail="Append-only review decisions since local midnight."
          tone="teal"
          icon={<IconCheck />}
        />
        <MetricCard
          label="All segments"
          value={totalEvents}
          detail="Non-deleted descriptive activity segments."
          icon={<IconFilm />}
        />
      </section>

      <div className="queue-toolbar panel">
        <div className="filter-tabs" role="tablist" aria-label="Review state filter">
          {FILTERS.map((option) => (
            <button
              key={option.value}
              type="button"
              role="tab"
              aria-selected={filter === option.value}
              className={`filter-tab${filter === option.value ? ' filter-tab--active' : ''}`}
              onClick={() => setFilter(option.value)}
            >
              {option.label}
            </button>
          ))}
        </div>
        <div className="search-field">
          <IconSearch />
          <label className="sr-only" htmlFor="queue-search">Filter loaded events</label>
          <input
            id="queue-search"
            type="search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Filter loaded events by source, reason or ID"
          />
        </div>
        <p className="queue-toolbar__note">
          {query ? `${items.length} matching on this page` : 'Filters use the local API; search applies to loaded records.'}
        </p>
      </div>

      <div className={`queue-layout${selectedId ? ' queue-layout--detail' : ''}`}>
        <section className="panel panel--flush queue-list-panel" aria-label="Events awaiting review">
          {feed.loading && !feed.data && <div className="panel-pad"><LoadingPanel rows={4} /></div>}
          {feed.error && !feed.data && (
            <div className="panel-pad">
              <ErrorPanel error={feed.error} onRetry={feed.reload} title="Review queue could not load" />
            </div>
          )}
          {feed.data && items.length === 0 && (
            <EmptyState
              icon={<IconFilm />}
              title={query ? 'No matching loaded events' : filter === 'unreviewed' ? 'Nothing waiting for review' : 'No events in this view'}
            >
              <p>
                {query
                  ? 'Clear the filter or load more records. Search does not scan records that have not been fetched.'
                  : filter === 'unreviewed'
                    ? 'A completed analysis job creates a segment only when a confirmed person track matches the baseline descriptive policy. An empty queue does not imply that no activity occurred.'
                    : 'No records match this state yet. Completed local analysis jobs will appear here.'}
              </p>
              {query && (
                <button className="button button--secondary" type="button" onClick={() => setQuery('')}>
                  Clear filter
                </button>
              )}
            </EmptyState>
          )}
          {items.length > 0 && (
            <div className="event-list event-list--queue">
              {items.map((event) => (
                <EventRow
                  key={event.event_id}
                  event={event}
                  selected={selectedId === event.event_id}
                />
              ))}
            </div>
          )}
          {feed.data && (
            <div className="queue-list-footer">
              <span>
                Showing {items.length}
                {feed.data.has_more || older.length > 0 ? ' loaded' : ''} record{items.length === 1 ? '' : 's'}
                {query ? ' after filtering' : ''}.
              </span>
              {availableCursor && (
                <button
                  className="button button--secondary button--small"
                  type="button"
                  disabled={loadingMore}
                  onClick={() => void loadMore()}
                >
                  {loadingMore ? 'Loading older…' : 'Load older'}
                </button>
              )}
            </div>
          )}
          {Boolean(loadMoreError) && (
            <div className="panel-pad">
              <InlineNotice tone="danger" title="Older records could not load">
                <button type="button" className="text-button" onClick={() => void loadMore()}>Retry</button>
              </InlineNotice>
            </div>
          )}
        </section>

        <aside className="queue-detail" aria-label="Selected event detail">
          {selectedId ? (
            <EventDetailPanel
              key={selectedId}
              eventId={selectedId}
              onBack={() => navigate('/review')}
              onDeleted={() => feed.reload()}
            />
          ) : (
            <div className="queue-detail-empty panel" data-empty-detail>
              <EmptyState icon={<IconQueue />} title="Select a segment">
                <p>
                  The evidence player, quality context, score semantics and review actions appear here. Choose a row
                  from the queue to begin.
                </p>
                <div className="queue-detail-empty__note">
                  <StatusPill tone="muted">No automatic enforcement</StatusPill>
                  <StatusPill tone="muted">Local evidence only</StatusPill>
                </div>
              </EmptyState>
            </div>
          )}
        </aside>
      </div>
    </div>
  )
}
