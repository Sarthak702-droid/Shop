import { Link, useNavigate } from 'react-router-dom'
import { useApiResource } from '../lib/hooks'
import type {
  Capabilities,
  EventListResponse,
  MetricsResponse,
  ReadinessResponse,
  SourceListResponse,
} from '../lib/types'
import {
  formatBytes,
  formatDuration,
  formatNumber,
  formatPercent,
  workerLabel,
} from '../lib/format'
import {
  DataRow,
  EmptyState,
  ErrorPanel,
  InlineNotice,
  LoadingPanel,
  MetricCard,
  PageHeader,
  SectionHeading,
  StatusPill,
} from '../components/Common'
import { EventRow } from '../components/EventList'
import {
  IconAlert,
  IconArrowRight,
  IconCamera,
  IconCheck,
  IconCpu,
  IconDatabase,
  IconFile,
  IconQueue,
  IconShield,
  IconUpload,
} from '../components/Icons'

export function OverviewPage() {
  const navigate = useNavigate()
  const metrics = useApiResource<MetricsResponse>('/metrics', { pollMs: 15_000 })
  const readiness = useApiResource<ReadinessResponse>('/health/ready', { pollMs: 12_000 })
  const sources = useApiResource<SourceListResponse>('/sources')
  const capabilities = useApiResource<Capabilities>('/capabilities')
  const events = useApiResource<EventListResponse>('/events?limit=4', { pollMs: 20_000 })

  if (metrics.loading && !metrics.data) return <LoadingPanel label="Loading local overview…" rows={4} />
  if (metrics.error && !metrics.data) {
    return (
      <>
        <PageHeader title="Overview" subtitle="A local status view for observable activity review." />
        <ErrorPanel error={metrics.error} onRetry={metrics.reload} title="Overview could not load" />
      </>
    )
  }

  const summary = metrics.data?.summary
  const worker = metrics.data?.worker
  const personModel = capabilities.data?.models.person
  const pending = summary?.events_by_review_state.unreviewed ?? 0
  const reviewedToday = summary?.reviews_today ?? 0
  const registeredCameras = sources.data?.items.length ?? 0
  const approvedCameras = sources.data?.approved_cameras.length ?? 0
  const recentEvents = events.data?.items ?? []
  const isDegraded = readiness.data && !readiness.data.ready

  return (
    <div className="page-stack">
      <PageHeader
        eyebrow="Workspace"
        title="Overview"
        subtitle="Observable activity becomes a queue for human context review. Shop does not decide that theft occurred."
        actions={
          <button className="button button--secondary" type="button" onClick={() => navigate('/review')}>
            Open review queue <IconArrowRight />
          </button>
        }
      />

      <div className="provenance-line">
        <span className="provenance-tag">Actual measured result</span>
        <span>Counts and timing below come from this local database and host; no illustrative incidents are shown.</span>
      </div>

      {isDegraded && (
        <InlineNotice
          tone="review"
          title="Local readiness is degraded"
          action={<Link className="text-link" to="/settings">View dependencies</Link>}
        >
          Some analysis or evidence capabilities are unavailable. Existing review records remain accessible.
        </InlineNotice>
      )}

      <section className="metrics-grid" aria-label="Current local state">
        <MetricCard
          label="Awaiting review"
          value={pending}
          detail={pending === 0 ? 'No events currently need a decision.' : 'Human review is requested, not inferred guilt.'}
          tone={pending > 0 ? 'review' : 'neutral'}
          icon={<IconQueue />}
        />
        <MetricCard
          label="Reviews today"
          value={reviewedToday}
          detail="Append-only decisions recorded since local midnight."
          tone="teal"
          icon={<IconCheck />}
        />
        <MetricCard
          label="Registered cameras"
          value={`${formatNumber(registeredCameras)} / ${formatNumber(approvedCameras)}`}
          detail="Registered local sources / approved camera configs."
          icon={<IconCamera />}
        />
        <MetricCard
          label="Worker"
          value={<span className="metric-card__value-text">{worker ? workerLabel(worker.status) : 'Unknown'}</span>}
          detail={worker?.job_id ? `Active job ${worker.job_id.slice(0, 8)}…` : worker?.detail ?? 'Status unavailable.'}
          tone={worker?.status === 'idle' || worker?.status === 'running' ? 'teal' : 'review'}
          icon={<IconCpu />}
        />
      </section>

      <div className="overview-grid">
        <section className="panel panel--flush">
          <div className="panel-pad">
            <SectionHeading
              title="Recent activity segments"
              detail="A chronological list of descriptive segments. Each opens evidence context and review actions."
              action={
                <Link to="/review" className="text-link">
                  All events <IconArrowRight />
                </Link>
              }
            />
          </div>
          {events.loading && !events.data && <div className="panel-pad"><LoadingPanel rows={2} /></div>}
          {events.error && !events.data && (
            <div className="panel-pad"><ErrorPanel error={events.error} onRetry={events.reload} /></div>
          )}
          {events.data && recentEvents.length === 0 && (
            <EmptyState
              icon={<IconFilm />}
              title="No activity segments yet"
              action={
                <button className="button button--secondary" type="button" onClick={() => navigate('/upload')}>
                  <IconUpload /> Analyse footage
                </button>
              }
            >
              <p>
                A completed local analysis job creates segments only when a confirmed person track matches the
                baseline descriptive policy. An empty queue does not imply that no activity occurred.
              </p>
            </EmptyState>
          )}
          {recentEvents.length > 0 && (
            <div className="event-list event-list--flush">
              {recentEvents.map((event) => <EventRow event={event} key={event.event_id} />)}
            </div>
          )}
        </section>

        <aside className="overview-aside">
          <section className="panel">
            <SectionHeading
              title="System readiness"
              detail="Readiness checks read live components. A degraded check does not erase existing review data."
            />
            {readiness.loading && !readiness.data && <LoadingPanel rows={2} />}
            {readiness.error && !readiness.data && <ErrorPanel error={readiness.error} onRetry={readiness.reload} />}
            {readiness.data && (
              <div className="check-list">
                {readiness.data.checks.map((check) => (
                  <div className="check-row" key={check.name}>
                    <span className={`check-mark check-mark--${check.status}`} aria-hidden="true">
                      {check.status === 'pass' ? <IconCheck /> : <IconAlert />}
                    </span>
                    <div>
                      <div className="check-row__top">
                        <strong>{check.name.replaceAll('_', ' ')}</strong>
                        <StatusPill tone={check.status === 'pass' ? 'success' : 'review'}>
                          {check.status === 'pass' ? 'Pass' : 'Fail'}
                        </StatusPill>
                      </div>
                      <p>{check.detail}</p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </section>

          <section className="panel">
            <SectionHeading title="This build" detail="Capability boundaries are explicit, not hidden behind disabled controls." />
            <div className="capability-list">
              <div className="capability-row">
                <span className="capability-icon capability-icon--on"><IconCheck /></span>
                <div>
                  <strong>Person activity baseline</strong>
                  <p>{personModel?.available ? `${personModel.model_id} · ${personModel.version}` : 'Person detector unavailable'}</p>
                  <span className="mini-label">Actual local model status</span>
                </div>
              </div>
              <div className="capability-row">
                <span className="capability-icon capability-icon--off"><IconShield /></span>
                <div>
                  <strong>Retail object detection</strong>
                  <p>{capabilities.data?.models.retail.reason ?? 'No trained retail detector is enabled.'}</p>
                  <span className="mini-label">Module disabled</span>
                </div>
              </div>
              <div className="capability-row">
                <span className="capability-icon capability-icon--off"><IconShield /></span>
                <div>
                  <strong>Pose and temporal scoring</strong>
                  <p>{capabilities.data?.models.temporal.reason ?? 'No approved temporal model is enabled.'}</p>
                  <span className="mini-label">Module disabled</span>
                </div>
              </div>
            </div>
            <Link to="/models" className="panel-link">
              Inspect model manifests and rights <IconArrowRight />
            </Link>
          </section>
        </aside>
      </div>

      <div className="overview-grid overview-grid--lower">
        <section className="panel">
          <SectionHeading
            title="Local processing profile"
            detail="Starting values are proposed settings unless measured results are shown."
            action={<span className="provenance-tag">Proposed starting setting</span>}
          />
          <dl className="data-grid">
            <DataRow label="Device">{capabilities.data?.host.device ?? 'CPU'}</DataRow>
            <DataRow label="CPU">{capabilities.data?.host.cpu_model ?? 'Not reported'}</DataRow>
            <DataRow label="Cores">
              {capabilities.data
                ? `${capabilities.data.host.physical_cores ?? '—'} physical / ${capabilities.data.host.logical_cores ?? '—'} logical`
                : 'Not reported'}
            </DataRow>
            <DataRow label="Sampling">
              {capabilities.data?.limits.sampling_fps_max ? `${capabilities.data.limits.sampling_fps_max} fps maximum` : 'Not reported'}
            </DataRow>
            <DataRow label="Analysis long edge">
              {capabilities.data?.limits.long_edge_px ? `${capabilities.data.limits.long_edge_px} px` : 'Not reported'}
            </DataRow>
            <DataRow label="Queue">
              {metrics.data ? `${metrics.data.jobs.queue_depth} active or queued / ${metrics.data.jobs.active_limit} active limit` : '—'}
            </DataRow>
            <DataRow label="Free storage">
              {metrics.data ? `${formatBytes(metrics.data.storage.free_bytes)} (${formatPercent(metrics.data.storage.free_fraction)})` : '—'}
            </DataRow>
            <DataRow label="Evidence retention">
              {capabilities.data?.limits.evidence_retention_days ? `${capabilities.data.limits.evidence_retention_days} days` : '—'}
            </DataRow>
          </dl>
          <div className="profile-footnote">
            <IconDatabase />
            <span>
              Analysis runs in a separate local worker process. This interface never runs model inference in
              an HTTP request.
            </span>
          </div>
        </section>

        <section className="panel">
          <SectionHeading title="Start here" detail="The minimum path from a local source to a reviewed record." />
          <div className="start-list">
            <button className="start-action" type="button" onClick={() => navigate('/upload')}>
              <span className="start-action__icon"><IconFile /></span>
              <span><strong>Add a recording</strong><small>Validate locally, then queue CPU analysis.</small></span>
              <IconArrowRight />
            </button>
            <button className="start-action" type="button" onClick={() => navigate('/cameras')}>
              <span className="start-action__icon"><IconCamera /></span>
              <span><strong>Connect an approved camera</strong><small>Register a configured local device and check its preview.</small></span>
              <IconArrowRight />
            </button>
            <button className="start-action" type="button" onClick={() => navigate('/review')}>
              <span className="start-action__icon"><IconQueue /></span>
              <span><strong>Review with context</strong><small>Inspect evidence and record a benefit-of-context decision.</small></span>
              <IconArrowRight />
            </button>
          </div>
          <div className="privacy-note">
            <IconShield />
            <p>
              No cloud upload, no audio analysis, no identity recognition. Evidence remains inside the managed
              local storage directory until its retention deadline or an authorized deletion.
            </p>
          </div>
        </section>
      </div>
    </div>
  )
}

function IconFilm() {
  return <IconFile />
}
