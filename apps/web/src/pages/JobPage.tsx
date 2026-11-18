import { useEffect, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { apiPost, errorMessage, newIdempotencyKey } from '../lib/api'
import {
  formatBytes,
  formatDateTime,
  formatDuration,
  formatMediaTime,
  formatNumber,
  formatPercent,
} from '../lib/format'
import { useApiResource } from '../lib/hooks'
import type { EventListResponse, Job, JobResponse } from '../lib/types'
import {
  DataRow,
  EmptyState,
  ErrorPanel,
  InlineNotice,
  LoadingPanel,
  MetricCard,
  Modal,
  PageHeader,
  SectionHeading,
  StatusPill,
} from '../components/Common'
import { EventRow } from '../components/EventList'
import { ActionAnalysisPanel } from '../components/ActionAnalysisPanel'
import {
  IconAlert,
  IconCheck,
  IconClock,
  IconCpu,
  IconFilm,
  IconRefresh,
  IconUpload,
} from '../components/Icons'

const ACTIVE = new Set(['queued', 'running', 'cancelling'])

function numberValue(record: Record<string, unknown> | undefined, key: string): number | null {
  const value = record?.[key]
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function textValue(record: Record<string, unknown> | undefined, key: string): string | null {
  const value = record?.[key]
  return typeof value === 'string' ? value : null
}

function statusTone(status: Job['status']): 'teal' | 'success' | 'review' | 'danger' | 'muted' {
  if (status === 'completed') return 'success'
  if (status === 'running' || status === 'queued' || status === 'cancelling') return 'teal'
  if (status === 'failed') return 'danger'
  return 'muted'
}

function statusLabel(status: Job['status']): string {
  return {
    queued: 'Queued locally',
    running: 'Analysis running',
    cancelling: 'Cancellation requested',
    completed: 'Analysis completed',
    failed: 'Analysis failed',
    cancelled: 'Analysis cancelled',
  }[status]
}

function stageLabel(stage: string | null): string {
  if (!stage) return 'Waiting for worker'
  return {
    queued: 'Queued · waiting for worker',
    analysing: 'Analysing sampled frames',
    extracting_features: 'Extracting shelf-action features',
    temporal_inference: 'Running the temporal action model',
    writing_evidence: 'Writing evidence clips',
    live: 'Capturing from approved camera',
    finished: 'Finishing local records',
  }[stage] ?? stage.replaceAll('_', ' ')
}

function hasStage(metrics: Record<string, unknown>, key: string): boolean {
  const timing = metrics.stage_timing
  if (!timing || typeof timing !== 'object') return false
  const entry = (timing as Record<string, unknown>)[key]
  return Boolean(entry && typeof entry === 'object')
}

function stageTiming(metrics: Record<string, unknown>, key: string): Record<string, unknown> | null {
  const timing = metrics.stage_timing
  if (!timing || typeof timing !== 'object') return null
  const entry = (timing as Record<string, unknown>)[key]
  return entry && typeof entry === 'object' ? entry as Record<string, unknown> : null
}

export function JobPage() {
  const { jobId } = useParams<{ jobId: string }>()
  const navigate = useNavigate()
  const resource = useApiResource<JobResponse>(jobId ? `/jobs/${jobId}` : null, { pollMs: 2_500 })
  const [cancelOpen, setCancelOpen] = useState(false)
  const [cancelling, setCancelling] = useState(false)
  const [cancelError, setCancelError] = useState<unknown>(null)
  const [cancelMessage, setCancelMessage] = useState('')
  const job = resource.data?.job
  const eventsPath = job && !ACTIVE.has(job.status) ? `/events?job_id=${job.job_id}&limit=100` : null
  const events = useApiResource<EventListResponse>(eventsPath, { pollMs: 15_000 })

  useEffect(() => {
    if (!job) return
    if (job.status === 'completed' || job.status === 'failed' || job.status === 'cancelled') {
      setCancelOpen(false)
    }
  }, [job])

  if (!jobId) {
    return <ErrorPanel error={new Error('No job identifier was supplied.')} title="Job not found" />
  }
  if (resource.loading && !job) return <LoadingPanel label="Loading local analysis job…" rows={4} />
  if (resource.error && !job) {
    return <ErrorPanel error={resource.error} onRetry={resource.reload} title="Analysis job could not load" />
  }
  if (!job) {
    return (
      <EmptyState icon={<IconFilm />} title="Analysis job not found">
        <p>The local database has no record for this job identifier.</p>
        <Link to="/review" className="button button--secondary">Open review queue</Link>
      </EmptyState>
    )
  }

  const progress = job.progress
  const metrics = job.metrics
  const coverage = job.coverage
  const stage = textValue(progress, 'stage')
  const analysed = numberValue(progress, 'analysed_frames') ?? numberValue(metrics, 'analyzed_frames') ?? 0
  const expected = numberValue(progress, 'expected_samples') ?? numberValue(metrics, 'expected_samples')
  const duration = numberValue(progress, 'duration_ms') ?? numberValue(metrics, 'media_duration_ms')
  const mediaPts = numberValue(progress, 'media_pts_ms') ?? 0
  const covered = numberValue(progress, 'analysis_coverage')
  let fraction = 0
  if (job.status === 'completed') fraction = 1
  else if (expected && expected > 0) fraction = Math.min(1, analysed / expected)
  else if (duration && duration > 0) fraction = Math.min(1, mediaPts / duration)
  const active = ACTIVE.has(job.status)
  const detection = stageTiming(metrics, 'detect')
  const evidence = stageTiming(metrics, 'evidence')
  const sourceLabel = job.source.label ?? (job.kind === 'file' ? 'Local upload' : 'Approved camera')
  const eventItems = events.data?.items ?? []
  const actionJob = job.analysis_profile === 'merl_shelf_actions_research_v1'

  const requestCancel = async () => {
    setCancelling(true)
    setCancelError(null)
    try {
      const result = await apiPost<JobResponse & { message: string }>(
        `/jobs/${job.job_id}/cancel`,
        {},
        { 'Idempotency-Key': newIdempotencyKey() },
      )
      setCancelMessage(result.message ?? 'Cancellation requested.')
      setCancelOpen(false)
      resource.reload()
    } catch (caught) {
      setCancelError(caught)
    } finally {
      setCancelling(false)
    }
  }

  return (
    <div className="page-stack job-page">
      <PageHeader
        eyebrow={`Job ${job.job_id.slice(0, 8)}`}
        title={statusLabel(job.status)}
        subtitle={`${sourceLabel} · created ${formatDateTime(job.created_at)}. Processing happens in the local worker, not in this browser.`}
        actions={
          <>
            <button className="button button--ghost button--small" type="button" onClick={resource.reload} disabled={resource.refreshing}>
              <IconRefresh /> {resource.refreshing ? 'Refreshing…' : 'Refresh'}
            </button>
            {active && (
              <button className="button button--danger-ghost button--small" type="button" onClick={() => setCancelOpen(true)}>
                Cancel job
              </button>
            )}
            <button className="button button--secondary button--small" type="button" onClick={() => navigate('/upload')}>
              <IconUpload /> Add footage
            </button>
          </>
        }
      />

      <div className="provenance-line">
        <span className="provenance-tag">Actual measured result</span>
        <span>Status, progress and timing are read from this job's durable SQLite record and worker reports.</span>
      </div>

      {job.status === 'failed' && job.error && (
        <InlineNotice tone="danger" title="Analysis failed closed">
          <p><strong>{job.error.code}</strong> · {job.error.message}</p>
          <p className="small">
            No partial event is presented as complete. You can inspect the job record, adjust the source and retry.
          </p>
        </InlineNotice>
      )}
      {(job.status === 'cancelled' || job.status === 'cancelling') && (
        <InlineNotice tone="neutral" title={job.status === 'cancelled' ? 'Job cancelled' : 'Cancellation in progress'}>
          {cancelMessage || (job.status === 'cancelling'
            ? 'The worker settles the job after its current operation. No duplicate replay occurs.'
            : 'The job was cancelled before completion. Existing evidence clips are not presented as a finished analysis.')}
        </InlineNotice>
      )}
      {job.status === 'completed' && job.events_count === 0 && (
        <InlineNotice tone="neutral" title={actionJob ? 'Analysis completed with no shelf interaction above threshold' : 'Analysis completed with no activity segment'}>
          {actionJob
            ? 'No action score stayed above its validation threshold long enough to publish an episode. This does not prove that no interaction occurred; check the score timeline and warnings below.'
            : 'The baseline did not find a confirmed person track meeting the descriptive policy. This does not prove that no activity occurred; coverage and quality are reported below.'}
        </InlineNotice>
      )}
      {cancelMessage && job.status === 'cancelling' && (
        <InlineNotice tone="success" title="Cancellation recorded">{cancelMessage}</InlineNotice>
      )}

      <section className="panel job-progress-panel">
        <div className="job-progress-head">
          <div>
            <span className="eyebrow">Worker progress</span>
            <h2>{stageLabel(stage)}</h2>
          </div>
          <div className="job-progress-pills">
            <StatusPill tone={statusTone(job.status)} dot>{job.status}</StatusPill>
            {job.cancel_requested && <StatusPill tone="review">Cancel requested</StatusPill>}
            {job.retain_evidence && <StatusPill tone="muted">Evidence retained</StatusPill>}
          </div>
        </div>
        <div className="progress-block">
          <div className="progress-label">
            <span>
              {expected ? `${formatNumber(analysed)} of ${formatNumber(expected)} expected samples` : 'Sample count unavailable'}
              {duration ? ` · media ${formatMediaTime(mediaPts)} of ${formatMediaTime(duration)}` : ''}
            </span>
            <strong>{formatPercent(fraction, 0)}</strong>
          </div>
          <div className="progress-track progress-track--large" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(fraction * 100)}>
            <span className={`progress-fill${job.status === 'failed' ? ' progress-fill--danger' : ''}`} style={{ width: `${Math.max(1, fraction * 100)}%` }} />
          </div>
          <p className="small text-muted">
            {active
              ? 'This page polls the local API. Inference continues even if the page is closed.'
              : `Finished ${formatDateTime(job.finished_at)}.`}
          </p>
        </div>
        <div className="job-stat-grid">
          <MetricCard label="Analysed samples" value={formatNumber(analysed)} detail={expected ? `of ${formatNumber(expected)} expected` : 'Expected count not reported'} icon={<IconFilm />} />
          <MetricCard label="Analysis coverage" value={covered !== null ? formatPercent(covered, 1) : '—'} detail="Sampled frames with a valid analysis result." tone="teal" icon={<IconCheck />} />
          <MetricCard label="Dropped frames" value={formatNumber(numberValue(progress, 'dropped_frames') ?? 0)} detail="Scheduler or decode drops recorded by the source adapter." icon={<IconClock />} />
          <MetricCard label={actionJob ? 'Interaction episodes' : 'Activity segments'} value={formatNumber(job.events_count ?? 0)} detail={actionJob ? 'Shelf-interaction events for human review.' : 'Descriptive events created by the baseline policy.'} tone="teal" icon={<IconCpu />} />
        </div>
      </section>

      {actionJob && <ActionAnalysisPanel job={job} events={eventItems} />}

      <div className="job-grid">
        <section className="panel">
          <SectionHeading title="Measured timing" detail="Wall-clock and stage timings reported by the worker on this host." action={<span className="provenance-tag">Actual measured result</span>} />
          <dl className="data-grid data-grid--compact">
            <DataRow label="Wall time">{numberValue(metrics, 'wall_time_ms') !== null ? formatDuration(numberValue(metrics, 'wall_time_ms')) : 'Not reported'}</DataRow>
            <DataRow label="Media duration">{numberValue(metrics, 'media_duration_ms') !== null ? formatDuration(numberValue(metrics, 'media_duration_ms')) : 'Not reported'}</DataRow>
            <DataRow label="Valid analysis FPS">{numberValue(metrics, 'valid_analysis_fps') !== null ? numberValue(metrics, 'valid_analysis_fps')?.toFixed(2) : 'Not reported'}</DataRow>
            <DataRow label="Real-time factor">{numberValue(metrics, 'real_time_factor') !== null ? numberValue(metrics, 'real_time_factor')?.toFixed(3) : 'Not reported'}</DataRow>
            <DataRow label="Peak RSS">{numberValue(metrics, 'peak_rss_kb') !== null ? formatBytes((numberValue(metrics, 'peak_rss_kb') ?? 0) * 1024) : 'Not reported'}</DataRow>
            <DataRow label="Inference threads">{numberValue(metrics, 'inference_threads') ?? 'Not reported'}</DataRow>
            <DataRow label="Detection failures">{formatNumber(numberValue(metrics, 'detection_failures') ?? 0)}</DataRow>
            <DataRow label="Person detections">{formatNumber(numberValue(metrics, 'person_detections') ?? 0)}</DataRow>
          </dl>
          <h3 className="detail-subheading">Stage timing</h3>
          <div className="table-scroll">
            <table className="data-table">
              <thead>
                <tr><th>Stage</th><th>Mean</th><th>p50</th><th>p95</th><th>Samples</th></tr>
              </thead>
              <tbody>
                {['decode', 'preprocess', 'detect', 'track', 'evidence'].filter((key) => hasStage(metrics, key)).map((key) => {
                  const timing = stageTiming(metrics, key) ?? {}
                  return (
                    <tr key={key}>
                      <td>{key === 'evidence' ? 'evidence generation' : key}</td>
                      <td>{numberValue(timing, 'mean_ms')?.toFixed(1) ?? '—'} ms</td>
                      <td>{numberValue(timing, 'p50_ms')?.toFixed(1) ?? '—'} ms</td>
                      <td>{numberValue(timing, 'p95_ms')?.toFixed(1) ?? '—'} ms</td>
                      <td>{formatNumber(numberValue(timing, 'samples'))}</td>
                    </tr>
                  )
                })}
                {!['decode', 'preprocess', 'detect', 'track', 'evidence'].some((key) => hasStage(metrics, key)) && (
                  <tr><td colSpan={5}>No stage timing has been reported yet.</td></tr>
                )}
              </tbody>
            </table>
          </div>
          {evidence && (
            <p className="small text-muted">
              Evidence generation is a separate post-analysis pass; its duration is not inference throughput.
            </p>
          )}
        </section>

        <section className="panel">
          <SectionHeading title="Job contract" detail="Stored identifiers and limits applied to this analysis." />
          <dl className="data-grid data-grid--compact">
            <DataRow label="Job ID"><code className="breakable">{job.job_id}</code></DataRow>
            <DataRow label="Source">{sourceLabel}</DataRow>
            <DataRow label="Source kind">{job.kind}</DataRow>
            <DataRow label="Analysis profile"><code>{job.analysis_profile}</code></DataRow>
            <DataRow label="Sampling cadence">{job.sampling_fps} fps</DataRow>
            <DataRow label="Privacy mask">{job.privacy_mask_version}</DataRow>
            <DataRow label="Model bundle"><code className="breakable">{job.model_bundle}</code></DataRow>
            <DataRow label="Policy">{job.policy_version}</DataRow>
            <DataRow label="Content deletion">{job.retain_evidence ? 'Evidence retained to local deadline' : 'Evidence generation disabled'}</DataRow>
            <DataRow label="Created">{formatDateTime(job.created_at)}</DataRow>
            <DataRow label="Started">{formatDateTime(job.started_at)}</DataRow>
            <DataRow label="Finished">{formatDateTime(job.finished_at)}</DataRow>
          </dl>
          {job.error && (
            <InlineNotice tone="danger" title={`Error ${job.error.code}`}>{job.error.message}</InlineNotice>
          )}
        </section>
      </div>

      <div className="job-grid">
        <section className="panel">
          <SectionHeading title="Coverage and source integrity" detail="Whether the job actually analysed the media interval it claimed." />
          {Object.keys(coverage).length === 0 ? (
            <p className="muted-copy">Coverage is written when the worker settles the job.</p>
          ) : (
            <dl className="data-grid data-grid--compact">
              <DataRow label="Analysis coverage">{numberValue(coverage, 'analysis_coverage') !== null ? formatPercent(numberValue(coverage, 'analysis_coverage'), 1) : '—'}</DataRow>
              <DataRow label="Expected samples">{formatNumber(numberValue(coverage, 'expected_samples'))}</DataRow>
              <DataRow label="Analysed samples">{formatNumber(numberValue(coverage, 'analyzed_samples'))}</DataRow>
              <DataRow label="Missing samples">{formatNumber(numberValue(coverage, 'missing_samples'))}</DataRow>
              <DataRow label="Decode gaps">{formatNumber(numberValue(coverage, 'decode_gaps'))}</DataRow>
              <DataRow label="Decode incomplete">{coverage.decode_incomplete === true ? 'Yes' : 'No'}</DataRow>
              <DataRow label="Measured media span">{numberValue(coverage, 'measured_media_span_ms') !== null ? formatDuration(numberValue(coverage, 'measured_media_span_ms')) : '—'}</DataRow>
              <DataRow label="Declared duration">{numberValue(coverage, 'declared_duration_ms') !== null ? formatDuration(numberValue(coverage, 'declared_duration_ms')) : '—'}</DataRow>
            </dl>
          )}
          {Array.isArray(progress.notes) && progress.notes.length > 0 && (
            <InlineNotice tone="review" title="Source notes">
              <ul className="plain-list">
                {progress.notes.map((note, index) => <li key={index}>{String(note)}</li>)}
              </ul>
            </InlineNotice>
          )}
        </section>

        <section className="panel">
          <SectionHeading title="State transitions" detail="Append-only job state changes from the local store." />
          {job.transitions?.length ? (
            <ol className="transition-list">
              {job.transitions.map((transition, index) => (
                <li key={`${transition.created_at}-${index}`}>
                  <span className="transition-dot" aria-hidden="true" />
                  <div>
                    <strong>{transition.from_state ?? 'new'} → {transition.to_state}</strong>
                    <p>{transition.reason ?? 'No reason recorded'} · {formatDateTime(transition.created_at)}</p>
                  </div>
                </li>
              ))}
            </ol>
          ) : (
            <p className="muted-copy">No transitions have been recorded yet.</p>
          )}
        </section>
      </div>

      <section className="panel">
        <SectionHeading
          title={actionJob ? 'Interaction episodes from this job' : 'Activity segments from this job'}
          detail={actionJob ? 'Each episode links to its evidence clip, predicted actions and a human review form.' : 'Only descriptive segments; each links to evidence and a human review outcome.'}
          action={
            (job.events_count ?? 0) > 0 ? (
              <Link to={`/review`} className="text-link">Open review queue</Link>
            ) : undefined
          }
        />
        {events.loading && !events.data && <LoadingPanel rows={2} />}
        {events.error && !events.data && <ErrorPanel error={events.error} onRetry={events.reload} title="Job events could not load" />}
        {events.data && eventItems.length === 0 && (
          <EmptyState icon={<IconFilm />} title="No activity segments from this job">
            <p>
              {job.status === 'completed'
                ? 'The baseline found no confirmed person track matching the descriptive policy. Absence of an event is not proof that no activity occurred.'
                : 'Segments appear only after the worker settles the job.'}
            </p>
          </EmptyState>
        )}
        {eventItems.length > 0 && (
          <div className="event-list">
            {eventItems.map((event) => <EventRow key={event.event_id} event={event} />)}
          </div>
        )}
      </section>

      {cancelOpen && (
        <Modal
          title="Cancel this local analysis job?"
          onClose={() => setCancelOpen(false)}
          footer={
            <>
              <button className="button button--ghost" type="button" onClick={() => setCancelOpen(false)}>Keep analysing</button>
              <button className="button button--danger" type="button" onClick={() => void requestCancel()} disabled={cancelling}>
                {cancelling ? 'Requesting cancellation…' : 'Cancel job'}
              </button>
            </>
          }
        >
          <p>
            Cancellation is durable and idempotent. A queued job is cancelled immediately; a running job settles after
            the worker's current operation. The interrupted job fails closed rather than replaying duplicate events.
          </p>
          {Boolean(cancelError) && <InlineNotice tone="danger">{errorMessage(cancelError)}</InlineNotice>}
        </Modal>
      )}
    </div>
  )
}
