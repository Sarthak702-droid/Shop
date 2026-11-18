import { useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import { formatDuration, formatMediaTime, formatNumber, formatPercent, reasonLabel } from '../lib/format'
import { useApiResource } from '../lib/hooks'
import type { ActionAnalysis, ActionLabel, ActionTimelineResponse, EventSummary, Job } from '../lib/types'
import { ACTION_COLORS, ActionTimelineChart, ChartFrame, Legend } from './Charts'
import { DataRow, ErrorPanel, InlineNotice, LoadingPanel, MetricCard, SectionHeading, StatusPill } from './Common'
import { IconAlert, IconCpu, IconEyeOff, IconFilm } from './Icons'

export function actionAnalysisOf(job: Job): ActionAnalysis | null {
  const value = (job.coverage as Record<string, unknown>)?.action_analysis
  return value && typeof value === 'object' ? (value as ActionAnalysis) : null
}

/** Research-mode results for one shelf-action job: what was observed, when, how reliable, at what cost. */
export function ActionAnalysisPanel({ job, events }: { job: Job; events: EventSummary[] }) {
  const navigate = useNavigate()
  const analysis = actionAnalysisOf(job)
  const timeline = useApiResource<ActionTimelineResponse>(
    job.status === 'completed' && analysis ? `/jobs/${job.job_id}/action-timeline` : null,
  )
  const byStart = useMemo(() => [...events].sort((a, b) => a.media_interval_ms.start - b.media_interval_ms.start), [events])

  if (!analysis) {
    if (job.status === 'failed') return null
    return (
      <section className="panel">
        <SectionHeading title="Shelf-action analysis" detail="Research mode · MERL causal TCN" />
        <p className="muted-copy">Feature extraction and action inference results appear when the worker settles the job.</p>
      </section>
    )
  }

  const { summary, warnings, model, input, timing_ms: timing } = analysis
  const classes = Object.keys(summary.per_class) as ActionLabel[]
  const featureMs = (timing.detect_and_backbone ?? 0) + (timing.motion ?? 0) + (timing.decode ?? 0)
  const timingParts = [
    { key: 'decode', label: 'Decode', ms: timing.decode ?? 0, color: '#86b6ef' },
    { key: 'detect', label: 'Detector + backbone', ms: timing.detect_and_backbone ?? 0, color: '#2a78d6' },
    { key: 'motion', label: 'Motion features', ms: timing.motion ?? 0, color: '#1c5cab' },
    { key: 'tcn', label: 'TCN', ms: timing.tcn ?? 0, color: '#0d366b' },
    { key: 'evidence', label: 'Evidence clips', ms: timing.evidence ?? 0, color: '#9aa9b2' },
  ]
  const timingTotal = Math.max(1, timingParts.reduce((s, p) => s + p.ms, 0))
  const domainWarnings = warnings.filter((w) => w.severity === 'warning')

  const openNearest = (ms: number) => {
    if (!byStart.length) return
    const containing = byStart.find((e) => e.media_interval_ms.start <= ms && ms <= e.media_interval_ms.end)
    const nearest = containing ?? byStart.reduce((best, e) => {
      const d = Math.min(Math.abs(e.media_interval_ms.start - ms), Math.abs(e.media_interval_ms.end - ms))
      const bd = Math.min(Math.abs(best.media_interval_ms.start - ms), Math.abs(best.media_interval_ms.end - ms))
      return d < bd ? e : best
    }, byStart[0])
    navigate(`/review/${nearest.event_id}`)
  }

  return (
    <section className="panel action-panel">
      <SectionHeading
        title="Shelf-action analysis"
        detail="What the research model observed in this recording, when, and how much to trust it."
        action={<StatusPill tone="review">Research mode</StatusPill>}
      />

      <div className="job-stat-grid">
        <MetricCard label="Interaction episodes" value={formatNumber(summary.episodes)} detail={`${formatNumber(events.length)} published for review`} tone="teal" icon={<IconFilm />} />
        <MetricCard label="Person visible" value={formatPercent(summary.person_visible_fraction, 0)} detail="Steps where the detector found a person (score ≥ 0.15)." tone={summary.person_visible_fraction < 0.3 ? 'review' : 'neutral'} icon={<IconEyeOff />} />
        <MetricCard label="Analysed" value={formatDuration(summary.analysed_seconds * 1000)} detail={`${formatNumber(summary.steps)} steps at 5 Hz`} icon={<IconCpu />} />
        <MetricCard label="Processing speed" value={`${analysis.steps_per_second.toFixed(1)} steps/s`} detail={`Real-time factor ${Number(job.metrics.real_time_factor ?? 0).toFixed(2)} (below 1 = faster than real time)`} icon={<IconCpu />} />
      </div>

      {domainWarnings.length > 0 ? (
        <InlineNotice tone="review" title={`${domainWarnings.length} input warning${domainWarnings.length > 1 ? 's' : ''}: results may not transfer`}>
          <ul className="plain-list">
            {domainWarnings.map((w) => <li key={w.code}><strong>{reasonLabel(w.code)}.</strong> {w.message}</li>)}
          </ul>
        </InlineNotice>
      ) : (
        <InlineNotice tone="neutral" title="Input matches the training format">
          Frame rate, aspect ratio and person visibility are within the training range. The model is still unvalidated outside MERL's single overhead view.
        </InlineNotice>
      )}

      <ChartFrame
        title="Action score timeline"
        detail="Each lane is one action. Filled curve = uncalibrated score; dashed line = validation threshold; bar above the lane = published segment. Grey in the top strip = no person detected."
        legend={<Legend items={[
          ...classes.map((c) => ({ key: c, label: reasonLabel(c), color: ACTION_COLORS[c] })),
          { key: 'thr', label: 'Threshold', color: '#536572', dashed: true },
        ]} />}
        table={timeline.data ? (
          <table className="data-table">
            <thead><tr><th>Episode</th><th>Start</th><th>End</th><th>Primary action</th><th>Actions in episode</th></tr></thead>
            <tbody>
              {timeline.data.timeline.episodes.map((ep, i) => (
                <tr key={i}>
                  <td>{i + 1}</td><td>{formatMediaTime(ep.start_ms)}</td><td>{formatMediaTime(ep.end_ms)}</td>
                  <td>{reasonLabel(ep.primary)}</td>
                  <td>{ep.actions.map((a) => `${reasonLabel(a.label)} (${a.mean_score.toFixed(2)})`).join(', ')}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : undefined}
      >
        {timeline.loading && <LoadingPanel label="Loading per-step scores…" rows={2} />}
        {timeline.error && <ErrorPanel error={timeline.error} onRetry={timeline.reload} title="Timeline could not load" />}
        {timeline.data && <ActionTimelineChart timeline={timeline.data.timeline} onSelectTime={events.length ? openNearest : undefined} />}
      </ChartFrame>

      <div className="job-grid">
        <div>
          <h3 className="detail-subheading">Per-action summary</h3>
          <div className="table-scroll">
            <table className="data-table">
              <thead>
                <tr><th>Action</th><th>Segments</th><th>Per min</th><th>Time above threshold</th><th>Mean score</th><th>Peak</th></tr>
              </thead>
              <tbody>
                {classes.map((c) => {
                  const row = summary.per_class[c]
                  return (
                    <tr key={c}>
                      <td><span className="chart-legend__swatch" style={{ background: ACTION_COLORS[c] }} aria-hidden="true" /> {reasonLabel(c)}</td>
                      <td>{row.segments}</td>
                      <td>{row.per_minute.toFixed(1)}</td>
                      <td>{row.active_seconds.toFixed(1)} s ({formatPercent(row.active_fraction, 0)})</td>
                      <td>{row.mean_score.toFixed(2)}</td>
                      <td>{row.max_score.toFixed(2)}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
          {summary.episodes_with_low_visibility > 0 && (
            <p className="small text-muted">
              <IconAlert /> {summary.episodes_with_low_visibility} of {summary.episodes} episodes had a person visible for less than 40% of their duration; review these with extra care.
            </p>
          )}
        </div>
        <div>
          <h3 className="detail-subheading">Where CPU time went</h3>
          <div className="stack-bar" role="img" aria-label="Processing time by stage">
            {timingParts.filter((p) => p.ms > 0).map((p) => (
              <span key={p.key} style={{ width: `${(p.ms / timingTotal) * 100}%`, background: p.color }} title={`${p.label}: ${formatDuration(p.ms)}`} />
            ))}
          </div>
          <ul className="stack-legend">
            {timingParts.map((p) => (
              <li key={p.key}>
                <span className="chart-legend__swatch" style={{ background: p.color }} aria-hidden="true" />
                {p.label}<b>{formatDuration(p.ms)}</b><small>{formatPercent(p.ms / timingTotal, 0)}</small>
              </li>
            ))}
          </ul>
          <p className="small text-muted">Feature extraction {formatDuration(featureMs)} · {timing.per_step_feature_ms?.toFixed(1)} ms per step against a 200 ms budget at 5 Hz.</p>
        </div>
      </div>

      <dl className="data-grid data-grid--compact">
        <DataRow label="Model">{model.model_id} · run <code>{model.run}</code></DataRow>
        <DataRow label="Checkpoint SHA-256"><code className="hash-value">{model.sha256}</code></DataRow>
        <DataRow label="Feature schema">{model.feature_schema} · detector {model.detector}</DataRow>
        <DataRow label="Temporal context">{model.receptive_field_steps} steps ({(model.receptive_field_steps / 5).toFixed(0)} s), causal</DataRow>
        <DataRow label="Episode policy">{model.policy} (chosen on MERL validation)</DataRow>
        <DataRow label="Input">{input.width}×{input.height} · {input.fps} fps · {formatNumber(input.frames)} frames · {input.step_frames} frames per step</DataRow>
      </dl>
    </section>
  )
}
