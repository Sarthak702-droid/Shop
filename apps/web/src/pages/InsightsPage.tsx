import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { formatBytes, formatDateTime, formatDuration, formatNumber, formatPercent, reasonLabel, actionVerdictLabel } from '../lib/format'
import { useApiResource } from '../lib/hooks'
import { ACTION_LABELS, type ActionLabel, type ResearchInsights } from '../lib/types'
import { ACTION_COLORS, Bars, ChartFrame, ClassDotPlot, Heatmap, Legend, LineChart } from '../components/Charts'
import { ErrorPanel, InlineNotice, LoadingPanel, MetricCard, PageHeader, SectionHeading, StatusPill } from '../components/Common'
import { IconAlert, IconCheck, IconCpu, IconDatabase, IconFilm, IconModel, IconShield } from '../components/Icons'

const SERIES = {
  val: { key: 'val', label: 'Validation', color: '#2a78d6' },
  test: { key: 'test', label: 'Test (held out)', color: '#eb6834' },
  chance: { key: 'chance', label: 'Chance', color: '#536572', shape: 'tick' as const },
}
const pct = (v: number | null | undefined, d = 0) => (v === null || v === undefined ? '—' : formatPercent(v, d))
const f3 = (v: number | null | undefined) => (v === null || v === undefined || !Number.isFinite(v) ? '—' : v.toFixed(3))

type Finding = { tone: 'teal' | 'review' | 'neutral'; title: string; body: string }

/** Turn the measured results into plain-language findings. Every number is read from the payload. */
function deriveFindings(data: ResearchInsights): Finding[] {
  const out: Finding[] = []
  const merl = data.merl
  if (merl.available && merl.test && merl.validation) {
    const f = merl.test.frame
    const lift = (f._map ?? 0) / Math.max(1e-6, f._chance_map ?? 1)
    const base = merl.baseline_no_context
    out.push({
      tone: 'teal',
      title: `The temporal model learns real signal: test frame mAP ${f3(f._map)} vs ${f3(f._chance_map)} chance (${lift.toFixed(1)}×)`,
      body: base
        ? `Twelve seconds of causal context beats the per-step baseline (frame mAP ${f3(base.frame_map)}, segment mAP ${f3(base.segment_map_iou_0_5)}): segment mAP rises to ${f3(merl.test.segment_eval_policy._map)}. Validation (${f3(merl.validation.frame._map)}) and test agree, so selection did not overfit.`
        : 'Validation and test agree closely, so model selection did not overfit.',
    })
    const ranked = [...ACTION_LABELS].sort((a, b) => f[b].ap - f[a].ap)
    out.push({
      tone: 'neutral',
      title: `${reasonLabel(ranked[0])} is easiest (AP ${f3(f[ranked[0]].ap)}); ${reasonLabel(ranked[4])} is hardest (AP ${f3(f[ranked[4]].ap)})`,
      body: `Segment localisation is the weak point: at temporal IoU 0.5 the best class reaches only ${f3(Math.max(...ACTION_LABELS.map((c) => merl.test!.segment_eval_policy[c].ap)))} segment AP. Many actions last under a second${merl.class_stats ? ` (median ${reasonLabel('reach_to_shelf').toLowerCase()} ${merl.class_stats.test.reach_to_shelf.median_s ?? '—'} s)` : ''}, so a 0.2 s boundary error costs a lot of IoU.`,
    })
  }
  if (merl.confusion_test) {
    const { labels, row_normalised: m } = merl.confusion_test
    let worst = { i: 0, j: 1, v: 0 }
    for (let i = 0; i < 5; i++) for (let j = 0; j < 6; j++) if (i !== j && m[i][j] > worst.v) worst = { i, j, v: m[i][j] }
    const fp = merl.failures_test
    out.push({
      tone: 'review',
      title: `Most errors are confusions between neighbouring actions, not invented activity`,
      body: `${pct(worst.v)} of ${reasonLabel(labels[worst.i]).toLowerCase()} steps are predicted as ${reasonLabel(labels[worst.j]).toLowerCase()}.${fp?.false_positive_with_other_gt_fraction != null ? ` ${pct(fp.false_positive_with_other_gt_fraction)} of false-positive segments overlap a different annotated action.` : ''} Reach → hand in shelf → retract is one continuous motion, so the label boundaries are the hard part.`,
    })
  }
  if (merl.visibility_test) {
    const v = merl.visibility_test
    const drops = ACTION_LABELS.map((c) => ({ c, d: (v.person_visible[c] ?? 0) - (v.person_not_detected[c] ?? 0) })).sort((a, b) => b.d - a.d)
    out.push({
      tone: 'review',
      title: `Person detection is the main failure driver: found in only ${pct(v.person_present_fraction)} of test steps`,
      body: `When no person is detected, ${reasonLabel(drops[0].c).toLowerCase()} AP falls from ${f3(v.person_visible[drops[0].c])} to ${f3(v.person_not_detected[drops[0].c])}. The COCO detector struggles with MERL's overhead view; a pose or overhead-trained person model is the highest-value improvement.`,
    })
  }
  if (merl.reliability_test) {
    const eces = ACTION_LABELS.map((c) => merl.reliability_test![c].ece)
    out.push({
      tone: 'neutral',
      title: `Scores are not probabilities (expected calibration error ${Math.min(...eces).toFixed(2)}–${Math.max(...eces).toFixed(2)})`,
      body: 'The app therefore shows them only as uncalibrated action scores against a validation threshold and keeps the review score null. Calibration (e.g. temperature scaling on validation) is future work.',
    })
  }
  if (data.cpu_benchmark) {
    const b = data.cpu_benchmark
    out.push({
      tone: 'teal',
      title: `Runs ${(1 / b.real_time_factor.mean).toFixed(1)}× faster than real time on this CPU`,
      body: `${b.host.cpu}, ${b.host.threads} threads: ${b.per_step_ms.detect_and_backbone.toFixed(0)} ms per 5 Hz step for the detector/backbone and ${b.per_step_ms.tcn.toFixed(3)} ms for the TCN; peak memory ${formatBytes(b.peak_rss_kb * 1024)}.${data.runtime_e2e?.all_parity ? ' Runtime features are bit-identical to the training features on held-out videos.' : ''}`,
    })
  }
  const app = data.app
  if (app.reviewer_precision.overall.n > 0) {
    const o = app.reviewer_precision.overall
    out.push({
      tone: o.n < 30 ? 'review' : 'teal',
      title: `Reviewers confirmed ${pct(o.estimate)} of predicted actions on uploaded footage (n = ${o.n})`,
      body: `95% interval ${pct(o.low)}–${pct(o.high)}.${o.n < 30 ? ' The sample is still small; treat this as an early signal, not a measured accuracy.' : ''} This is the only measure of transfer to footage outside MERL until the staged evaluation set exists.`,
    })
  } else if (app.events.total > 0) {
    out.push({ tone: 'review', title: 'No reviewer action feedback yet', body: `There are ${app.events.total} shelf-interaction events, but none has a verdict on the predicted actions. Review them to measure transfer to your footage.` })
  }
  const warn = Object.entries(app.job_warnings).filter(([k]) => k !== 'research_model_domain')
  if (warn.length) {
    out.push({
      tone: 'review',
      title: 'Uploaded footage differs from the training domain',
      body: warn.map(([k, n]) => `${reasonLabel(k)} in ${n} job${n > 1 ? 's' : ''}`).join('; ') + '. Results on these jobs are less reliable than MERL test numbers suggest.',
    })
  }
  const tr = data.merl_transfer
  if (tr) {
    const a = tr.conditions.as_uploaded
    out.push({
      tone: 'review',
      title: `Does not transfer to other cameras: clip AP ${f3(a.clip_ap_interaction)} vs ${f3(a.chance_ap)} chance on ${tr.samples * 2} RetailAction videos`,
      body: `Episodes are published for ${pct(a.detection_rate.none)} of no-action clips, about as often as for take/put/touch. Causes: ${tr.motion_diagnostic ? `${tr.motion_diagnostic.retailaction.frame_interval_s} s between frames (MERL 0.033 s), ` : ''}fisheye top-down views with rotated shoppers, and clips shorter than the 12 s context.`,
    })
  }
  const ra = data.retailaction
  if (ra.steps) {
    const usable = ra.data?.usable_positive_clips
    out.push({
      tone: 'neutral',
      title: `RetailAction: ${ra.steps.filter((s) => s.state === 'complete' || s.state === 'passed').length} of ${ra.steps.length} gates passed; not enabled in the app`,
      body: `${usable ? `Usable positive training clips: take ${formatNumber(usable.take)}, put ${formatNumber(usable.put)}, touch ${formatNumber(usable.touch)}, a ${Math.round((usable.take ?? 0) / Math.max(1, usable.touch ?? 1))}:1 imbalance. ` : ''}${data.pose_compatibility && !data.pose_compatibility.verdict.compatible ? 'The raw-video pose path is not compatible with its supplied poses yet (see below).' : ''}`,
    })
  }
  return out
}

function Gate({ state }: { state: string }) {
  const tone = state === 'complete' || state === 'passed' ? 'success' : state === 'failed' ? 'danger' : state === 'in_progress' || state === 'partial' ? 'review' : 'muted'
  return <StatusPill tone={tone} dot>{state.replaceAll('_', ' ')}</StatusPill>
}

export function InsightsPage() {
  const resource = useApiResource<ResearchInsights>('/research/insights', { pollMs: 30_000 })
  const [focus, setFocus] = useState<ActionLabel>('reach_to_shelf')
  const data = resource.data
  const findings = useMemo(() => (data ? deriveFindings(data) : []), [data])

  if (resource.loading && !data) return <LoadingPanel label="Loading research results…" rows={5} />
  if (resource.error && !data) return <ErrorPanel error={resource.error} onRetry={resource.reload} title="Research insights could not load" />
  if (!data) return null
  const merl = data.merl
  const app = data.app
  const test = merl.test
  const val = merl.validation
  const bench = data.cpu_benchmark
  const pose = data.pose_compatibility
  const ra = data.retailaction

  return (
    <div className="page-stack insights-page">
      <PageHeader
        eyebrow="Research mode"
        title="Research insights"
        subtitle="How well the shelf-action models work, where they fail, what they cost on this CPU, and how far the results transfer to your footage."
        actions={<Link className="button button--secondary button--small" to="/upload"><IconFilm /> Analyse a video</Link>}
      />
      <div className="provenance-line">
        <span className="provenance-tag">Actual measured result</span>
        <span>Offline results come from versioned files in <code>ml/results/</code> with the scripts that produced them; app figures are live aggregates of this host's database.</span>
      </div>

      {merl.available && test ? (
        <div className="job-stat-grid">
          <MetricCard label="Test frame mAP" value={f3(test.frame._map)} detail={`Chance ${f3(test.frame._chance_map)} · 28 held-out videos`} tone="teal" icon={<IconModel />} />
          <MetricCard label="Test segment mAP @ IoU 0.5" value={f3(test.segment_runtime_policy._map)} detail={`Runtime episode policy (evaluation policy ${f3(test.segment_eval_policy._map)})`} icon={<IconFilm />} />
          <MetricCard label="CPU speed" value={bench ? `${(1 / bench.real_time_factor.mean).toFixed(1)}× real time` : '—'} detail={bench ? `${bench.host.threads} threads · peak ${formatBytes(bench.peak_rss_kb * 1024)}` : 'Run ml/merl/runtime_e2e.py'} icon={<IconCpu />} />
          <MetricCard
            label="Reviewer-confirmed actions"
            value={app.reviewer_precision.overall.n ? pct(app.reviewer_precision.overall.estimate) : 'No feedback yet'}
            detail={app.reviewer_precision.overall.n ? `n = ${app.reviewer_precision.overall.n} · 95% CI ${pct(app.reviewer_precision.overall.low)}–${pct(app.reviewer_precision.overall.high)}` : 'Review shelf-interaction events to measure transfer'}
            tone={app.reviewer_precision.overall.n ? 'teal' : 'review'}
            icon={<IconCheck />}
          />
        </div>
      ) : (
        <InlineNotice tone="review" title="MERL analysis results are missing">{merl.reason}</InlineNotice>
      )}

      <section className="panel">
        <SectionHeading title="Key findings" detail="Generated from the measured results below; nothing here is hand-entered." />
        <div className="finding-grid">
          {findings.map((item, index) => (
            <article key={index} className={`finding finding--${item.tone}`}>
              <span className="finding__icon">{item.tone === 'review' ? <IconAlert /> : item.tone === 'teal' ? <IconCheck /> : <IconModel />}</span>
              <div>
                <h3>{item.title}</h3>
                <p>{item.body}</p>
              </div>
            </article>
          ))}
        </div>
      </section>

      {merl.available && test && val && (
        <>
          <section className="panel">
            <SectionHeading
              title="1 · MERL shelf-action model: per-class accuracy"
              detail={`Causal TCN, run ${merl.run}, best epoch ${merl.best_epoch}. Thresholds and episode policy were chosen on validation (subjects 21–26) and applied unchanged to test (subjects 27–41).`}
              action={<StatusPill tone="review">development_only</StatusPill>}
            />
            <div className="chart-grid">
              <ChartFrame
                title="Frame-level average precision"
                detail="Per 5 Hz step. The tick marks class prevalence (what a random scorer would get)."
                legend={<Legend items={[SERIES.val, SERIES.test, { ...SERIES.chance }]} />}
                table={
                  <table className="data-table">
                    <thead><tr><th>Class</th><th>Val AP</th><th>Test AP</th><th>Chance</th><th>Test P</th><th>Test R</th><th>Test F1</th><th>Threshold</th></tr></thead>
                    <tbody>{ACTION_LABELS.map((c) => (
                      <tr key={c}><td>{reasonLabel(c)}</td><td>{f3(val.frame[c].ap)}</td><td>{f3(test.frame[c].ap)}</td><td>{f3(test.frame[c].chance)}</td>
                        <td>{f3(test.frame[c].precision)}</td><td>{f3(test.frame[c].recall)}</td><td>{f3(test.frame[c].f1)}</td><td>{f3(test.frame[c].threshold)}</td></tr>
                    ))}</tbody>
                  </table>
                }
              >
                <ClassDotPlot
                  rows={ACTION_LABELS.map((c) => ({ key: c, label: reasonLabel(c), values: { val: val.frame[c].ap, test: test.frame[c].ap, chance: test.frame[c].chance ?? null } }))}
                  series={[SERIES.val, SERIES.test, SERIES.chance]}
                />
              </ChartFrame>
              <ChartFrame
                title="Segment detection on test (temporal IoU ≥ 0.5)"
                detail="Each predicted run of steps must overlap an annotated interval by half. Runtime policy drops 1-step blips and bridges 1-step gaps."
                legend={<Legend items={[{ key: 'e', label: 'Evaluation policy', color: '#6250d6' }, { key: 'r', label: 'Runtime policy (used in app)', color: '#1baf7a' }]} />}
                table={
                  <table className="data-table">
                    <thead><tr><th>Class</th><th>Seg AP (eval)</th><th>Seg AP (runtime)</th><th>Precision</th><th>Recall</th><th>F1</th><th>Predicted</th><th>Annotated</th></tr></thead>
                    <tbody>{ACTION_LABELS.map((c) => {
                      const r = test.segment_runtime_policy[c]
                      return <tr key={c}><td>{reasonLabel(c)}</td><td>{f3(test.segment_eval_policy[c].ap)}</td><td>{f3(r.ap)}</td><td>{f3(r.precision)}</td><td>{f3(r.recall)}</td><td>{f3(r.f1)}</td><td>{formatNumber(r.predicted)}</td><td>{formatNumber(r.ground_truth)}</td></tr>
                    })}</tbody>
                  </table>
                }
              >
                <ClassDotPlot
                  rows={ACTION_LABELS.map((c) => ({ key: c, label: reasonLabel(c), values: { e: test.segment_eval_policy[c].ap, r: test.segment_runtime_policy[c].ap } }))}
                  series={[{ key: 'e', label: 'Evaluation policy', color: '#6250d6' }, { key: 'r', label: 'Runtime policy', color: '#1baf7a' }]}
                  domain={[0, 0.6]}
                />
              </ChartFrame>
            </div>
            {merl.runtime_policy_selection && (
              <p className="small text-muted">
                Runtime policy grid on validation: {merl.runtime_policy_selection.grid.map((g) => `min ${g.min_segment_steps}/gap ${g.merge_gap_steps} → ${g.val_segment_map.toFixed(3)}`).join(' · ')}.
                Chosen: min {merl.runtime_policy_selection.chosen.min_segment_steps} steps, bridge {merl.runtime_policy_selection.chosen.merge_gap_steps}.
              </p>
            )}
          </section>

          <section className="panel">
            <SectionHeading title="2 · Training curves" detail="Training stopped early on validation mAP. Loss and mAP have different scales, so they are separate charts." />
            <div className="chart-grid">
              <ChartFrame title="Training loss (masked, class-weighted BCE)">
                <LineChart
                  series={[{ key: 'loss', label: 'Train loss', color: '#2a78d6', points: (merl.training_curve ?? []).map((r) => [r.epoch, r.train_loss]) }]}
                  xLabel="Epoch" yLabel="Loss" marker={merl.best_epoch ? { x: merl.best_epoch, label: 'Selected' } : undefined}
                />
              </ChartFrame>
              <ChartFrame
                title="Validation AP per class"
                legend={<Legend items={ACTION_LABELS.map((c) => ({ key: c, label: reasonLabel(c), color: ACTION_COLORS[c] }))} />}
                table={
                  <table className="data-table">
                    <thead><tr><th>Epoch</th><th>Loss</th><th>Val mAP</th>{ACTION_LABELS.map((c) => <th key={c}>{reasonLabel(c)}</th>)}</tr></thead>
                    <tbody>{(merl.training_curve ?? []).map((r) => (
                      <tr key={r.epoch}><td>{r.epoch}</td><td>{r.train_loss.toFixed(3)}</td><td>{r.val_map.toFixed(3)}</td>{ACTION_LABELS.map((c) => <td key={c}>{r.val_ap[c].toFixed(3)}</td>)}</tr>
                    ))}</tbody>
                  </table>
                }
              >
                <LineChart
                  series={ACTION_LABELS.map((c) => ({ key: c, label: reasonLabel(c), color: ACTION_COLORS[c], points: (merl.training_curve ?? []).map((r) => [r.epoch, r.val_ap[c]] as [number, number]) }))}
                  xLabel="Epoch" yLabel="Validation AP" yDomain={[0.2, 0.8]} marker={merl.best_epoch ? { x: merl.best_epoch, label: 'Selected' } : undefined}
                />
              </ChartFrame>
            </div>
          </section>

          <section className="panel">
            <SectionHeading title="3 · Error analysis" detail="Where the held-out test predictions go wrong, and why." />
            <div className="chart-grid">
              {merl.confusion_test && (
                <ChartFrame
                  title="Confusion between actions (test, row-normalised)"
                  detail={`Steps with at most one annotated action (${formatNumber(merl.confusion_test.excluded_multi_label_steps)} overlapping-label steps excluded). Predicted = strongest class above its threshold, else background.`}
                >
                  <Heatmap rows={merl.confusion_test.labels} cols={merl.confusion_test.labels} values={merl.confusion_test.row_normalised} counts={merl.confusion_test.counts} />
                </ChartFrame>
              )}
              {merl.visibility_test && (
                <ChartFrame
                  title="Effect of person detection on accuracy"
                  detail={`Test frame AP split by whether the person detector fired (person found in ${pct(merl.visibility_test.person_present_fraction)} of steps).`}
                  legend={<Legend items={[{ key: 'v', label: 'Person detected', color: '#2a78d6' }, { key: 'n', label: 'No person detected', color: '#eb6834' }]} />}
                >
                  <ClassDotPlot
                    rows={ACTION_LABELS.map((c) => ({ key: c, label: reasonLabel(c), values: { v: merl.visibility_test!.person_visible[c], n: merl.visibility_test!.person_not_detected[c] } }))}
                    series={[{ key: 'v', label: 'Person detected', color: '#2a78d6' }, { key: 'n', label: 'No person detected', color: '#eb6834' }]}
                  />
                </ChartFrame>
              )}
            </div>

            <div className="focus-row">
              <span>Inspect class:</span>
              {ACTION_LABELS.map((c) => (
                <button key={c} type="button" className={`chip chip--button${focus === c ? ' chip--selected' : ''}`} aria-pressed={focus === c} onClick={() => setFocus(c)}>
                  <span className="chart-legend__swatch" style={{ background: ACTION_COLORS[c] }} aria-hidden="true" />{reasonLabel(c)}
                </button>
              ))}
            </div>
            <div className="chart-grid">
              {merl.threshold_sweep_validation && (
                <ChartFrame
                  title={`Threshold trade-off · ${reasonLabel(focus)} (validation)`}
                  detail="Raising the threshold buys precision with recall. The dashed line is the frozen threshold used in the app."
                  legend={<Legend items={[{ key: 'p', label: 'Precision', color: '#2a78d6' }, { key: 'r', label: 'Recall', color: '#eb6834' }, { key: 'f', label: 'F1', color: '#1baf7a' }]} />}
                >
                  <LineChart
                    series={[
                      { key: 'p', label: 'Precision', color: '#2a78d6', points: merl.threshold_sweep_validation[focus].map((r) => [r.threshold * 100, r.precision]) },
                      { key: 'r', label: 'Recall', color: '#eb6834', points: merl.threshold_sweep_validation[focus].map((r) => [r.threshold * 100, r.recall]) },
                      { key: 'f', label: 'F1', color: '#1baf7a', points: merl.threshold_sweep_validation[focus].map((r) => [r.threshold * 100, r.f1]) },
                    ]}
                    xLabel="Threshold (×100)" yLabel="Value" yDomain={[0, 1]}
                    marker={{ x: Math.round((test.frame[focus].threshold ?? 0.5) * 100), label: 'Frozen' }}
                  />
                </ChartFrame>
              )}
              {merl.reliability_test && (
                <ChartFrame
                  title={`Calibration · ${reasonLabel(focus)} (test)`}
                  detail={`Mean score vs. observed positive rate per score bin. A calibrated model follows the diagonal. ECE ${merl.reliability_test[focus].ece.toFixed(3)}.`}
                  legend={<Legend items={[{ key: 'o', label: 'Observed positive rate', color: ACTION_COLORS[focus] }, { key: 'd', label: 'Perfect calibration', color: '#536572', dashed: true }]} />}
                  table={
                    <table className="data-table">
                      <thead><tr><th>Score bin</th><th>Mean score</th><th>Positive rate</th><th>Steps</th></tr></thead>
                      <tbody>{merl.reliability_test[focus].bins.map((b) => <tr key={b.bin}><td>{b.bin}</td><td>{b.mean_score.toFixed(3)}</td><td>{b.positive_rate.toFixed(3)}</td><td>{formatNumber(b.steps)}</td></tr>)}</tbody>
                    </table>
                  }
                >
                  <LineChart
                    series={[
                      { key: 'o', label: 'Observed', color: ACTION_COLORS[focus], points: merl.reliability_test[focus].bins.map((b) => [b.mean_score * 100, b.positive_rate]) },
                      { key: 'd', label: 'Diagonal', color: '#536572', dashed: true, points: [[0, 0], [100, 1]] },
                    ]}
                    xLabel="Mean score (×100)" yLabel="Positive rate" yDomain={[0, 1]}
                  />
                </ChartFrame>
              )}
            </div>

            {merl.failures_test && (
              <>
                <h3 className="detail-subheading">Failed examples (test)</h3>
                <p className="small text-muted">
                  {formatNumber(merl.failures_test.false_positive_count)} false-positive segments and {formatNumber(merl.failures_test.missed_count)} missed annotated segments under the runtime policy.
                  Open the local video (<code>Videos_MERL_Shopping_Dataset/</code>) at the listed time to inspect each case. The person-visible column shows how often the detector found the shopper.
                </p>
                <div className="table-scroll">
                  <table className="data-table">
                    <thead><tr><th>Type</th><th>Video</th><th>Class</th><th>Time</th><th>Score</th><th>Person visible</th><th>Likely cause</th></tr></thead>
                    <tbody>
                      {merl.failures_test.false_positives.map((r, i) => (
                        <tr key={`fp${i}`}>
                          <td><StatusPill tone="review">False positive</StatusPill></td><td>{r.video}</td><td>{reasonLabel(r.class)}</td><td>{r.start_s}–{r.end_s} s</td>
                          <td>{r.mean_score.toFixed(2)}</td><td>{pct(r.person_visible)}</td>
                          <td>{r.overlapping_gt.length ? `Label confusion: annotated ${r.overlapping_gt.map(reasonLabel).join(', ')}` : 'No action annotated here'}</td>
                        </tr>
                      ))}
                      {merl.failures_test.missed.map((r, i) => (
                        <tr key={`m${i}`}>
                          <td><StatusPill tone="muted">Missed</StatusPill></td><td>{r.video}</td><td>{reasonLabel(r.class)}</td><td>{r.start_s}–{r.end_s} s</td>
                          <td>max {r.max_score.toFixed(2)} / thr {r.threshold.toFixed(2)}</td><td>{pct(r.person_visible)}</td>
                          <td>{r.person_visible < 0.3 ? 'Person not detected' : r.max_score >= r.threshold ? 'Fragmented: score crossed threshold only briefly' : 'Score below threshold'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            )}

            {merl.per_video_test && merl.per_video_test.length > 0 && (
              <ChartFrame
                title="Per-video test frame mAP"
                detail="Spread across the 28 held-out recordings, ordered from weakest. Low-visibility videos cluster at the bottom."
                table={
                  <table className="data-table">
                    <thead><tr><th>Video</th><th>Frame mAP</th><th>Person visible</th><th>Steps</th></tr></thead>
                    <tbody>{merl.per_video_test.map((r) => <tr key={r.video}><td>{r.video}</td><td>{r.map.toFixed(3)}</td><td>{pct(r.person_visible)}</td><td>{formatNumber(r.steps)}</td></tr>)}</tbody>
                  </table>
                }
              >
                <Bars items={merl.per_video_test.map((r) => ({ label: r.video.replace('_crop.mp4', ''), value: Math.round(r.map * 100) }))} format={(v) => String(v)} />
              </ChartFrame>
            )}
          </section>
        </>
      )}

      <section className="panel">
        <SectionHeading title="4 · On this host: CPU cost and integration checks" detail="Measured on this machine through the same code path the worker runs." action={<span className="provenance-tag">Actual measured result</span>} />
        {bench ? (
          <div className="job-grid">
            <div>
              <h3 className="detail-subheading">Per-step processing time (5 Hz budget = {bench.budget.step_ms_at_5hz} ms)</h3>
              <div className="stack-bar" role="img" aria-label="Per-step CPU time">
                {Object.entries(bench.per_step_ms).map(([k, v], i) => (
                  <span key={k} style={{ width: `${(v / bench.budget.step_ms_at_5hz) * 100}%`, background: ['#86b6ef', '#2a78d6', '#1c5cab', '#0d366b'][i] }} title={`${k}: ${v} ms`} />
                ))}
              </div>
              <ul className="stack-legend">
                {Object.entries(bench.per_step_ms).map(([k, v], i) => (
                  <li key={k}><span className="chart-legend__swatch" style={{ background: ['#86b6ef', '#2a78d6', '#1c5cab', '#0d366b'][i] }} aria-hidden="true" />{reasonLabel(k)}<b>{v} ms</b></li>
                ))}
              </ul>
              <p className="small text-muted">{bench.host.cpu} · {bench.host.logical_cores} logical cores · {bench.host.threads} inference threads · torch {bench.host.torch}. {bench.note}</p>
            </div>
            {data.runtime_e2e && (
              <div>
                <h3 className="detail-subheading">Runtime path on held-out MERL test videos</h3>
                <div className="table-scroll">
                  <table className="data-table">
                    <thead><tr><th>Video</th><th>Length</th><th>Feature parity</th><th>Frame mAP</th><th>Episodes</th><th>RTF</th></tr></thead>
                    <tbody>{data.runtime_e2e.videos.map((v) => (
                      <tr key={v.video}><td>{v.video}</td><td>{formatDuration(v.media_seconds * 1000)}</td>
                        <td>{v.feature_parity_with_training ? <StatusPill tone="success">bit-exact</StatusPill> : <StatusPill tone="danger">differs</StatusPill>}</td>
                        <td>{v.frame_map.toFixed(3)}</td><td>{v.episodes} ({v.gt_intervals} annotated intervals)</td><td>{v.real_time_factor.toFixed(2)}</td></tr>
                    ))}</tbody>
                  </table>
                </div>
                <p className="small text-muted">{data.runtime_e2e.note}</p>
              </div>
            )}
          </div>
        ) : <p className="muted-copy">No benchmark recorded. Run <code>ml/merl/runtime_e2e.py</code>.</p>}
      </section>

      <section className="panel">
        <SectionHeading title="5 · Your footage: jobs and reviewer feedback" detail="Live from this host's database. Reviewer verdicts are the ground truth for transfer beyond MERL." />
        <div className="job-stat-grid">
          <MetricCard label="Research jobs" value={formatNumber(app.jobs.length)} detail={Object.entries(app.job_states).map(([k, v]) => `${v} ${k}`).join(' · ') || 'None yet'} icon={<IconDatabase />} />
          <MetricCard label="Footage analysed" value={`${app.analysed_minutes.toFixed(1)} min`} icon={<IconFilm />} />
          <MetricCard label="Interaction events" value={formatNumber(app.events.total)} detail={`${app.events.reviewed} reviewed · ${app.events.with_action_feedback} with action feedback`} icon={<IconCheck />} />
          <MetricCard label="Median episode" value={app.events.median_duration_ms ? formatDuration(app.events.median_duration_ms) : '—'} icon={<IconShield />} />
        </div>
        {app.events.with_action_feedback > 0 ? (
          <div className="chart-grid">
            <ChartFrame
              title="Reviewer-confirmed precision per action"
              detail="Share of predicted actions the reviewer also saw. Bars show the Wilson 95% interval; wide bars mean too few reviews."
              legend={<Legend items={[{ key: 'p', label: 'Confirmed share', color: '#087f78' }]} />}
              table={
                <table className="data-table">
                  <thead><tr><th>Action</th><th>Predicted events</th><th>Confirmed</th><th>Rejected</th><th>Missed (added by reviewer)</th><th>Precision (95% CI)</th></tr></thead>
                  <tbody>{ACTION_LABELS.map((c) => {
                    const r = app.reviewer_precision.per_class[c]
                    return <tr key={c}><td>{reasonLabel(c)}</td><td>{r.predicted_events}</td><td>{r.confirmed}</td><td>{r.rejected}</td><td>{r.reviewer_added}</td><td>{pct(r.precision.estimate)} ({pct(r.precision.low)}–{pct(r.precision.high)}, n={r.precision.n})</td></tr>
                  })}</tbody>
                </table>
              }
            >
              <ClassDotPlot
                rows={ACTION_LABELS.map((c) => {
                  const p = app.reviewer_precision.per_class[c].precision
                  return { key: c, label: `${reasonLabel(c)} (n=${p.n})`, values: { p: p.estimate }, interval: p.low !== null && p.high !== null ? [p.low, p.high] as [number, number] : null }
                })}
                series={[{ key: 'p', label: 'Confirmed share', color: '#087f78' }]}
                format={(v) => `${Math.round(v * 100)}%`}
              />
            </ChartFrame>
            <ChartFrame title="Reviewer verdicts on predicted actions">
              <Bars items={Object.entries(app.events.verdicts).map(([k, v]) => ({ label: actionVerdictLabel(k).split(' ')[0], value: v }))} color="#087f78" />
              <ul className="plain-list small">{Object.entries(app.events.verdicts).map(([k, v]) => <li key={k}>{actionVerdictLabel(k)}: {v}</li>)}</ul>
            </ChartFrame>
          </div>
        ) : (
          <InlineNotice tone="review" title="No reviewer feedback on predicted actions yet">
            Open a shelf-interaction event in the <Link to="/review">review queue</Link> and answer “Do the predicted actions match the clip?”. Each answer adds to the per-action precision estimate here.
          </InlineNotice>
        )}
        {app.events.total > 0 && (
          <ChartFrame title="Person visibility inside published episodes" detail="Episodes where the shopper is rarely detected are flagged for careful review.">
            <Bars items={app.events.person_visibility_histogram.map((b) => ({ label: b.bin, value: b.events }))} />
          </ChartFrame>
        )}
        {app.jobs.length > 0 && (
          <div className="table-scroll">
            <table className="data-table">
              <thead><tr><th>Job</th><th>File</th><th>Status</th><th>Length</th><th>Episodes</th><th>Person visible</th><th>RTF</th><th>Peak RSS</th><th>Warnings</th></tr></thead>
              <tbody>{app.jobs.map((j) => (
                <tr key={j.job_id}>
                  <td><Link className="text-link" to={`/jobs/${j.job_id}`}>{j.job_id.slice(0, 8)}</Link></td>
                  <td>{j.filename ?? '—'}</td>
                  <td>{j.status}{j.error ? ` · ${j.error.code}` : ''}</td>
                  <td>{j.analysed_seconds ? formatDuration(j.analysed_seconds * 1000) : '—'}</td>
                  <td>{j.episodes ?? '—'}</td>
                  <td>{pct(j.person_visible_fraction)}</td>
                  <td>{j.real_time_factor?.toFixed(2) ?? '—'}</td>
                  <td>{j.peak_rss_kb ? formatBytes(j.peak_rss_kb * 1024) : '—'}</td>
                  <td>{j.warnings.filter((w) => w !== 'research_model_domain').map(reasonLabel).join(', ') || 'none'}</td>
                </tr>
              ))}</tbody>
            </table>
          </div>
        )}
      </section>


      {data.merl_transfer && (() => {
        const t = data.merl_transfer
        const a = t.conditions.as_uploaded
        const c = t.conditions.time_corrected
        const groups = ['take', 'put', 'touch', 'none']
        const vis = t.breakdowns?.as_uploaded.by_person_visibility
        return (
          <section className="panel">
            <SectionHeading
              title="6a · Cross-camera test: MERL model on RetailAction views"
              detail={`${t.samples} RetailAction clips × 2 fisheye ceiling views = ${t.samples * 2} videos from other stores (take ${t.groups.take}, put ${t.groups.put}, touch ${t.groups.touch}, no action ${t.groups.none}). The MERL model never saw this data. Question: does Shop publish a reach / retract / hand-in-shelf episode when, and only when, an action happens?`}
              action={<StatusPill tone="danger">Does not transfer</StatusPill>}
            />
            <div className="job-stat-grid">
              <MetricCard label="Clip AP · as uploaded" value={f3(a.clip_ap_interaction)} detail={`Random scorer ${f3(a.chance_ap)}`} tone="review" icon={<IconAlert />} />
              <MetricCard label="Clip AP · time-corrected" value={f3(c.clip_ap_interaction)} detail="Frames re-timed to real 30 fps" tone="review" icon={<IconAlert />} />
              <MetricCard label="False episodes on no-action clips" value={pct(a.detection_rate.none)} detail={`${pct(c.detection_rate.none)} after re-timing`} tone="review" icon={<IconFilm />} />
              <MetricCard label="rank0 / rank1 agreement" value={pct(a.view_agreement)} detail="Same clip, two cameras, same answer" icon={<IconModel />} />
            </div>
            <div className="chart-grid">
              <ChartFrame
                title="How often an interaction episode is published"
                detail="A useful model is high for take/put/touch and low for no-action clips. Here every group looks alike."
                legend={<Legend items={[{ key: 'a', label: 'As uploaded (6 fps declared)', color: '#2a78d6' }, { key: 'c', label: 'Time-corrected (30 fps)', color: '#eb6834' }]} />}
                table={
                  <table className="data-table">
                    <thead><tr><th>Group</th><th>As uploaded</th><th>Time-corrected</th></tr></thead>
                    <tbody>{groups.map((g) => <tr key={g}><td>{g === 'none' ? 'no action' : g}</td><td>{pct(a.detection_rate[g])}</td><td>{pct(c.detection_rate[g])}</td></tr>)}</tbody>
                  </table>
                }
              >
                <ClassDotPlot
                  rows={groups.map((g) => ({ key: g, label: g === 'none' ? 'no action' : g, values: { a: a.detection_rate[g], c: c.detection_rate[g] } }))}
                  series={[{ key: 'a', label: 'As uploaded', color: '#2a78d6' }, { key: 'c', label: 'Time-corrected', color: '#eb6834' }]}
                  format={(v) => `${Math.round(v * 100)}%`}
                />
              </ChartFrame>
              <ChartFrame
                title="Distance from the training data (mean |z| per feature group)"
                detail="How far RetailAction features sit from MERL's training distribution, compared with MERL's own held-out test videos."
                legend={<Legend items={[{ key: 'm', label: 'MERL test', color: '#1baf7a' }, { key: 'a', label: 'RetailAction as uploaded', color: '#2a78d6' }, { key: 'c', label: 'RetailAction time-corrected', color: '#eb6834' }]} />}
              >
                <ClassDotPlot
                  rows={['roi', 'global', 'geom', 'motion'].map((k) => ({ key: k, label: { roi: 'Person region', global: 'Whole frame', geom: 'Person geometry', motion: 'Motion' }[k] ?? k, values: { m: t.merl_reference.feature_shift_z[k], a: a.feature_shift_z[k], c: c.feature_shift_z[k] } }))}
                  series={[{ key: 'm', label: 'MERL test', color: '#1baf7a' }, { key: 'a', label: 'As uploaded', color: '#2a78d6' }, { key: 'c', label: 'Time-corrected', color: '#eb6834' }]}
                  domain={[0, 3]}
                />
              </ChartFrame>
            </div>
            {t.findings && (
              <>
                <h3 className="detail-subheading">Where it fails</h3>
                <ol className="plain-list numbered">{t.findings.map((f, i) => <li key={i}>{f}</li>)}</ol>
              </>
            )}
            <div className="job-grid">
              {vis && (
                <div className="table-scroll">
                  <table className="data-table">
                    <thead><tr><th>Person visible (as uploaded)</th><th>Videos</th><th>Clip AP</th><th>Chance</th><th>False episodes, no-action</th></tr></thead>
                    <tbody>{Object.entries(vis).map(([k, b]) => (
                      <tr key={k}><td>{k.replace('person_visible_', '').replace('lt_', '< ').replace('ge_', '≥ ').replace('_', '–')}%</td><td>{b.clips}</td><td>{f3(b.clip_ap)}</td><td>{f3(b.chance)}</td><td>{pct(b.false_episode_rate_negative)}</td></tr>
                    ))}</tbody>
                  </table>
                </div>
              )}
              {t.motion_diagnostic && (
                <div className="table-scroll">
                  <table className="data-table">
                    <thead><tr><th>Motion signal</th><th>MERL test</th><th>RetailAction</th></tr></thead>
                    <tbody>
                      <tr><td>Time between frames</td><td>{t.motion_diagnostic.merl_test.frame_interval_s} s</td><td>{t.motion_diagnostic.retailaction.frame_interval_s} s</td></tr>
                      <tr><td>Whole-frame motion energy</td><td>{t.motion_diagnostic.merl_test.global_motion}</td><td>{t.motion_diagnostic.retailaction.global_motion}</td></tr>
                      <tr><td>Person / surround motion</td><td>{t.motion_diagnostic.merl_test.inside_over_ring.toFixed(1)}×</td><td>{t.motion_diagnostic.retailaction.inside_over_ring.toFixed(1)}×</td></tr>
                    </tbody>
                  </table>
                </div>
              )}
            </div>
            {t.examples && (
              <p className="small text-muted">
                Strongest false alarms (no-action clips): {t.examples.false_alarms.slice(0, 4).map((r) => `${r.sample_id}/${r.view} (score ${r.interaction_score.toFixed(2)}× threshold)`).join(', ')}.
                Weakest misses: {t.examples.missed.slice(0, 4).map((r) => `${r.sample_id}/${r.view} ${r.group} (${r.interaction_score.toFixed(2)}×, person ${pct(r.person_visible)})`).join(', ')}.
                Clips are in <code>var/research/retailaction_probe/clips/</code> (local only). Re-run with <code>ml/retailaction/merl_transfer.py</code>.
              </p>
            )}
            <InlineNotice tone="neutral" title="What this means">
              On new stores and camera angles, Shop's MERL model output is not evidence of a shelf interaction. The in-app domain warnings are therefore correct to flag such videos. Transfer needs training data from the target camera (own staged recordings, or RetailAction after the pose/feature gates pass), not just a different input format.
            </InlineNotice>
          </section>
        )
      })()}
      <section className="panel">
        <SectionHeading
          title="6 · RetailAction experiment (take / put / touch)"
          detail="A separate improvement experiment. The app does not depend on it and keeps it disabled until every gate passes."
          action={<StatusPill tone={ra.runtime_enabled ? 'success' : 'muted'}>{ra.runtime_enabled ? 'Enabled' : 'Not enabled'}</StatusPill>}
        />
        {ra.steps ? (
          <>
            <ol className="gate-list">
              {ra.steps.map((step, i) => (
                <li key={step.key}><span className="gate-list__index">{i + 1}</span><span>{step.label}</span><Gate state={step.state} /></li>
              ))}
            </ol>
            {ra.next_action && <InlineNotice tone="neutral" title="Next action">{ra.next_action}</InlineNotice>}
            <div className="chart-grid">
              {ra.data?.usable_positive_clips && (
                <ChartFrame title="Usable positive training clips per label" detail={ra.class_imbalance_note}>
                  <ClassDotPlot
                    rows={Object.entries(ra.data.usable_positive_clips).map(([k, v]) => ({ key: k, label: k, values: { n: Math.log10(Math.max(1, v)) } }))}
                    series={[{ key: 'n', label: 'Clips (log scale)', color: '#2a78d6' }]}
                    domain={[0, 5]}
                    format={(v) => formatNumber(Math.round(10 ** v))}
                  />
                  <p className="small text-muted">
                    Log scale. {Object.entries(ra.data.usable_positive_clips).map(([k, v]) => `${k}: ${formatNumber(v)}`).join(' · ')} of {formatNumber(ra.data.train_source_clips)} training clips;
                    {` ${ra.data.quarantined_metadata} quarantined for metadata errors, ${ra.data.no_valid_pose_excluded} without usable pose.`}
                  </p>
                </ChartFrame>
              )}
              {ra.archives && (
                <div>
                  <h3 className="detail-subheading">Archives on Drive</h3>
                  <ul className="plain-list">
                    {ra.archives.map((a) => (
                      <li key={a.name}>{a.verified ? <IconCheck /> : <IconAlert />} <code>{a.name}</code> · {formatBytes(a.bytes)} · {a.verified ? 'SHA-256 verified' : 'upload / verification pending'}</li>
                    ))}
                  </ul>
                  <p className="small text-muted">Snapshot {formatDateTime(ra.updated_utc)}. Refresh with <code>ml/retailaction/status_snapshot.py</code>.</p>
                </div>
              )}
            </div>
          </>
        ) : <p className="muted-copy">No RetailAction status snapshot.</p>}

        {pose && (
          <>
            <h3 className="detail-subheading">Raw video → pose → RetailAction features: compatibility gate</h3>
            <InlineNotice tone={pose.verdict.compatible ? 'success' : 'danger'} title={pose.verdict.compatible ? 'Compatible' : 'Not compatible: RetailAction stays disabled'}>
              <ul className="plain-list">{pose.verdict.blockers.map((b, i) => <li key={i}>{b}</li>)}</ul>
            </InlineNotice>
            <div className="job-grid">
              <div>
                <p className="small"><strong>Joint coverage with a COCO-17 pose model: {pct(pose.joint_mapping.coverage)}</strong></p>
                <div className="chip-row">
                  {pose.joint_mapping.direct.map((j) => <span key={j} className="chip chip--ok">{j}</span>)}
                  {Object.keys(pose.joint_mapping.derivable_as_midpoint).map((j) => <span key={j} className="chip chip--derived">{j} (derived)</span>)}
                  {pose.joint_mapping.missing_in_coco17.map((j) => <span key={j} className="chip chip--missing">{j} missing</span>)}
                </div>
                {pose.supplied_pose_semantics && (
                  <p className="small text-muted">
                    Supplied poses ({formatNumber(pose.supplied_pose_semantics.joint_observations)} joints from {pose.supplied_pose_semantics.samples_read} clips): {pct(pose.supplied_pose_semantics.fraction_on_1_64_grid, 1)} of coordinates lie on a 1/64 grid; joint scores range {pose.supplied_pose_semantics.joint_score.p0}–{pose.supplied_pose_semantics.joint_score.p100} ({pct(pose.supplied_pose_semantics.joint_score_above_1_fraction, 1)} above 1).
                  </p>
                )}
              </div>
              {pose.cpu_pose_timing && (
                <div>
                  <p className="small"><strong>CPU pose cost: {pose.cpu_pose_timing.mean_ms.toFixed(0)} ms per frame</strong> ({pose.cpu_pose_timing.max_fps} fps; {pose.cpu_pose_timing.input}, {pose.cpu_pose_timing.threads} threads, peak {formatBytes(pose.cpu_pose_timing.peak_rss_kb * 1024)})</p>
                  <div className="stack-bar" role="img" aria-label="Pose cost against the 200 ms budget">
                    <span style={{ width: `${Math.min(100, (200 / pose.cpu_pose_timing.mean_ms) * 100)}%`, background: '#087f78' }} title="200 ms budget" />
                    <span style={{ width: `${Math.max(0, 100 - (200 / pose.cpu_pose_timing.mean_ms) * 100)}%`, background: '#c53b42' }} title="Over budget" />
                  </div>
                  <p className="small text-muted">Green = the 200 ms per-step budget at 5 Hz; red = overrun. {pose.cpu_pose_timing.model}.</p>
                </div>
              )}
            </div>
            <h3 className="detail-subheading">Required before enabling</h3>
            <ol className="plain-list numbered">{pose.verdict.required_before_enabling.map((r, i) => <li key={i}>{r}</li>)}</ol>
          </>
        )}
      </section>

      <section className="panel">
        <SectionHeading title="7 · Own staged evaluation set" detail="Consented recordings, split by session and person, never used for training." />
        {data.own_evaluation.available ? (
          <pre className="code-block">{JSON.stringify(data.own_evaluation, null, 2)}</pre>
        ) : (
          <InlineNotice tone="review" title="Not recorded yet">
            {String(data.own_evaluation.reason ?? '')} Until it exists, MERL test numbers say nothing about other stores, camera angles or lighting.
          </InlineNotice>
        )}
      </section>

      <section className="panel">
        <SectionHeading title="Scope and limitations" />
        <ul className="plain-list limitations">
          <li>One mock store, one overhead camera, 41 actors (MERL). No other angle, store or lighting condition has measured accuracy.</li>
          <li>Five observable shelf actions only. No product detection, bag association or concealment; these are future work and the app never infers them.</li>
          <li>Scores are uncalibrated; thresholds come from MERL validation and may not suit other footage.</li>
          <li>Person detection (COCO SSDLite) misses the shopper in a large share of overhead frames; actions are then predicted from global motion only.</li>
          <li>MERL is licensed for research and education only; the checkpoint is <code>development_only</code> and not approved for commercial use.</li>
          <li>Every prediction goes to a human reviewer. Nothing here is evidence of theft, intent or guilt.</li>
        </ul>
      </section>
    </div>
  )
}
