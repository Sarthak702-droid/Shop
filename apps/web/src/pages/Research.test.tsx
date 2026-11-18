import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import analysis from '../../../../ml/results/merl_tcn_v1/analysis.json'
import pose from '../../../../ml/results/retailaction/pose_compatibility.json'
import retail from '../../../../ml/results/retailaction/status.json'
import bench from '../../../../ml/results/benchmarks/cpu_e2e.json'
import runtime from '../../../../ml/results/merl_tcn_v1/runtime_e2e.json'
import { EventDetailPanel } from '../components/EventDetailPanel'
import { EVENT_ID, JOB_ID, makeEvent, makeJob } from '../test/fixtures'
import { mockFetch, renderAt } from '../test/utils'
import type { ActionTimeline, ResearchInsights } from '../lib/types'
import { InsightsPage } from './InsightsPage'
import { JobPage } from './JobPage'

const CLASSES = ['reach_to_shelf', 'retract_from_shelf', 'hand_in_shelf', 'inspect_product', 'inspect_shelf'] as const
const emptyClass = { predicted_events: 0, confirmed: 0, rejected: 0, reviewer_added: 0, precision: { estimate: null, low: null, high: null, n: 0 } }

function insights(overrides: Partial<ResearchInsights['app']> = {}): ResearchInsights {
  return {
    merl: { ...(analysis as unknown as ResearchInsights['merl']), available: true, baseline_no_context: { run: 'linear', frame_map: 0.45, segment_map_iou_0_5: 0.124 } },
    retailaction: retail as unknown as ResearchInsights['retailaction'],
    pose_compatibility: pose as unknown as ResearchInsights['pose_compatibility'],
    own_evaluation: { available: false, reason: 'No staged own recordings have been annotated yet.' },
    cpu_benchmark: bench as unknown as ResearchInsights['cpu_benchmark'],
    runtime_e2e: runtime as unknown as ResearchInsights['runtime_e2e'],
    app: {
      jobs: [], job_states: {}, analysed_minutes: 0, job_warnings: {},
      events: { total: 0, reviewed: 0, with_action_feedback: 0, decisions: {}, verdicts: {}, warnings: {}, person_visibility_histogram: [], median_duration_ms: null },
      reviewer_precision: { per_class: Object.fromEntries(CLASSES.map((c) => [c, emptyClass])) as ResearchInsights['app']['reviewer_precision']['per_class'], overall: { estimate: null, low: null, high: null, n: 0 } },
      ...overrides,
    },
    action_model: { available: true, model_id: 'merl_action_tcn_v1', version: 'tcn', reason: '', commercial_status: 'development_only' },
  }
}

describe('Research insights page', () => {
  it('derives findings from measured results and never claims theft detection', async () => {
    mockFetch(() => ({ body: insights() }))
    renderAt(<InsightsPage />)
    expect(await screen.findByText('Key findings')).toBeInTheDocument()
    const testMap = (analysis as { test: { frame: { _map: number } } }).test.frame._map.toFixed(3)
    expect(screen.getAllByText(testMap).length).toBeGreaterThan(0)
    expect(screen.getByText(/Most errors are confusions between neighbouring actions/)).toBeInTheDocument()
    expect(screen.getByText(/Person detection is the main failure driver/)).toBeInTheDocument()
    expect(screen.getByText(/Not compatible: RetailAction stays disabled/)).toBeInTheDocument()
    expect(screen.getByText(/No reviewer feedback on predicted actions yet/)).toBeInTheDocument()
    expect(screen.getByRole('table', { name: 'Confusion matrix' })).toBeInTheDocument()
    expect(screen.queryByText(/theft detected|shoplifter|guilty/i)).not.toBeInTheDocument()
  })

  it('switches the focused class for threshold and calibration charts', async () => {
    mockFetch(() => ({ body: insights() }))
    renderAt(<InsightsPage />)
    await screen.findByText('Key findings')
    await userEvent.click(screen.getByRole('button', { name: /Inspect shelf/ }))
    expect(screen.getByText(/Threshold trade-off · Inspect shelf/)).toBeInTheDocument()
    expect(screen.getByText(/Calibration · Inspect shelf/)).toBeInTheDocument()
  })

  it('reports reviewer precision with its uncertainty', async () => {
    const perClass = Object.fromEntries(CLASSES.map((c) => [c, emptyClass])) as ResearchInsights['app']['reviewer_precision']['per_class']
    perClass.reach_to_shelf = { predicted_events: 4, confirmed: 3, rejected: 1, reviewer_added: 0, precision: { estimate: 0.75, low: 0.3, high: 0.95, n: 4 } }
    mockFetch(() => ({ body: insights({
      events: { total: 4, reviewed: 4, with_action_feedback: 4, decisions: { inconclusive: 4 }, verdicts: { correct: 3, incorrect: 1 }, warnings: {}, person_visibility_histogram: [], median_duration_ms: 3000 },
      reviewer_precision: { per_class: perClass, overall: { estimate: 0.75, low: 0.3, high: 0.95, n: 4 } },
    }) }))
    renderAt(<InsightsPage />)
    expect(await screen.findByText(/Reviewers confirmed 75% of predicted actions on uploaded footage \(n = 4\)/)).toBeInTheDocument()
    expect(screen.getByText(/early signal, not a measured accuracy/)).toBeInTheDocument()
  })
})

const shelfActions = [
  { label: 'reach_to_shelf' as const, start_ms: 1000, end_ms: 1800, mean_score: 0.81, peak_score: 0.9, threshold: 0.55 },
  { label: 'inspect_product' as const, start_ms: 1800, end_ms: 4000, mean_score: 0.77, peak_score: 0.93, threshold: 0.73 },
]

describe('Shelf-interaction review', () => {
  it('shows predicted actions and requires an action verdict before saving', async () => {
    const event = makeEvent({ event_type: 'shelf_interaction_observed', actions: shelfActions, media_interval_ms: { start: 1000, end: 4000 }, reasons: ['action:reach_to_shelf'] })
    const { calls } = mockFetch((url, init) => {
      if (init.method === 'POST') return { status: 201, body: { review: {}, event_id: EVENT_ID, review_state: 'benign' } }
      return { body: { event } }
    })
    renderAt(<EventDetailPanel eventId={EVENT_ID} />, { route: `/review/${EVENT_ID}`, path: '/review/:eventId' })
    expect(await screen.findByText('Shelf interaction observed')).toBeInTheDocument()
    expect(screen.getByText('Predicted shelf actions')).toBeInTheDocument()
    expect(screen.getAllByText('Inspect product').length).toBeGreaterThan(0)
    await userEvent.click(screen.getByRole('button', { name: /Save · Benign/ }))
    expect(await screen.findByText(/Say whether the predicted shelf actions match/)).toBeInTheDocument()
    expect(calls.filter((c) => c.init.method === 'POST')).toHaveLength(0)

    await userEvent.click(screen.getByLabelText(/Partly correct/))
    await userEvent.click(screen.getByRole('checkbox', { name: /Inspect product/ }))
    await userEvent.click(screen.getByRole('button', { name: /Save · Benign/ }))
    await waitFor(() => expect(calls.filter((c) => c.init.method === 'POST')).toHaveLength(1))
    const body = JSON.parse(String(calls.find((c) => c.init.method === 'POST')!.init.body))
    expect(body.action_feedback).toEqual({ verdict: 'partially_correct', observed_actions: ['reach_to_shelf'] })
  })
})

describe('Shelf-action job page', () => {
  it('renders the action timeline, warnings and per-action summary', async () => {
    const steps = 50
    const timeline: ActionTimeline = {
      schema: 'shop_action_timeline_v1',
      model: { model_id: 'merl_action_tcn_v1', run: 'tcn-x', sha256: 'a'.repeat(64), feature_schema: 'merl_features_v1' },
      classes: [...CLASSES],
      display: {},
      thresholds: [0.5, 0.5, 0.5, 0.5, 0.5],
      step_ms: 200,
      pts_ms: Array.from({ length: steps }, (_, i) => i * 200),
      scores: CLASSES.map((_, k) => Array.from({ length: steps }, (_, i) => (i > 10 && i < 20 && k === 0 ? 0.9 : 0.1))),
      person_present: Array.from({ length: steps }, (_, i) => (i < 30 ? 1 : 0)),
      episodes: [{ start_ms: 2200, end_ms: 4000, primary: 'reach_to_shelf', actions: [shelfActions[0]] }],
      policy: {},
      score_semantics: 'uncalibrated_action_score',
    }
    const perClass = Object.fromEntries(CLASSES.map((c) => [c, { segments: c === 'reach_to_shelf' ? 1 : 0, active_seconds: 1.8, active_fraction: 0.18, mean_score: 0.2, max_score: 0.9, per_minute: 6 }]))
    const job = makeJob({
      analysis_profile: 'merl_shelf_actions_research_v1',
      metrics: { real_time_factor: 0.35 },
      coverage: {
        action_analysis: {
          profile: 'merl_shelf_actions_research_v1',
          model: { model_id: 'merl_action_tcn_v1', run: 'tcn-x', sha256: 'a'.repeat(64), feature_schema: 'merl_features_v1', detector: 'ssdlite', policy: 'merl_action_episodes_v1', receptive_field_steps: 61 },
          input: { fps: 25, frames: 250, width: 1920, height: 1080, step_frames: 5 },
          warnings: [{ code: 'frame_rate_differs', severity: 'warning', message: 'Video is 25.00 fps.' }],
          summary: { steps, analysed_seconds: 10, episodes: 1, person_visible_fraction: 0.6, per_class: perClass, episodes_with_low_visibility: 0 },
          timeline: { path: 'timelines/x.json', sha256: 'b' },
          timing_ms: { decode: 100, detect_and_backbone: 2000, motion: 20, tcn: 2, evidence: 500, wall: 3000, per_step_feature_ms: 40.4 },
          steps_per_second: 16.7,
          label: 'actual_measured_result',
        },
      },
    })
    mockFetch((url) => {
      if (url.includes('/action-timeline')) return { body: { job_id: JOB_ID, timeline } }
      if (url.includes('/events')) return { body: { items: [], next_cursor: null, has_more: false, unreviewed_count: 0 } }
      return { body: { job } }
    })
    renderAt(<JobPage />, { route: `/jobs/${JOB_ID}`, path: '/jobs/:jobId' })
    expect(await screen.findByText('Shelf-action analysis')).toBeInTheDocument()
    expect(screen.getByText(/Frame rate differs from training/)).toBeInTheDocument()
    expect(await screen.findByRole('img', { name: /Shelf-action score timeline/ })).toBeInTheDocument()
    expect(screen.getByText('Per-action summary')).toBeInTheDocument()
    expect(screen.getByText('Interaction episodes from this job')).toBeInTheDocument()
  })
})
