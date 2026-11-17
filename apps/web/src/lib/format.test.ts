import { describe, expect, it } from 'vitest'
import {
  clamp,
  eventTypeLabel,
  evidenceStatusLabel,
  formatBytes,
  formatDuration,
  formatMediaTime,
  formatNumber,
  formatPercent,
  formatShortDate,
  formatTime,
  reasonLabel,
  reviewDecisionLabel,
  reviewStateLabel,
  workerLabel,
} from '../lib/format'

describe('format helpers', () => {
  it('formats byte sizes with unit rollover', () => {
    expect(formatBytes(null)).toBe('—')
    expect(formatBytes(Number.NaN)).toBe('—')
    expect(formatBytes(512)).toBe('512 B')
    expect(formatBytes(1536)).toBe('1.5 KB')
    expect(formatBytes(10 * 1024 * 1024)).toBe('10 MB')
    expect(formatBytes(2 * 1024 ** 3)).toBe('2.0 GB')
  })

  it('formats elapsed durations with seconds, minutes and hours', () => {
    expect(formatDuration(null)).toBe('—')
    expect(formatDuration(-5)).toBe('—')
    expect(formatDuration(1250)).toBe('1.3s')
    expect(formatDuration(8300)).toBe('8.3s')
    expect(formatDuration(65_000)).toBe('1m 05s')
    expect(formatDuration(3_600_000 + 25 * 60_000)).toBe('1h 25m')
  })

  it('formats media offsets as minutes and zero-padded seconds', () => {
    expect(formatMediaTime(undefined)).toBe('—')
    expect(formatMediaTime(4000)).toBe('0:04')
    expect(formatMediaTime(125_000)).toBe('2:05')
  })

  it('formats percentages and numbers without pretending missing data is zero', () => {
    expect(formatPercent(null)).toBe('—')
    expect(formatPercent(0)).toBe('0%')
    expect(formatPercent(0.825, 1)).toBe('82.5%')
    expect(formatNumber(null)).toBe('—')
    expect(formatNumber(1234)).toBe('1,234')
  })

  it('formats invalid timestamps as unavailable rather than invalid dates', () => {
    expect(formatTime(null)).toBe('—')
    expect(formatTime('not-a-date')).toBe('—')
    expect(formatShortDate(undefined)).toBe('—')
    expect(formatShortDate('not-a-date')).toBe('—')
  })

  it('maps reason codes to calm review language', () => {
    expect(reasonLabel('person_track_observed')).toBe('Person track observed')
    expect(reasonLabel('item_visibility_lost')).toBe('Item visibility lost')
    expect(reasonLabel('activity_benign')).toBe('Activity appears benign in context')
    expect(reasonLabel('unmapped_reason')).toBe('unmapped reason')
  })

  it('labels event types without claiming a conclusion', () => {
    expect(eventTypeLabel('person_activity_observed')).toBe('Person activity observed')
    expect(eventTypeLabel('some_future_event')).toBe('some future event')
  })

  it('labels review outcomes separately from unreviewed state', () => {
    expect(reviewStateLabel('unreviewed')).toBe('Unreviewed')
    expect(reviewStateLabel('review_needed')).toBe('Reviewed · needs review')
    expect(reviewStateLabel('inconclusive')).toBe('Reviewed · inconclusive')
    expect(reviewDecisionLabel('benign')).toBe('Benign')
    expect(reviewDecisionLabel('review_needed')).toBe('Needs review')
  })

  it('labels evidence and worker states', () => {
    expect(evidenceStatusLabel('available')).toBe('Evidence available')
    expect(evidenceStatusLabel('failed')).toBe('Evidence generation failed')
    expect(evidenceStatusLabel('custom_state')).toBe('custom_state')
    expect(workerLabel('not_started')).toBe('Worker offline')
    expect(workerLabel('mystery')).toBe('mystery')
  })

  it('clamps values into the inclusive range', () => {
    expect(clamp(-2)).toBe(0)
    expect(clamp(0.4)).toBe(0.4)
    expect(clamp(2)).toBe(1)
    expect(clamp(5, 0, 10)).toBe(5)
  })
})
