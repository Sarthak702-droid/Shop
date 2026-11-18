import type { ReviewDecision, ReviewState } from './types'

const dateTimeFormatter = new Intl.DateTimeFormat(undefined, {
  dateStyle: 'medium',
  timeStyle: 'short',
})
const timeFormatter = new Intl.DateTimeFormat(undefined, {
  hour: 'numeric',
  minute: '2-digit',
  second: '2-digit',
})
const dateFormatter = new Intl.DateTimeFormat(undefined, { month: 'short', day: 'numeric' })

export function formatDateTime(value: string | null | undefined): string {
  if (!value) return 'Not available'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return 'Not available'
  return dateTimeFormatter.format(date)
}

export function formatTime(value: string | null | undefined): string {
  if (!value) return '—'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return '—'
  return timeFormatter.format(date)
}

export function formatShortDate(value: string | null | undefined): string {
  if (!value) return '—'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return '—'
  return dateFormatter.format(date)
}

export function formatRelative(value: string | null | undefined): string {
  if (!value) return 'unknown'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return 'unknown'
  const seconds = Math.round((date.getTime() - Date.now()) / 1000)
  const formatter = new Intl.RelativeTimeFormat(undefined, { numeric: 'auto' })
  if (Math.abs(seconds) < 60) return formatter.format(seconds, 'second')
  const minutes = Math.round(seconds / 60)
  if (Math.abs(minutes) < 60) return formatter.format(minutes, 'minute')
  const hours = Math.round(minutes / 60)
  if (Math.abs(hours) < 24) return formatter.format(hours, 'hour')
  return formatter.format(Math.round(hours / 24), 'day')
}

export function formatBytes(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—'
  if (value < 1024) return `${value} B`
  const units = ['KB', 'MB', 'GB', 'TB']
  let amount = value / 1024
  let unit = 0
  while (amount >= 1024 && unit < units.length - 1) {
    amount /= 1024
    unit += 1
  }
  return `${amount >= 10 ? amount.toFixed(0) : amount.toFixed(1)} ${units[unit]}`
}

export function formatDuration(ms: number | null | undefined): string {
  if (ms === null || ms === undefined || !Number.isFinite(ms) || ms < 0) return '—'
  const totalSeconds = ms / 1000
  if (totalSeconds < 60) return `${totalSeconds < 10 ? totalSeconds.toFixed(1) : Math.round(totalSeconds)}s`
  const hours = Math.floor(totalSeconds / 3600)
  const minutes = Math.floor((totalSeconds % 3600) / 60)
  const seconds = Math.floor(totalSeconds % 60)
  if (hours > 0) return `${hours}h ${minutes}m`
  return `${minutes}m ${seconds.toString().padStart(2, '0')}s`
}

export function formatMediaTime(ms: number | null | undefined): string {
  if (ms === null || ms === undefined || !Number.isFinite(ms) || ms < 0) return '—'
  const total = Math.floor(ms / 1000)
  const minutes = Math.floor(total / 60)
  const seconds = total % 60
  return `${minutes}:${seconds.toString().padStart(2, '0')}`
}

export function formatPercent(value: number | null | undefined, digits = 0): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—'
  return `${(value * 100).toFixed(digits)}%`
}

export function formatNumber(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—'
  return new Intl.NumberFormat().format(value)
}

export function reasonLabel(reason: string): string {
  const labels: Record<string, string> = {
    person_track_observed: 'Person track observed',
    source_coverage: 'Source coverage',
    person_detections: 'Person detections',
    track_observations: 'Track observations',
    item_visibility_lost: 'Item visibility lost',
    hand_near_bag: 'Hand near bag',
    reach_to_shelf: 'Reach to shelf',
    retract_from_shelf: 'Retract from shelf',
    hand_in_shelf: 'Hand in shelf',
    inspect_product: 'Inspect product',
    inspect_shelf: 'Inspect shelf',
    insufficient_person_visibility: 'Person not visible for most of the episode',
    limited_temporal_context: 'Less than 12 s of preceding context',
    insufficient_visibility: 'Low person visibility across the video',
    frame_rate_differs: 'Frame rate differs from training (30 fps)',
    aspect_ratio_differs: 'Aspect ratio differs from training',
    short_video: 'Video shorter than model context',
    research_model_domain: 'Research model outside its training view',
    action_timeline: 'Action score timeline',
    person_box: 'Person box',
    clip_with_context: 'Clip with context',
    product_pickup_observed: 'Product pickup observed',
    return_to_shelf: 'Return to shelf',
    intermittent_track_observations: 'Intermittent track observations',
    crowded_scene_identity_ambiguity: 'Crowded scene identity ambiguity',
    source_discontinuity: 'Source discontinuity',
    context_insufficient: 'Context insufficient',
    activity_benign: 'Activity appears benign in context',
    occlusion: 'Occlusion',
    product_unclear: 'Product unclear',
    customer_owned_item: 'Customer-owned item',
    equipment_fault: 'Equipment fault',
    other: 'Other',
  }
  if (reason.startsWith('action:')) return labels[reason.slice(7)] ?? reason.slice(7).replaceAll('_', ' ')
  return labels[reason] ?? reason.replaceAll('_', ' ')
}

export function eventTypeLabel(value: string): string {
  if (value === 'person_activity_observed') return 'Person activity observed'
  if (value === 'shelf_interaction_observed') return 'Shelf interaction observed'
  return value.replaceAll('_', ' ')
}

export function reviewStateLabel(value: ReviewState): string {
  return {
    unreviewed: 'Unreviewed',
    benign: 'Reviewed · benign',
    review_needed: 'Reviewed · needs review',
    inconclusive: 'Reviewed · inconclusive',
  }[value]
}

export function reviewDecisionLabel(value: ReviewDecision): string {
  return {
    benign: 'Benign',
    review_needed: 'Needs review',
    inconclusive: 'Inconclusive',
  }[value]
}

export function evidenceStatusLabel(value: string): string {
  return {
    available: 'Evidence available',
    pending: 'Evidence pending',
    unavailable: 'Evidence unavailable',
    failed: 'Evidence generation failed',
    deleted: 'Evidence deleted',
    expired: 'Evidence expired',
  }[value] ?? value
}

export function workerLabel(status: string): string {
  return {
    idle: 'Worker ready',
    running: 'Analysis running',
    blocked: 'Worker blocked',
    stopped: 'Worker stopped',
    stale: 'Worker heartbeat stale',
    not_started: 'Worker offline',
    invalid: 'Worker status unreadable',
    unknown: 'Worker status unknown',
  }[status] ?? status
}

export function clamp(value: number, min = 0, max = 1): number {
  return Math.min(max, Math.max(min, value))
}

export function formatTimestampQuery(value: string): string {
  return encodeURIComponent(value)
}

export function actionVerdictLabel(value: string): string {
  return (
    {
      correct: 'Predicted actions correct',
      partially_correct: 'Partly correct',
      incorrect: 'Wrong actions',
      no_interaction: 'No shelf interaction',
      unclear: 'Cannot tell from clip',
    } as Record<string, string>
  )[value] ?? value.replaceAll('_', ' ')
}
