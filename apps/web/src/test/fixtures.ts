import type {
  AuditListResponse,
  Capabilities,
  ConfigResponse,
  EventListResponse,
  EventSummary,
  Job,
  MetricsResponse,
  Review,
  Upload,
} from '../lib/types'

export const EVENT_ID = 'evt-11111111-1111-4111-8111-111111111111'
export const JOB_ID = 'job-22222222-2222-4222-8222-222222222222'
export const CREATED_AT = '2026-11-17T10:15:00+00:00'

export function makeReview(overrides: Partial<Review> = {}): Review {
  return {
    review_id: 'rev-33333333-3333-4333-8333-333333333333',
    event_id: EVENT_ID,
    actor: 'alex',
    decision: 'benign',
    reason_codes: ['activity_benign'],
    notes: 'Customer returned the item to the shelf.',
    supersedes_review_id: null,
    created_at: '2026-11-17T10:30:00+00:00',
    ...overrides,
  }
}

export function makeEvent(overrides: Partial<EventSummary> = {}): EventSummary {
  return {
    schema_version: 'shop_event_v1',
    event_id: EVENT_ID,
    source_id: 'upload',
    session_id: 'session-1',
    job_id: JOB_ID,
    track_ids: ['t7'],
    media_interval_ms: { start: 4000, end: 6600 },
    event_type: 'person_activity_observed',
    reasons: ['person_track_observed'],
    review_score: null,
    score_semantics: null,
    calibration_version: null,
    quality: {
      policy: 'baseline_descriptive_v1',
      observed_samples: 13,
      observed_duration_ms: 2600,
      observed_fraction: 1,
      crowded_max_persons: 1,
      warnings: [],
    },
    required_evidence: ['person_detections', 'track_observations', 'source_coverage'],
    available_evidence: ['person_detections', 'track_observations', 'source_coverage'],
    evidence: {
      status: 'unavailable',
      clip_id: null,
      reason: 'Evidence capture was disabled for this analysis.',
      thumbnail_url: null,
      media_url: null,
    },
    model_bundle: 'shop-baseline-cpu-v1',
    policy_version: 'baseline_descriptive_v1',
    review_state: 'unreviewed',
    created_at: CREATED_AT,
    deleted_at: null,
    source: {
      source_id: 'upload',
      kind: 'file',
      label: 'Front entrance camera',
      camera_key: null,
    },
    job_status: 'completed',
    reviews: [],
    ...overrides,
  }
}

export function makeEventList(items: EventSummary[] = [makeEvent()]): EventListResponse {
  return {
    items,
    next_cursor: null,
    has_more: false,
    unreviewed_count: items.filter((item) => item.review_state === 'unreviewed').length,
  }
}

export function makeMetrics(overrides: Partial<MetricsResponse['summary']> = {}): MetricsResponse {
  return {
    summary: {
      uploads: 1,
      events_total: 1,
      events_by_review_state: { unreviewed: 1, benign: 0, review_needed: 0, inconclusive: 0 },
      reviews_today: 0,
      ...overrides,
    },
    jobs: { queue_depth: 0, counts_by_state: { completed: 1 }, active_limit: 1 },
    worker: {
      status: 'idle',
      fresh: true,
      last_seen_at: CREATED_AT,
      detail: 'Worker heartbeat is fresh.',
    },
    storage: {
      free_bytes: 8_000_000_000,
      total_bytes: 100_000_000_000,
      used_bytes: 92_000_000_000,
      free_fraction: 0.08,
      minimum_free_bytes: 2_000_000_000,
      below_minimum: false,
    },
    models: {},
    recent_events: [],
    uptime_seconds: 120,
    label: 'Actual measured result',
  }
}

export function makeUpload(overrides: Partial<Upload> = {}): Upload {
  return {
    upload_id: 'upl-44444444-4444-4444-8444-444444444444',
    filename: 'front-door.mp4',
    bytes: 1_048_576,
    sha256: 'a'.repeat(64),
    content_type: 'video/mp4',
    status: 'ready',
    probe: {
      container: 'mp4',
      codec: 'h264',
      width: 1280,
      height: 720,
      duration_ms: 65_000,
      fps: 25,
      frame_count: 1625,
      first_frame_decoded: true,
    },
    created_at: CREATED_AT,
    deleted_at: null,
    ...overrides,
  }
}

export function makeJob(overrides: Partial<Job> = {}): Job {
  return {
    job_id: JOB_ID,
    kind: 'file',
    status: 'completed',
    source: { kind: 'file', upload_id: 'upl-1', source_id: null, label: 'front-door.mp4' },
    analysis_profile: 'cpu_baseline_v1',
    sampling_fps: 5,
    privacy_mask_version: 'zones-v1',
    retain_evidence: true,
    progress: {},
    metrics: {},
    coverage: {},
    model_bundle: 'shop-baseline-cpu-v1',
    policy_version: 'baseline_descriptive_v1',
    error: null,
    cancel_requested: false,
    created_at: CREATED_AT,
    started_at: CREATED_AT,
    finished_at: CREATED_AT,
    events_count: 1,
    transitions: [],
    ...overrides,
  }
}

export function makeConfig(overrides: Partial<ConfigResponse> = {}): ConfigResponse {
  return {
    schema_version: 1,
    runtime: {
      device: 'cpu',
      inference_threads: 4,
      active_jobs: 1,
      queued_jobs: 8,
      allow_development_models: true,
    },
    storage: {
      root: '/managed/local/storage',
      auto_migrate: true,
      max_upload_bytes: 2 * 1024 ** 3,
      max_upload_duration_ms: 7_200_000,
      max_upload_width: 3840,
      max_upload_height: 2160,
      min_free_bytes: 2_000_000_000,
      evidence_days: 7,
      metadata_days: 30,
      audit_days: 90,
    },
    analysis: {
      long_edge_px: 960,
      sampling_fps: 5,
      person_score_threshold: 0.5,
      tracker_iou_weight: 0.6,
      tracker_motion_weight: 0.4,
      tracker_gate_chi2: 9.21,
      tracker_confirm_hits: 3,
      tracker_max_prediction_ms: 600,
      tracker_tentative_max_prediction_ms: 250,
      tracker_remove_after_ms: 1200,
      max_person_crops: 8,
      live_queue_frames: 3,
      evidence_context_before_ms: 2000,
      evidence_context_after_ms: 2000,
      event_min_dwell_ms: 1000,
      event_gap_split_ms: 1500,
      max_events_per_job: 100,
    },
    quality: { minimum_valid_fraction: 0.5 },
    policy: {
      version: 'baseline_descriptive_v1',
      score_semantics: null,
      high_threshold: 0.8,
      low_threshold: 0.35,
      open_dwell_ms: 1000,
      close_dwell_ms: 1000,
      cooldown_ms: 5000,
    },
    security: {
      bind_host: '127.0.0.1:8000',
      audio_enabled: false,
      reviewers: ['alex', 'sam'],
      approved_cameras: [{ key: 'front', label: 'Front entrance', device_index: 0 }],
    },
    models: {
      person: { enabled: true, manifest: 'person.ssdlite320_mobilenet_v3_large.cpu.json' },
    },
    zones: {
      version: 'zones-v1',
      coordinate_space: 'normalized',
      analysis_roi: [],
      privacy_mask: [],
    },
    has_api_token: true,
    worker: {
      status: 'idle',
      fresh: true,
      last_seen_at: CREATED_AT,
      detail: 'Worker heartbeat is fresh.',
    },
    ...overrides,
  }
}

export function makeCapabilities(overrides: Partial<Capabilities> = {}): Capabilities {
  const model = {
    role: 'person',
    model_id: 'ssdlite320_mobilenet_v3_large',
    version: 'coco-a79551df',
    enabled: true,
    available: true,
    capability: 'person_detection',
    commercial_status: 'development_only',
    architecture: 'SSDLite320 MobileNetV3 Large',
    sha256: 'b'.repeat(64),
    reason: 'Development-only checkpoint; commercial release is not approved.',
    details: {},
  }
  return {
    service: { name: 'Shop API', version: '0.1.0', api_version: 'v1' },
    host: {
      os: 'Linux',
      cpu_model: 'Intel Core i7-1065G7',
      logical_cores: 8,
      physical_cores: 4,
      memory_total_kb: 16_000_000,
      python: '3.12.0',
      torch: '2.10.0+cpu',
      torchvision: '0.25.0+cpu',
      device: 'cpu',
      cuda_build: null,
    },
    cpu_verification: { passed: true, checks: [], label: 'Actual measured result' },
    sources: { webcam: true, file: true, rtsp: false },
    models: {
      bundle: 'shop-baseline-cpu-v1',
      person: model,
      retail: { ...model, role: 'retail', enabled: false, available: false, reason: 'No approved retail checkpoint.' },
      pose: { ...model, role: 'pose', enabled: false, available: false, reason: 'No approved pose checkpoint.' },
      temporal: { ...model, role: 'temporal', enabled: false, available: false, reason: 'No trained temporal model.' },
      policy: { version: 'baseline_descriptive_v1', score_semantics: null },
    },
    limits: {
      max_upload_bytes: 2 * 1024 ** 3,
      sampling_fps_max: 5,
      analysis_profile: 'cpu_baseline_v1',
      long_edge_px: 960,
    },
    security: { auth_configured: true, reviewer_allowlist: true, audio_enabled: false },
    features: { retail_detection: false, pose: false, temporal_model: false, evidence: true },
    ...overrides,
  }
}

export function makeAudit(items: AuditListResponse['items'] = []): AuditListResponse {
  return { items, next_cursor: null, has_more: false }
}
