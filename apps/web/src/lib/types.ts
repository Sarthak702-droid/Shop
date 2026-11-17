export type ReviewState = 'unreviewed' | 'benign' | 'review_needed' | 'inconclusive'
export type ReviewDecision = Exclude<ReviewState, 'unreviewed'>

export interface SourceSummary {
  source_id: string
  kind: string
  label: string
  camera_key: string | null
  state?: string
  approved_camera?: boolean
  active_job?: { job_id: string; status: string; cancel_requested: boolean } | null
  camera?: { label: string; device_index: number } | null
  health?: { last_known_state: string; live_checked: boolean; preview_url: string | null }
  created_at?: string
  updated_at?: string
}

export interface EventQuality {
  policy?: string
  mean_person_score?: number
  max_person_score?: number
  observed_samples?: number
  observed_duration_ms?: number
  observed_fraction?: number
  crowded_max_persons?: number
  source_discontinuity?: boolean
  no_activity_score_available?: boolean
  warnings?: string[]
  [key: string]: unknown
}

export interface EventEvidence {
  status: 'pending' | 'available' | 'unavailable' | 'failed' | 'deleted' | 'expired'
  clip_id: string | null
  reason: string | null
  thumbnail_url: string | null
  media_url: string | null
  asset?: {
    clip_id: string
    bytes: number
    sha256: string
    mime_type: string
    width: number
    height: number
    fps: number
    duration_ms: number
    frame_count: number
    encoder: string
    media_interval_ms?: { start: number; end: number }
    retain_until: string | null
  }
}

export interface EventSummary {
  schema_version: string
  event_id: string
  source_id: string
  session_id: string
  job_id: string
  track_ids: string[]
  media_interval_ms: { start: number; end: number }
  event_type: string
  reasons: string[]
  review_score: number | null
  score_semantics: string | null
  calibration_version: string | null
  quality: EventQuality
  required_evidence: string[]
  available_evidence: string[]
  evidence: EventEvidence
  model_bundle: string
  policy_version: string
  review_state: ReviewState
  created_at: string
  deleted_at: string | null
  source?: SourceSummary | null
  job_status?: string | null
  reviews?: Review[]
}

export interface Review {
  review_id: string
  event_id: string
  actor: string
  decision: ReviewDecision
  reason_codes: string[]
  notes: string
  supersedes_review_id: string | null
  created_at: string
}

export interface EventListResponse {
  items: EventSummary[]
  next_cursor: string | null
  has_more: boolean
  unreviewed_count: number
}

export interface Job {
  job_id: string
  kind: 'file' | 'webcam'
  status: 'queued' | 'running' | 'cancelling' | 'completed' | 'failed' | 'cancelled'
  source: { kind: string; upload_id: string | null; source_id: string | null; label: string | null }
  analysis_profile: string
  sampling_fps: number
  privacy_mask_version: string
  retain_evidence: boolean
  progress: Record<string, unknown>
  metrics: Record<string, unknown>
  coverage: Record<string, unknown>
  model_bundle: string
  policy_version: string
  error: { code: string; message: string } | null
  cancel_requested: boolean
  created_at: string
  started_at: string | null
  finished_at: string | null
  events_count?: number
  transitions?: Array<{ from_state: string | null; to_state: string; reason: string | null; created_at: string }>
  request_id?: string
}

export interface Upload {
  upload_id: string
  filename: string
  bytes: number
  sha256: string
  content_type: string | null
  status: string
  probe: Record<string, unknown>
  created_at: string
  deleted_at: string | null
}

export interface ReadinessCheck {
  name: string
  status: 'pass' | 'fail'
  detail: string
  observed: Record<string, unknown> | null
}

export interface WorkerStatus {
  status: string
  fresh: boolean
  last_seen_at: string | null
  age_seconds?: number
  job_id?: string | null
  reason?: string | null
  message?: string | null
  detail: string
}

export interface ReadinessResponse {
  status: 'ready' | 'degraded'
  ready: boolean
  checks: ReadinessCheck[]
  worker: WorkerStatus
  checked_at: string
}

export interface ModelStatus {
  role: string
  model_id: string
  version: string
  enabled: boolean
  available: boolean
  capability: string
  commercial_status: string
  architecture: string | null
  sha256: string | null
  reason: string
  details: Record<string, unknown>
}

export interface Capabilities {
  service: { name: string; version: string; api_version: string }
  host: {
    os: string
    cpu_model: string
    logical_cores: number | null
    physical_cores: number | null
    memory_total_kb: number | null
    python: string
    torch: string
    torchvision: string
    device: string
    cuda_build: string | null
  }
  cpu_verification: {
    passed: boolean
    checks: Array<{ name: string; passed: boolean; observed: unknown; expectation: string }>
    label: string
  }
  sources: Record<string, unknown>
  models: {
    bundle: string
    person: ModelStatus
    retail: ModelStatus
    pose: ModelStatus
    temporal: ModelStatus
    policy: Record<string, unknown>
  }
  limits: Record<string, unknown>
  security: Record<string, unknown>
  features: Record<string, boolean>
}

export interface MetricsResponse {
  summary: {
    uploads: number
    events_total: number
    events_by_review_state: Record<string, number>
    reviews_today: number
  }
  jobs: { queue_depth: number; counts_by_state: Record<string, number>; active_limit: number }
  worker: WorkerStatus
  storage: {
    free_bytes: number
    total_bytes: number
    used_bytes: number
    free_fraction: number
    minimum_free_bytes: number
    below_minimum: boolean
  }
  models: Record<string, ModelStatus>
  recent_events: EventSummary[]
  uptime_seconds: number
  label: string
}

export interface SourceListResponse {
  items: SourceSummary[]
  approved_cameras: Array<{
    key: string
    label: string
    device_index: number
    registered: boolean
  }>
}

export interface UploadResponse {
  upload: Upload
  request_id: string
}

export interface AuditEntry {
  audit_id: string
  actor: string
  action: string
  object_type: string
  object_id: string | null
  request_id: string | null
  metadata: Record<string, unknown>
  created_at: string
}

export interface AuditListResponse {
  items: AuditEntry[]
  next_cursor: string | null
  has_more: boolean
}

export interface ModelsResponse {
  bundle: string
  person: ModelStatus
  retail: ModelStatus
  pose: ModelStatus
  temporal: ModelStatus
  policy: {
    version: string
    score_semantics: string | null
    baseline_event_type: string
    no_activity_score_available: boolean
    detail: string
  }
  labels: Record<string, string>
}

export interface ConfigResponse {
  schema_version: number
  runtime: {
    device: string
    inference_threads: number
    active_jobs: number
    queued_jobs: number
    allow_development_models: boolean
  }
  storage: {
    root: string
    auto_migrate: boolean
    max_upload_bytes: number
    max_upload_duration_ms: number
    max_upload_width: number
    max_upload_height: number
    min_free_bytes: number
    evidence_days: number
    metadata_days: number
    audit_days: number
  }
  analysis: {
    long_edge_px: number
    sampling_fps: number
    person_score_threshold: number
    tracker_iou_weight: number
    tracker_motion_weight: number
    tracker_gate_chi2: number
    tracker_confirm_hits: number
    tracker_max_prediction_ms: number
    tracker_tentative_max_prediction_ms: number
    tracker_remove_after_ms: number
    max_person_crops: number
    live_queue_frames: number
    evidence_context_before_ms: number
    evidence_context_after_ms: number
    event_min_dwell_ms: number
    event_gap_split_ms: number
    max_events_per_job: number
  }
  quality: { minimum_valid_fraction: number }
  policy: {
    version: string
    score_semantics: string | null
    high_threshold: number
    low_threshold: number
    open_dwell_ms: number
    close_dwell_ms: number
    cooldown_ms: number
  }
  security: {
    bind_host: string
    audio_enabled: false
    reviewers: string[]
    approved_cameras: Array<{ key: string; label: string; device_index: number }>
  }
  models: Record<string, { enabled: boolean; manifest: string }>
  zones: {
    version: string
    coordinate_space: string
    analysis_roi: number[][]
    privacy_mask: number[][]
  }
  has_api_token: boolean
  worker: WorkerStatus
}

export interface JobResponse {
  job: Job
  request_id: string
}

export interface JobListResponse {
  items: Job[]
  next_cursor: string | null
}

export interface SourceHealthResponse {
  source_id: string
  camera_key: string | null
  state: string
  probe: Record<string, unknown>
  checked_at: string
}

export interface RetentionResponse {
  clips_expired: number
  events_updated: number
  file_cleanup_failures: number
  applied_at: string
  request_id: string
}
