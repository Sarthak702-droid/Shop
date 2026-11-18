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
  actions?: ActionSegment[]
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

export type ActionLabel =
  | 'reach_to_shelf'
  | 'retract_from_shelf'
  | 'hand_in_shelf'
  | 'inspect_product'
  | 'inspect_shelf'

export const ACTION_LABELS: ActionLabel[] = [
  'reach_to_shelf',
  'retract_from_shelf',
  'hand_in_shelf',
  'inspect_product',
  'inspect_shelf',
]

export interface ActionSegment {
  label: ActionLabel
  display?: string
  start_ms: number
  end_ms: number
  steps?: number
  peak_score?: number
  mean_score: number
  threshold?: number
  score_semantics?: string
}

export type ActionVerdict = 'correct' | 'partially_correct' | 'incorrect' | 'no_interaction' | 'unclear'

export interface ActionFeedback {
  verdict: ActionVerdict
  observed_actions: ActionLabel[]
}

export interface Review {
  review_id: string
  event_id: string
  actor: string
  decision: ReviewDecision
  reason_codes: string[]
  notes: string
  action_feedback?: ActionFeedback | null
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

export interface AnalysisProfileInfo {
  profile: string
  label: string
  available: boolean
  sources: string[]
  event_type: string
  research_mode: boolean
  classes?: ActionLabel[] | null
  model?: string
  reason?: string | null
  detail: string
}

export interface ActionWarning {
  code: string
  severity: 'info' | 'warning'
  message: string
}

export interface ActionClassSummary {
  segments: number
  active_seconds: number
  active_fraction: number
  mean_score: number
  max_score: number
  per_minute: number
}

export interface ActionAnalysis {
  profile: string
  model: {
    model_id: string
    run: string
    sha256: string
    feature_schema: string
    detector: string
    policy: string
    receptive_field_steps: number
  }
  input: { fps: number; frames: number; width: number; height: number; step_frames: number }
  warnings: ActionWarning[]
  summary: {
    steps: number
    analysed_seconds: number
    episodes: number
    person_visible_fraction: number
    per_class: Record<ActionLabel, ActionClassSummary>
    episodes_with_low_visibility: number
  }
  timeline: { path: string; sha256: string }
  timing_ms: Record<string, number>
  steps_per_second: number
  label: string
}

export interface ActionTimeline {
  schema: string
  model: { model_id: string; run: string; sha256: string; feature_schema: string }
  classes: ActionLabel[]
  display: Record<string, string>
  thresholds: number[]
  step_ms: number
  pts_ms: number[]
  scores: number[][]
  person_present: number[]
  episodes: Array<{ start_ms: number; end_ms: number; primary: ActionLabel; actions: ActionSegment[] }>
  policy: Record<string, number | string>
  score_semantics: string
}

export interface ActionTimelineResponse {
  job_id: string
  timeline: ActionTimeline
}

export interface WilsonEstimate {
  estimate: number | null
  low: number | null
  high: number | null
  n: number
}

export interface ClassMetrics {
  ap: number
  chance?: number
  precision: number
  recall: number
  f1: number
  threshold?: number
  support_steps?: number
  predicted?: number
  ground_truth?: number
}

export type PerClass<T> = Record<ActionLabel, T> & { _map?: number; _chance_map?: number }

export interface ResearchInsights {
  merl: {
    available: boolean
    reason?: string
    run?: string
    checkpoint_sha256?: string
    split?: Record<string, number>
    steps?: Record<string, number>
    best_epoch?: number
    training_curve?: Array<{ epoch: number; train_loss: number; val_map: number; val_ap: Record<ActionLabel, number> }>
    class_stats?: { train: Record<ActionLabel, { segments: number; median_s: number | null; p10_s: number | null; steps: number }>; test: Record<ActionLabel, { segments: number; median_s: number | null; p10_s: number | null; steps: number }> }
    validation?: { frame: PerClass<ClassMetrics>; segment_eval_policy: PerClass<ClassMetrics> }
    test?: { frame: PerClass<ClassMetrics>; segment_eval_policy: PerClass<ClassMetrics>; segment_runtime_policy: PerClass<ClassMetrics> }
    runtime_policy_selection?: { grid: Array<{ min_segment_steps: number; merge_gap_steps: number; val_segment_map: number }>; chosen: { min_segment_steps: number; merge_gap_steps: number; val_segment_map: number }; selected_on: string }
    confusion_test?: { labels: string[]; counts: number[][]; row_normalised: number[][]; excluded_multi_label_steps: number }
    visibility_test?: { person_present_fraction: number; person_visible: Record<string, number | null>; person_not_detected: Record<string, number | null> }
    failures_test?: {
      false_positives: Array<{ video: string; class: ActionLabel; start_s: number; end_s: number; mean_score: number; overlapping_gt: string[]; person_visible: number }>
      missed: Array<{ video: string; class: ActionLabel; start_s: number; end_s: number; max_score: number; threshold: number; person_visible: number }>
      false_positive_count: number
      missed_count: number
      missed_mean_person_visible: number | null
      false_positive_with_other_gt_fraction: number | null
    }
    reliability_test?: Record<ActionLabel, { ece: number; bins: Array<{ bin: string; mean_score: number; positive_rate: number; steps: number }> }>
    threshold_sweep_validation?: Record<ActionLabel, Array<{ threshold: number; precision: number; recall: number; f1: number }>>
    per_video_test?: Array<{ video: string; map: number; steps: number; person_visible: number }>
    baseline_no_context?: { run: string; frame_map: number; segment_map_iou_0_5: number } | null
    model?: Record<string, unknown>
    notes?: string[]
  }
  retailaction: {
    available?: boolean
    updated_utc?: string
    classes?: string[]
    steps?: Array<{ key: string; label: string; state: string }>
    archives?: Array<{ name: string; bytes: number; verified: boolean }>
    data?: {
      train_source_clips: number | null
      source_positive_clips: Record<string, number> | null
      usable_positive_clips: Record<string, number> | null
      quarantined_metadata: number
      no_valid_pose_excluded: number | null
    }
    class_imbalance_note?: string
    next_action?: string
    runtime_enabled?: boolean
  }
  pose_compatibility: {
    joint_mapping: { direct: string[]; derivable_as_midpoint: Record<string, string[]>; missing_in_coco17: string[]; action_critical_missing: string[]; coverage: number }
    supplied_pose_semantics: { samples_read: number; joint_observations: number; fraction_on_1_64_grid: number | null; joint_score: Record<string, number>; joint_score_above_1_fraction: number | null; per_joint_present_fraction: Record<string, number | null> } | null
    cpu_pose_timing: { model: string; input: string; threads: number; mean_ms: number; p95_ms: number; max_fps: number; realtime_at_5hz: boolean; peak_rss_kb: number } | null
    verdict: { compatible: boolean; runtime_enabled: boolean; blockers: string[]; required_before_enabling: string[] }
  } | null
  own_evaluation: { available: boolean; reason?: string; [key: string]: unknown }
  cpu_benchmark: {
    host: { cpu: string; logical_cores: number; threads: number; python: string; torch: string }
    model_load_ms: number
    videos: number
    media_seconds: number
    real_time_factor: { mean: number; max: number }
    per_step_ms: Record<string, number>
    peak_rss_kb: number
    budget: { step_ms_at_5hz: number }
    note: string
  } | null
  runtime_e2e: {
    videos: Array<{ video: string; media_seconds: number; feature_parity_with_training: boolean; frame_map: number; episodes: number; gt_intervals: number; person_visible_fraction: number; real_time_factor: number }>
    all_parity: boolean
    pooled_frame_map: number
    note: string
  } | null
  app: {
    jobs: Array<{
      job_id: string
      status: string
      filename: string | null
      created_at: string
      analysed_seconds: number | null
      episodes: number | null
      person_visible_fraction: number | null
      per_class: Record<ActionLabel, ActionClassSummary> | null
      warnings: string[]
      real_time_factor: number | null
      wall_time_ms: number | null
      peak_rss_kb: number | null
      steps_per_second: number | null
      error: { code: string; message: string } | null
    }>
    job_states: Record<string, number>
    analysed_minutes: number
    job_warnings: Record<string, number>
    events: {
      total: number
      reviewed: number
      with_action_feedback: number
      decisions: Record<string, number>
      verdicts: Record<string, number>
      warnings: Record<string, number>
      person_visibility_histogram: Array<{ bin: string; events: number }>
      median_duration_ms: number | null
    }
    reviewer_precision: {
      per_class: Record<ActionLabel, { predicted_events: number; confirmed: number; rejected: number; reviewer_added: number; precision: WilsonEstimate }>
      overall: WilsonEstimate
    }
  }
  action_model: { available: boolean; model_id: string; version: string; reason: string; commercial_status: string }
}
