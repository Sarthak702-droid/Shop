import { useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  ApiError,
  apiPost,
  errorMessage,
  newIdempotencyKey,
  uploadVideo,
} from '../lib/api'
import {
  formatBytes,
  formatDateTime,
  formatDuration,
  formatPercent,
  reasonLabel,
} from '../lib/format'
import { useApiResource } from '../lib/hooks'
import type {
  AnalysisProfileInfo,
  Capabilities,
  ConfigResponse,
  JobResponse,
  Upload,
} from '../lib/types'
import {
  DataRow,
  ErrorPanel,
  Field,
  InlineNotice,
  LoadingPanel,
  PageHeader,
  SectionHeading,
  StatusPill,
} from '../components/Common'
import {
  IconCheck,
  IconFile,
  IconFilm,
  IconShield,
  IconUpload,
} from '../components/Icons'

const ACCEPTED = '.mp4,.mov,.webm,.mkv,.avi,.m4v'

function probeValue(probe: Record<string, unknown>, key: string): string {
  const value = probe[key]
  if (value === null || value === undefined || value === '') return '—'
  if (typeof value === 'boolean') return value ? 'Yes' : 'No'
  if (typeof value === 'number') return Number.isFinite(value) ? String(value) : '—'
  return String(value)
}

function UploadProbe({ upload }: { upload: Upload }) {
  const probe = upload.probe
  const width = Number(probe.width ?? 0)
  const height = Number(probe.height ?? 0)
  const duration = Number(probe.duration_ms ?? 0)
  return (
    <div className="probe-card">
      <div className="probe-card__head">
        <div className="probe-card__icon"><IconCheck /></div>
        <div>
          <h3>File validated locally</h3>
          <p>{upload.filename}</p>
        </div>
        <StatusPill tone="success" dot>Ready for analysis</StatusPill>
      </div>
      <dl className="data-grid data-grid--compact">
        <DataRow label="Container label">{probeValue(probe, 'container')} (decoder observation is authoritative)</DataRow>
        <DataRow label="Codec">{probeValue(probe, 'codec')}</DataRow>
        <DataRow label="Dimensions">
          {width > 0 && height > 0 ? `${width} × ${height}` : 'Not reported'}
        </DataRow>
        <DataRow label="Duration">{duration > 0 ? formatDuration(duration) : 'Not reported'}</DataRow>
        <DataRow label="Frame rate">{probeValue(probe, 'fps')}</DataRow>
        <DataRow label="Sampled frame count">{probeValue(probe, 'frame_count')}</DataRow>
        <DataRow label="First frame decoded">{probeValue(probe, 'first_frame_decoded')}</DataRow>
        <DataRow label="Size">{formatBytes(upload.bytes)}</DataRow>
        <DataRow label="SHA-256"><code className="hash-value">{upload.sha256}</code></DataRow>
        <DataRow label="Validated at">{formatDateTime(upload.created_at)}</DataRow>
      </dl>
      <p className="provenance-line provenance-line--tight">
        <span className="provenance-tag">Actual measured result</span>
        <span>Metadata comes from a local decode probe of this uploaded file, not from its filename.</span>
      </p>
    </div>
  )
}

export function UploadPage() {
  const navigate = useNavigate()
  const capabilities = useApiResource<Capabilities>('/capabilities')
  const config = useApiResource<ConfigResponse>('/config')
  const uploads = useApiResource<{ items: Upload[]; next_cursor: string | null }>('/uploads')
  const [file, setFile] = useState<File | null>(null)
  const [selectedUpload, setSelectedUpload] = useState<Upload | null>(null)
  const [phase, setPhase] = useState<'idle' | 'uploading' | 'ready' | 'error'>('idle')
  const [progress, setProgress] = useState(0)
  const [uploadError, setUploadError] = useState<unknown>(null)
  const [samplingFps, setSamplingFps] = useState('5')
  const [retainEvidence, setRetainEvidence] = useState(true)
  const [jobError, setJobError] = useState<unknown>(null)
  const [starting, setStarting] = useState(false)
  const [jobKey, setJobKey] = useState('')
  const [dragging, setDragging] = useState(false)
  const abortRef = useRef<AbortController | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const maxBytes = Number(capabilities.data?.limits.max_upload_bytes ?? 0)
  const maxFps = Number(capabilities.data?.limits.sampling_fps_max ?? 5)
  const defaultProfile = String(capabilities.data?.limits.analysis_profile ?? 'cpu_baseline_v1')
  const profiles = (capabilities.data?.limits.analysis_profiles as AnalysisProfileInfo[] | undefined) ?? []
  const [profileChoice, setProfileChoice] = useState<string | null>(null)
  const analysisProfile = profileChoice ?? defaultProfile
  const selectedProfile = profiles.find((item) => item.profile === analysisProfile)
  const actionProfile = Boolean(selectedProfile?.research_mode)
  const zoneVersion = config.data?.zones.version ?? ''
  const fpsOptions = useMemo(
    () => [1, 2, 3, 5, 10, 15].filter((value) => value <= maxFps),
    [maxFps],
  )

  const chooseFile = (candidate: File | null) => {
    setUploadError(null)
    setJobError(null)
    if (!candidate) return
    const extension = candidate.name.slice(candidate.name.lastIndexOf('.')).toLowerCase()
    if (!ACCEPTED.split(',').includes(extension)) {
      setFile(null)
      setPhase('error')
      setUploadError(new Error('Choose a supported video container: MP4, MOV, WebM, MKV, AVI or M4V.'))
      return
    }
    if (maxBytes > 0 && candidate.size > maxBytes) {
      setFile(null)
      setPhase('error')
      setUploadError(new Error(`This file is ${formatBytes(candidate.size)}; the configured local limit is ${formatBytes(maxBytes)}.`))
      return
    }
    setFile(candidate)
    setSelectedUpload(null)
    setPhase('idle')
    setProgress(0)
    setJobKey('')
  }

  const startUpload = async () => {
    if (!file || phase === 'uploading') return
    setPhase('uploading')
    setProgress(0)
    setUploadError(null)
    const controller = new AbortController()
    abortRef.current = controller
    try {
      const result = await uploadVideo(file, setProgress, controller.signal)
      setSelectedUpload(result.upload)
      setPhase('ready')
      setJobKey(newIdempotencyKey())
      uploads.reload()
      window.dispatchEvent(new CustomEvent('shop-queue-changed'))
    } catch (caught) {
      if (caught instanceof DOMException && caught.name === 'AbortError') {
        setPhase('idle')
        setProgress(0)
      } else {
        setUploadError(caught)
        setPhase('error')
      }
    } finally {
      abortRef.current = null
    }
  }

  const cancelUpload = () => {
    abortRef.current?.abort()
    abortRef.current = null
    setPhase('idle')
    setProgress(0)
  }

  const useExistingUpload = (upload: Upload) => {
    setSelectedUpload(upload)
    setFile(null)
    setUploadError(null)
    setJobError(null)
    setPhase('ready')
    setJobKey(newIdempotencyKey())
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  const startAnalysis = async () => {
    if (!selectedUpload) return
    setStarting(true)
    setJobError(null)
    try {
      const result = await apiPost<JobResponse>(
        '/jobs',
        {
          source: { kind: 'file', upload_id: selectedUpload.upload_id },
          analysis_profile: analysisProfile,
          // The shelf-action model is defined at 5 Hz steps; its cadence is not user-selectable.
          sampling_fps: actionProfile ? 5 : Number(samplingFps),
          privacy_mask_version: zoneVersion,
          retain_evidence: retainEvidence,
        },
        { 'Idempotency-Key': jobKey || newIdempotencyKey() },
      )
      navigate(`/jobs/${result.job.job_id}`)
    } catch (caught) {
      setJobError(caught)
    } finally {
      setStarting(false)
    }
  }

  const activeUpload = selectedUpload
  const recentUploads = (uploads.data?.items ?? []).filter((item) => item.status === 'ready')

  return (
    <div className="page-stack">
      <PageHeader
        eyebrow="Local source"
        title="Add footage"
        subtitle="Validate a recording locally, then queue CPU analysis. Files never leave this host."
        actions={
          <button className="button button--secondary" type="button" onClick={() => navigate('/review')}>
            Review queue
          </button>
        }
      />

      <div className="step-strip" aria-label="Upload steps">
        <div className={`step${file || activeUpload ? ' step--done' : ' step--active'}`}>
          <span className="step__index">{file || activeUpload ? <IconCheck /> : '1'}</span>
          <div><strong>Choose a recording</strong><small>Supported local video container</small></div>
        </div>
        <div className={`step${activeUpload ? ' step--done' : phase === 'uploading' ? ' step--active' : ''}`}>
          <span className="step__index">{activeUpload ? <IconCheck /> : '2'}</span>
          <div><strong>Validate</strong><small>Decode probe and SHA-256</small></div>
        </div>
        <div className={`step${activeUpload ? ' step--active' : ''}`}>
          <span className="step__index">3</span>
          <div><strong>Queue analysis</strong><small>Separate local worker process</small></div>
        </div>
      </div>

      {capabilities.loading && !capabilities.data && <LoadingPanel label="Loading local upload limits…" rows={2} />}
      {capabilities.error && !capabilities.data && (
        <ErrorPanel error={capabilities.error} onRetry={capabilities.reload} title="Upload limits could not load" />
      )}

      <div className="upload-grid">
        <section className="panel">
          <SectionHeading
            title="1 · Choose and upload"
            detail="The server sniffs the actual container and decodes a first frame before accepting the file."
            action={<span className="provenance-tag">Actual measured result</span>}
          />
          <div
            className={`dropzone${dragging ? ' dropzone--dragging' : ''}${file ? ' dropzone--filled' : ''}`}
            onDragOver={(event) => {
              event.preventDefault()
              setDragging(true)
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(event) => {
              event.preventDefault()
              setDragging(false)
              chooseFile(event.dataTransfer.files[0] ?? null)
            }}
            onClick={() => inputRef.current?.click()}
            role="button"
            tabIndex={0}
            onKeyDown={(event) => {
              if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault()
                inputRef.current?.click()
              }
            }}
            aria-label="Choose a local video file"
          >
            <input
              ref={inputRef}
              type="file"
              className="sr-only"
              accept={ACCEPTED}
              onChange={(event) => chooseFile(event.target.files?.[0] ?? null)}
            />
            <span className="dropzone__icon"><IconUpload /></span>
            {file ? (
              <>
                <strong>{file.name}</strong>
                <p>{formatBytes(file.size)} selected · choose another file to replace it</p>
              </>
            ) : (
              <>
                <strong>Drop a recording here, or browse</strong>
                <p>MP4, MOV, WebM, MKV, AVI or M4V. Local validation rejects unreadable or oversized files.</p>
              </>
            )}
          </div>

          <div className="dropzone-meta">
            <span>Byte limit: {maxBytes > 0 ? formatBytes(maxBytes) : 'reported by API'}</span>
            <span>Long-edge analysis resize: {capabilities.data?.limits.long_edge_px ? `${capabilities.data.limits.long_edge_px} px` : '—'}</span>
            <span>Audio is ignored and never analysed.</span>
          </div>

          <div className="upload-actions">
            <button
              className="button button--primary"
              type="button"
              disabled={!file || phase === 'uploading'}
              onClick={() => void startUpload()}
            >
              <IconUpload /> {phase === 'uploading' ? 'Uploading locally…' : 'Upload and validate'}
            </button>
            {phase === 'uploading' && (
              <button className="button button--ghost" type="button" onClick={cancelUpload}>
                Cancel upload
              </button>
            )}
          </div>

          {phase === 'uploading' && (
            <div className="progress-block" aria-live="polite">
              <div className="progress-label">
                <span>Transferring to local managed storage</span>
                <strong>{formatPercent(progress, 0)}</strong>
              </div>
              <div className="progress-track" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(progress * 100)}>
                <span className="progress-fill" style={{ width: `${Math.max(2, progress * 100)}%` }} />
              </div>
              <p className="small text-muted">Upload progress only; validation starts after the bytes are stored.</p>
            </div>
          )}

          {Boolean(uploadError) && (
            <InlineNotice tone="danger" title="Upload was not accepted">
              {errorMessage(uploadError)}
            </InlineNotice>
          )}

          {activeUpload && phase === 'ready' && <UploadProbe upload={activeUpload} />}

          {recentUploads.length > 0 && (
            <details className="recent-uploads">
              <summary>Recently validated uploads ({recentUploads.length})</summary>
              <div className="recent-upload-list">
                {recentUploads.map((item) => (
                  <div className="recent-upload" key={item.upload_id}>
                    <span className="recent-upload__icon"><IconFile /></span>
                    <div>
                      <strong>{item.filename}</strong>
                      <p>{formatBytes(item.bytes)} · {formatDuration(Number(item.probe.duration_ms ?? 0))} · {formatDateTime(item.created_at)}</p>
                    </div>
                    <button className="button button--secondary button--small" type="button" onClick={() => useExistingUpload(item)}>
                      Use
                    </button>
                  </div>
                ))}
              </div>
            </details>
          )}
        </section>

        <section className="panel">
          <SectionHeading
            title="2 · Analysis options"
            detail="Values are proposed starting settings. Actual timing is reported by the worker after completion."
          />
          {!activeUpload ? (
            <div className="locked-panel">
              <IconFilm />
              <h3>Waiting for a validated upload</h3>
              <p>This panel unlocks after a local file passes the decode probe. No analysis runs in this browser or in an HTTP request.</p>
            </div>
          ) : (
            <div className="analysis-form">
              <fieldset className="profile-picker">
                <legend>Analysis profile</legend>
                {(profiles.length ? profiles : [{ profile: defaultProfile, label: 'Person activity (baseline)', available: true, sources: ['file'], event_type: 'person_activity_observed', research_mode: false, detail: '' }]).map((item) => (
                  <label
                    key={item.profile}
                    className={`profile-option${analysisProfile === item.profile ? ' profile-option--selected' : ''}${item.available ? '' : ' profile-option--disabled'}`}
                  >
                    <input
                      type="radio"
                      name="analysis-profile"
                      value={item.profile}
                      checked={analysisProfile === item.profile}
                      disabled={!item.available}
                      onChange={() => setProfileChoice(item.profile)}
                    />
                    <span className="profile-option__body">
                      <span className="profile-option__title">
                        {item.label}
                        {item.research_mode && <StatusPill tone="review">Research mode</StatusPill>}
                        {!item.available && <StatusPill tone="muted">Unavailable</StatusPill>}
                      </span>
                      <small>{item.detail}</small>
                      {item.research_mode && item.classes && (
                        <span className="chip-row">
                          {item.classes.map((label) => <span className="chip" key={label}>{reasonLabel(label)}</span>)}
                        </span>
                      )}
                      {!item.available && item.reason && <small className="text-danger">{item.reason}</small>}
                      <code className="profile-option__id">{item.profile}</code>
                    </span>
                  </label>
                ))}
              </fieldset>
              {!actionProfile && (
                <Field
                  label="Sampling cadence"
                  htmlFor="sampling-fps"
                  hint={`Maximum configured cadence is ${maxFps} fps. Lower cadence uses less CPU and can miss brief activity.`}
                >
                  <select id="sampling-fps" value={samplingFps} onChange={(event) => setSamplingFps(event.target.value)}>
                    {fpsOptions.map((value) => (
                      <option key={value} value={String(value)}>{value} fps</option>
                    ))}
                  </select>
                </Field>
              )}
              {actionProfile && (
                <InlineNotice tone="review" title="Research model: read results as observations, not findings">
                  The MERL shelf-action model was trained on one overhead camera in a mock store. It analyses 5 Hz steps
                  using every decoded frame, reports uncalibrated scores for five shelf interactions, and warns when the
                  frame rate, aspect ratio or person visibility differ from training. Expect roughly real-time CPU cost.
                </InlineNotice>
              )}
              <Field label="Privacy mask version" htmlFor="zone-version" hint="Read from the active local configuration. Zone editing is not exposed in this build.">
                <input id="zone-version" value={zoneVersion || 'Loading…'} readOnly />
              </Field>
              <label className="checkbox-row checkbox-row--panel">
                <input
                  type="checkbox"
                  checked={retainEvidence}
                  onChange={(event) => setRetainEvidence(event.target.checked)}
                />
                <span>
                  Retain generated evidence clips
                  <small>Applies the configured local retention window. Metadata tombstones remain after expiry.</small>
                </span>
              </label>

              <div className="analysis-summary">
                <IconShield />
                <p>
                  {actionProfile
                    ? 'Shelf-action analysis groups predicted reaches, retracts and inspections into short interaction episodes with evidence clips. It never scores theft, identifies people or infers intent. A reviewer confirms or corrects each predicted action.'
                    : 'The baseline reports confirmed person tracks as descriptive activity segments. It does not score theft, identify people or infer intent. A reviewer decides what the context means.'}
                </p>
              </div>

              {Boolean(jobError) && (
                <InlineNotice tone="danger" title="Analysis job was not created">
                  {errorMessage(jobError)}
                </InlineNotice>
              )}

              <button
                className="button button--primary button--wide"
                type="button"
                disabled={starting || !zoneVersion}
                onClick={() => void startAnalysis()}
              >
                {starting ? 'Creating local job…' : actionProfile ? 'Queue shelf-action analysis' : 'Queue CPU analysis'}
              </button>
              <p className="small text-muted center-copy">
                A separate local worker processes the job. You can leave this page; progress is durable in SQLite.
              </p>
            </div>
          )}
        </section>
      </div>
    </div>
  )
}
