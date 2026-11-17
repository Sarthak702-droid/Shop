import { useEffect, useRef, useState } from 'react'
import {
  apiBlob,
  apiGet,
  apiGetWithHeaders,
  apiPost,
  apiRequestWithHeaders,
  errorMessage,
} from '../lib/api'
import { formatDateTime } from '../lib/format'
import { useApiResource } from '../lib/hooks'
import type {
  SourceHealthResponse,
  SourceListResponse,
  SourceSummary,
} from '../lib/types'
import {
  DataRow,
  EmptyState,
  ErrorPanel,
  Field,
  InlineNotice,
  LoadingPanel,
  Modal,
  PageHeader,
  SectionHeading,
  Spinner,
  StatusPill,
} from '../components/Common'
import {
  IconAlert,
  IconCamera,
  IconCheck,
  IconRefresh,
  IconShield,
} from '../components/Icons'

function sourceStatus(source: SourceSummary): { label: string; tone: 'success' | 'review' | 'muted' } {
  if (!source.approved_camera) return { label: 'Approval removed', tone: 'review' }
  if (source.state === 'active') return { label: source.active_job ? 'Analysis using camera' : 'Active', tone: 'success' }
  if (source.state === 'queued') return { label: 'Analysis queued', tone: 'muted' }
  if (source.state === 'cancelling') return { label: 'Cancelling analysis', tone: 'review' }
  if (source.state === 'degraded') return { label: 'Degraded', tone: 'review' }
  if (source.state === 'disconnected') return { label: 'Disconnected', tone: 'review' }
  return { label: 'Registered · not checked live', tone: 'muted' }
}

function CameraPreview({ source }: { source: SourceSummary }) {
  const [url, setUrl] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<unknown>(null)
  const [capturedAt, setCapturedAt] = useState<string | null>(null)
  const urlRef = useRef<string | null>(null)

  useEffect(() => () => {
    if (urlRef.current) URL.revokeObjectURL(urlRef.current)
  }, [])

  const capture = async () => {
    setLoading(true)
    setError(null)
    try {
      const blob = await apiBlob(`/sources/${source.source_id}/preview`)
      if (urlRef.current) URL.revokeObjectURL(urlRef.current)
      const next = URL.createObjectURL(blob)
      urlRef.current = next
      setUrl(next)
      setCapturedAt(new Date().toISOString())
    } catch (caught) {
      setError(caught)
      setUrl(null)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="camera-preview">
      <div className="camera-preview__stage">
        {url ? (
          <img src={url} alt={`Single local preview frame from ${source.label}`} />
        ) : (
          <div className="camera-preview__placeholder">
            <IconCamera />
            <span>{loading ? 'Opening the local device…' : 'No preview captured'}</span>
          </div>
        )}
        {loading && <div className="camera-preview__loading"><Spinner label="Capturing a local preview frame" /></div>}
      </div>
      <div className="camera-preview__meta">
        <p>
          {capturedAt ? `Captured ${formatDateTime(capturedAt)} · single frame, not a live stream` : 'Previews are captured on demand and are not stored.'}
        </p>
        <button className="button button--secondary button--small" type="button" onClick={() => void capture()} disabled={loading}>
          <IconRefresh /> {url ? 'Recapture' : 'Capture preview'}
        </button>
      </div>
      {Boolean(error) && (
        <InlineNotice tone="review" title="Preview unavailable">
          {errorMessage(error)}
          <p className="small">
            The device may be in use by another application, disconnected, or blocked by host permissions. The
            registered source remains in the local database.
          </p>
        </InlineNotice>
      )}
    </div>
  )
}

function CameraCard({
  source,
  onRename,
}: {
  source: SourceSummary
  onRename: (source: SourceSummary) => void
}) {
  const [probe, setProbe] = useState<SourceHealthResponse | null>(null)
  const [checking, setChecking] = useState(false)
  const [probeError, setProbeError] = useState<unknown>(null)
  const status = sourceStatus(source)

  const checkHealth = async () => {
    setChecking(true)
    setProbeError(null)
    try {
      const result = await apiGet<SourceHealthResponse>(`/sources/${source.source_id}/health`)
      setProbe(result)
    } catch (caught) {
      setProbeError(caught)
      setProbe(null)
    } finally {
      setChecking(false)
    }
  }

  const probeState = probe?.state ?? null

  return (
    <article className="camera-card panel">
      <div className="camera-card__head">
        <div className="camera-card__identity">
          <span className="camera-card__icon"><IconCamera /></span>
          <div>
            <h3>{source.label}</h3>
            <p>
              {source.camera?.label ?? 'Camera approval removed'}
              {source.camera ? ` · device ${source.camera.device_index}` : ''}
              {source.camera_key ? ` · ${source.camera_key}` : ''}
            </p>
          </div>
        </div>
        <div className="camera-card__pills">
          <StatusPill tone={status.tone} dot>{status.label}</StatusPill>
          {source.active_job && (
            <StatusPill tone="muted">Job {source.active_job.job_id.slice(0, 8)}… · {source.active_job.status}</StatusPill>
          )}
          {probeState && (
            <StatusPill tone={probeState === 'active' ? 'success' : 'review'}>
              Live probe: {probeState}
            </StatusPill>
          )}
        </div>
      </div>

      <div className="camera-card__actions">
        <button className="button button--ghost button--small" type="button" onClick={() => void checkHealth()} disabled={checking}>
          {checking ? 'Checking…' : 'Check device health'}
        </button>
        <button className="button button--ghost button--small" type="button" onClick={() => onRename(source)}>
          Rename
        </button>
      </div>

      {Boolean(probeError) && (
        <InlineNotice tone="review" title="Health check failed">
          {errorMessage(probeError)}
        </InlineNotice>
      )}
      {probe && (
        <div className="health-probe">
          <dl className="data-grid data-grid--compact">
            <DataRow label="Probe state">{probe.state}</DataRow>
            <DataRow label="Device present">{String(probe.probe.device_present ?? 'unknown')}</DataRow>
            <DataRow label="Opened">{String(probe.probe.opened ?? 'unknown')}</DataRow>
            <DataRow label="Frame read">{String(probe.probe.frame_read ?? 'unknown')}</DataRow>
            <DataRow label="Backend">{String(probe.probe.backend ?? 'unknown')}</DataRow>
            <DataRow label="Checked at">{formatDateTime(probe.checked_at)}</DataRow>
          </dl>
          <span className="provenance-tag">Actual measured result · single probe</span>
        </div>
      )}

      {source.approved_camera ? (
        <CameraPreview source={source} />
      ) : (
        <InlineNotice tone="review" title="Camera is no longer approved">
          Preview and analysis are blocked. Re-approve this camera key in the local configuration before use.
        </InlineNotice>
      )}
    </article>
  )
}

export function CamerasPage() {
  const sources = useApiResource<SourceListResponse>('/sources', { pollMs: 30_000 })
  const [registerOpen, setRegisterOpen] = useState(false)
  const [registerKey, setRegisterKey] = useState('')
  const [registerLabel, setRegisterLabel] = useState('')
  const [registerError, setRegisterError] = useState<unknown>(null)
  const [registering, setRegistering] = useState(false)
  const [renameTarget, setRenameTarget] = useState<SourceSummary | null>(null)
  const [renameLabel, setRenameLabel] = useState('')
  const [renameError, setRenameError] = useState<unknown>(null)
  const [renaming, setRenaming] = useState(false)

  const approvedUnregistered = (sources.data?.approved_cameras ?? []).filter((camera) => !camera.registered)
  const registeredSources = sources.data?.items ?? []

  const openRegister = (key: string) => {
    const camera = sources.data?.approved_cameras.find((item) => item.key === key)
    setRegisterKey(key)
    setRegisterLabel(camera?.label ?? '')
    setRegisterError(null)
    setRegisterOpen(true)
  }

  const registerCamera = async () => {
    setRegistering(true)
    setRegisterError(null)
    try {
      await apiPost('/sources', {
        camera_key: registerKey,
        ...(registerLabel.trim() ? { label: registerLabel.trim() } : {}),
      })
      setRegisterOpen(false)
      sources.reload()
    } catch (caught) {
      setRegisterError(caught)
    } finally {
      setRegistering(false)
    }
  }

  const openRename = (source: SourceSummary) => {
    setRenameTarget(source)
    setRenameLabel(source.label)
    setRenameError(null)
  }

  const renameSource = async () => {
    if (!renameTarget) return
    setRenaming(true)
    setRenameError(null)
    try {
      const fresh = await apiGetWithHeaders<{ source: SourceSummary }>(`/sources/${renameTarget.source_id}`)
      const etag = fresh.headers.get('ETag')
      if (!etag) throw new Error('The source did not return a version tag; reload before renaming.')
      await apiRequestWithHeaders(`/sources/${renameTarget.source_id}`, {
        method: 'PATCH',
        headers: { 'If-Match': etag },
        body: JSON.stringify({ label: renameLabel.trim() }),
      })
      setRenameTarget(null)
      sources.reload()
    } catch (caught) {
      setRenameError(caught)
    } finally {
      setRenaming(false)
    }
  }

  return (
    <div className="page-stack">
      <PageHeader
        eyebrow="Local source"
        title="Cameras"
        subtitle="Only camera keys explicitly approved in local configuration can be registered, previewed or analysed."
        actions={
          <button
            className="button button--primary"
            type="button"
            disabled={approvedUnregistered.length === 0}
            onClick={() => openRegister(approvedUnregistered[0]?.key ?? '')}
          >
            <IconCamera /> Register approved camera
          </button>
        }
      />

      <div className="provenance-line">
        <span className="provenance-tag">Actual measured result</span>
        <span>Device health and previews are probed on demand on this host. “Registered” alone does not imply that a camera is online.</span>
      </div>

      <InlineNotice tone="info" title="Approved device allowlist">
        Camera keys and device indexes are read from <code>configs/shop.yaml</code>. Editing that file requires a local
        restart. RTSP is deferred and disabled in this build; audio is disabled by policy.
      </InlineNotice>

      {sources.loading && !sources.data && <LoadingPanel label="Loading local camera registrations…" rows={3} />}
      {sources.error && !sources.data && (
        <ErrorPanel error={sources.error} onRetry={sources.reload} title="Camera registrations could not load" />
      )}

      {sources.data && registeredSources.length === 0 && (
        <section className="panel">
          <EmptyState icon={<IconCamera />} title="No camera source is registered">
            <p>
              Register an approved local device below. The first browser preview opens the device only for a single
              snapshot; it is not continuous capture.
            </p>
          </EmptyState>
        </section>
      )}

      <div className="camera-grid">
        {registeredSources.map((source) => (
          <CameraCard key={source.source_id} source={source} onRename={openRename} />
        ))}
      </div>

      <section className="panel">
        <SectionHeading
          title="Approved camera keys"
          detail="Configured allowlist. Registering creates a local source record; it does not grant new permissions."
          action={<span className="provenance-tag">Configuration declaration</span>}
        />
        <div className="camera-allowlist">
          {(sources.data?.approved_cameras ?? []).map((camera) => (
            <div className="allowlist-row" key={camera.key}>
              <span className={`allowlist-icon${camera.registered ? ' allowlist-icon--on' : ''}`}>
                {camera.registered ? <IconCheck /> : <IconShield />}
              </span>
              <div>
                <strong>{camera.label}</strong>
                <p>key <code>{camera.key}</code> · device index {camera.device_index}</p>
              </div>
              <StatusPill tone={camera.registered ? 'teal' : 'muted'}>
                {camera.registered ? 'Registered' : 'Approved, not registered'}
              </StatusPill>
              {!camera.registered && (
                <button className="button button--secondary button--small" type="button" onClick={() => openRegister(camera.key)}>
                  Register
                </button>
              )}
            </div>
          ))}
          {(sources.data?.approved_cameras.length ?? 0) === 0 && (
            <p className="muted-copy">No cameras are approved in local configuration.</p>
          )}
        </div>
      </section>

      <section className="panel">
        <SectionHeading title="Source boundaries" detail="What this build does and does not do with a camera." />
        <div className="boundary-grid">
          <div><IconCheck /><strong>Local device capture</strong><p>OpenCV reads approved device indexes on this host only.</p></div>
          <div><IconAlert /><strong>No RTSP yet</strong><p>Network cameras are deferred; do not configure credentials for this build.</p></div>
          <div><IconAlert /><strong>No audio path</strong><p>Audio is disabled by policy and never extracted or analysed.</p></div>
          <div><IconShield /><strong>No identity features</strong><p>No face recognition or appearance matching is implemented.</p></div>
        </div>
      </section>

      {registerOpen && (
        <Modal
          title="Register an approved camera"
          onClose={() => setRegisterOpen(false)}
          footer={
            <>
              <button className="button button--ghost" type="button" onClick={() => setRegisterOpen(false)}>Cancel</button>
              <button className="button button--primary" type="button" onClick={() => void registerCamera()} disabled={registering || !registerKey}>
                {registering ? 'Registering locally…' : 'Register source'}
              </button>
            </>
          }
        >
          <p className="modal-intro">
            The server resolves the camera key against its own allowlist. A key not present in configuration is
            rejected even if it is submitted by this form.
          </p>
          <Field label="Approved camera key" htmlFor="camera-key" hint="Only keys shown in the allowlist can be registered.">
            <select id="camera-key" value={registerKey} onChange={(event) => setRegisterKey(event.target.value)}>
              {approvedUnregistered.map((camera) => (
                <option key={camera.key} value={camera.key}>{camera.label} · device {camera.device_index}</option>
              ))}
            </select>
          </Field>
          <Field label="Local display label" htmlFor="camera-label" optional hint="Shown in the review queue. It does not change the device or its permissions.">
            <input id="camera-label" value={registerLabel} maxLength={100} onChange={(event) => setRegisterLabel(event.target.value)} />
          </Field>
          {Boolean(registerError) && <InlineNotice tone="danger" title="Camera was not registered">{errorMessage(registerError)}</InlineNotice>}
        </Modal>
      )}

      {renameTarget && (
        <Modal
          title="Rename registered source"
          onClose={() => setRenameTarget(null)}
          footer={
            <>
              <button className="button button--ghost" type="button" onClick={() => setRenameTarget(null)}>Cancel</button>
              <button
                className="button button--primary"
                type="button"
                onClick={() => void renameSource()}
                disabled={renaming || !renameLabel.trim()}
              >
                {renaming ? 'Saving locally…' : 'Save label'}
              </button>
            </>
          }
        >
          <p className="modal-intro">
            Metadata updates use the source's version tag to prevent overwriting a concurrent change. The device key is
            immutable.
          </p>
          <Field label="Display label" htmlFor="rename-label" hint="1–100 characters." error={renameError ? errorMessage(renameError) : undefined}>
            <input id="rename-label" value={renameLabel} maxLength={100} onChange={(event) => setRenameLabel(event.target.value)} />
          </Field>
        </Modal>
      )}
    </div>
  )
}
