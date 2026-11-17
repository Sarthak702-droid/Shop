import { useMemo, useState } from 'react'
import {
  apiPost,
  errorMessage,
  getReviewer,
  getToken,
  setReviewer,
  setToken,
} from '../lib/api'
import {
  formatBytes,
  formatDateTime,
  formatDuration,
  formatPercent,
  workerLabel,
} from '../lib/format'
import { useApiResource } from '../lib/hooks'
import type {
  AuditListResponse,
  Capabilities,
  ConfigResponse,
  RetentionResponse,
} from '../lib/types'
import {
  DataRow,
  ErrorPanel,
  Field,
  InlineNotice,
  LoadingPanel,
  Modal,
  PageHeader,
  SectionHeading,
  StatusPill,
} from '../components/Common'
import {
  IconAlert,
  IconCheck,
  IconDatabase,
  IconRefresh,
  IconSettings,
  IconShield,
  IconTrash,
} from '../components/Icons'

function auditMetadata(metadata: Record<string, unknown>): string {
  const entries = Object.entries(metadata)
  if (entries.length === 0) return 'No metadata'
  return entries
    .slice(0, 4)
    .map(([key, value]) => `${key}: ${typeof value === 'object' ? JSON.stringify(value) : String(value)}`)
    .join(' · ')
}

export function SettingsPage() {
  const config = useApiResource<ConfigResponse>('/config')
  const capabilities = useApiResource<Capabilities>('/capabilities')
  const audit = useApiResource<AuditListResponse>('/audit?limit=25')
  const [reviewerDraft, setReviewerDraft] = useState(getReviewer())
  const [tokenDraft, setTokenDraft] = useState('')
  const [savedMessage, setSavedMessage] = useState('')
  const [retentionOpen, setRetentionOpen] = useState(false)
  const [runningRetention, setRunningRetention] = useState(false)
  const [retentionError, setRetentionError] = useState<unknown>(null)
  const [retentionResult, setRetentionResult] = useState<RetentionResponse | null>(null)
  const [tokenError, setTokenError] = useState('')

  const configData = config.data
  const reviewers = useMemo(() => configData?.security.reviewers ?? [], [configData])
  const reviewerAllowed = reviewers.includes(reviewerDraft.trim())

  const saveReviewer = () => {
    setSavedMessage('')
    const value = reviewerDraft.trim()
    if (!value) {
      setSavedMessage('Enter the reviewer identity that is allowed by local configuration.')
      return
    }
    setReviewer(value)
    setSavedMessage(`Reviewer identity stored in this browser as “${value}”.`)
    audit.reload()
  }

  const saveToken = () => {
    setTokenError('')
    if (!tokenDraft.trim()) {
      setTokenError('Enter the local API token, or use “Clear stored token”.')
      return
    }
    setToken(tokenDraft.trim())
    setTokenDraft('')
    setSavedMessage('Local API token stored in this browser. Subsequent requests use it.')
    config.reload()
    capabilities.reload()
    audit.reload()
  }

  const clearToken = () => {
    setToken('')
    setTokenDraft('')
    setSavedMessage('Stored API token cleared from this browser.')
  }

  const runRetention = async () => {
    setRunningRetention(true)
    setRetentionError(null)
    try {
      const result = await apiPost<RetentionResponse>('/maintenance/retention', {})
      setRetentionResult(result)
      setRetentionOpen(false)
      audit.reload()
    } catch (caught) {
      setRetentionError(caught)
    } finally {
      setRunningRetention(false)
    }
  }

  return (
    <div className="page-stack">
      <PageHeader
        eyebrow="Workspace settings"
        title="Settings"
        subtitle="Local configuration, reviewer identity, retention and audit. Secrets are never rendered back by the API."
        actions={
          <button
            className="button button--ghost"
            type="button"
            onClick={() => { config.reload(); capabilities.reload(); audit.reload() }}
          >
            <IconRefresh /> Refresh
          </button>
        }
      />

      <div className="provenance-line">
        <span className="provenance-tag">Configuration declaration</span>
        <span>Limits shown below are proposed starting settings. Measured results appear on jobs, not here.</span>
      </div>

      {savedMessage && (
        <InlineNotice tone="success" title="Local browser setting saved">
          {savedMessage}
        </InlineNotice>
      )}
      {retentionResult && (
        <InlineNotice tone="success" title="Retention maintenance applied">
          {retentionResult.clips_expired} expired clip{retentionResult.clips_expired === 1 ? '' : 's'} removed ·{' '}
          {retentionResult.events_updated} event tombstone{retentionResult.events_updated === 1 ? '' : 's'} updated ·{' '}
          {retentionResult.file_cleanup_failures} file cleanup failure{retentionResult.file_cleanup_failures === 1 ? '' : 's'}.
          Applied {formatDateTime(retentionResult.applied_at)}.
        </InlineNotice>
      )}

      <div className="settings-grid">
        <section className="panel">
          <SectionHeading title="Reviewer identity" detail="Sent as a request header and checked against the server's local allowlist." />
          <Field
            label="Reviewer identifier"
            htmlFor="reviewer-id"
            hint="This identifies who recorded a review in the local audit trail. It is not authentication on its own."
          >
            <input
              id="reviewer-id"
              value={reviewerDraft}
              maxLength={100}
              placeholder="local-reviewer"
              onChange={(event) => setReviewerDraft(event.target.value)}
            />
          </Field>
          <div className="settings-actions">
            <button className="button button--primary button--small" type="button" onClick={saveReviewer}>Store in this browser</button>
            <StatusPill tone={reviewerAllowed ? 'success' : 'review'} dot>
              {reviewers.length === 0 ? 'Allowlist loading' : reviewerAllowed ? 'Matches local allowlist' : 'Not in local allowlist'}
            </StatusPill>
          </div>
          <dl className="data-grid data-grid--compact">
            <DataRow label="Server allowlist">
              {reviewers.length ? reviewers.join(', ') : 'Not reported'}
            </DataRow>
            <DataRow label="Unknown reviewer behaviour">
              The API rejects a reviewer identity that is not on the local allowlist.
            </DataRow>
          </dl>
        </section>

        <section className="panel">
          <SectionHeading title="API access token" detail="Optional local bearer token. It is stored only in this browser and the server environment." />
          {configData && (
            <p className="settings-status">
              <StatusPill tone={configData.has_api_token ? 'teal' : 'muted'} dot>
                {configData.has_api_token ? 'Token required by server' : 'No token configured'}
              </StatusPill>
              {getToken() && <StatusPill tone="muted">A token is stored in this browser</StatusPill>}
            </p>
          )}
          <Field label="Bearer token" htmlFor="settings-token" hint="The API never returns the configured token value." error={tokenError || undefined}>
            <input
              id="settings-token"
              type="password"
              value={tokenDraft}
              autoComplete="off"
              placeholder={getToken() ? 'Token is already stored' : 'Enter SHOP_API_TOKEN'}
              onChange={(event) => setTokenDraft(event.target.value)}
            />
          </Field>
          <div className="settings-actions">
            <button className="button button--primary button--small" type="button" onClick={saveToken}>Store token</button>
            <button className="button button--ghost button--small" type="button" onClick={clearToken}>Clear stored token</button>
          </div>
          <InlineNotice tone="info">
            The token is sent only to this same-origin local API. It is not displayed by the server and is not part of
            any exported data.
          </InlineNotice>
        </section>
      </div>

      <section className="panel">
        <SectionHeading
          title="Retention and deletion"
          detail="Evidence expires only through the configured local window or an explicit authorized deletion."
          action={<span className="provenance-tag">Configuration declaration</span>}
        />
        {configData ? (
          <>
            <div className="retention-grid">
              <MetricTile label="Evidence clips" value={`${configData.storage.evidence_days} days`} detail="Generated evidence media and thumbnails." />
              <MetricTile label="Metadata records" value={`${configData.storage.metadata_days} days`} detail="Configured metadata retention window." />
              <MetricTile label="Audit records" value={`${configData.storage.audit_days} days`} detail="Append-only local audit trail." />
              <MetricTile label="Free-space reserve" value={formatBytes(configData.storage.min_free_bytes)} detail="Uploads and analysis stop below this reserve." />
            </div>
            <div className="retention-actions">
              <div>
                <h3>Apply evidence expiry now</h3>
                <p>
                  This explicit operator action removes clips whose retention deadline has passed. Event metadata remains
                  as a tombstone; individual events can also be deleted from their detail view.
                </p>
              </div>
              <button className="button button--secondary" type="button" onClick={() => setRetentionOpen(true)}>
                <IconTrash /> Run retention maintenance
              </button>
            </div>
          </>
        ) : config.loading ? (
          <LoadingPanel rows={2} />
        ) : (
          <ErrorPanel error={config.error} onRetry={config.reload} title="Retention configuration could not load" />
        )}
      </section>

      <div className="settings-grid">
        <section className="panel">
          <SectionHeading title="Local runtime" detail="Read from the active configuration and host process." />
          {config.loading && !configData && <LoadingPanel rows={2} />}
          {config.error && !configData && <ErrorPanel error={config.error} onRetry={config.reload} />}
          {configData && (
            <dl className="data-grid data-grid--compact">
              <DataRow label="Bind address"><code>{configData.security.bind_host}</code></DataRow>
              <DataRow label="Device">{configData.runtime.device}</DataRow>
              <DataRow label="Inference threads">{configData.runtime.inference_threads}</DataRow>
              <DataRow label="Active job limit">{configData.runtime.active_jobs}</DataRow>
              <DataRow label="Queued job limit">{configData.runtime.queued_jobs}</DataRow>
              <DataRow label="Development models allowed">{configData.runtime.allow_development_models ? 'Yes' : 'No'}</DataRow>
              <DataRow label="Audio">{configData.security.audio_enabled ? 'Enabled' : 'Disabled by policy'}</DataRow>
              <DataRow label="Managed storage">{configData.storage.root}</DataRow>
              <DataRow label="Auto-migrate on API start">{configData.storage.auto_migrate ? 'Yes' : 'No'}</DataRow>
              <DataRow label="Worker">
                {configData.worker ? `${workerLabel(configData.worker.status)} · ${configData.worker.detail}` : 'Not reported'}
              </DataRow>
              <DataRow label="Worker last seen">{formatDateTime(configData.worker?.last_seen_at)}</DataRow>
            </dl>
          )}
        </section>

        <section className="panel">
          <SectionHeading title="Analysis and policy" detail="Proposed starting limits for the local CPU profile." />
          {configData && (
            <dl className="data-grid data-grid--compact">
              <DataRow label="Sampling">{configData.analysis.sampling_fps} fps maximum</DataRow>
              <DataRow label="Long edge">{configData.analysis.long_edge_px} px</DataRow>
              <DataRow label="Person score threshold">{configData.analysis.person_score_threshold}</DataRow>
              <DataRow label="Tracker weights">IoU {configData.analysis.tracker_iou_weight} · motion {configData.analysis.tracker_motion_weight}</DataRow>
              <DataRow label="Confirmation hits">{configData.analysis.tracker_confirm_hits}</DataRow>
              <DataRow label="Prediction horizon">{formatDuration(configData.analysis.tracker_max_prediction_ms)} confirmed · {formatDuration(configData.analysis.tracker_tentative_max_prediction_ms)} tentative</DataRow>
              <DataRow label="Event dwell">{formatDuration(configData.analysis.event_min_dwell_ms)} minimum · {formatDuration(configData.analysis.event_gap_split_ms)} gap split</DataRow>
              <DataRow label="Evidence context">{formatDuration(configData.analysis.evidence_context_before_ms)} before · {formatDuration(configData.analysis.evidence_context_after_ms)} after</DataRow>
              <DataRow label="Policy version"><code>{configData.policy.version}</code></DataRow>
              <DataRow label="Score semantics">{configData.policy.score_semantics ?? 'null · no activity score in this build'}</DataRow>
              <DataRow label="Zones version"><code>{configData.zones.version}</code></DataRow>
              <DataRow label="Privacy mask">{(configData.zones.privacy_mask?.length ?? 0) > 0 ? `${configData.zones.privacy_mask.length} points configured` : 'No mask polygon configured'}</DataRow>
            </dl>
          )}
        </section>
      </div>

      <section className="panel">
        <SectionHeading
          title="Audit trail"
          detail="Append-only local records for uploads, jobs, reviews, settings changes and deletions."
          action={
            <span className="provenance-tag">Actual measured result</span>
          }
        />
        {audit.loading && !audit.data && <LoadingPanel rows={3} />}
        {audit.error && !audit.data && <ErrorPanel error={audit.error} onRetry={audit.reload} title="Audit trail could not load" />}
        {audit.data && audit.data.items.length === 0 && (
          <p className="muted-copy">No audit records have been written yet.</p>
        )}
        {audit.data && audit.data.items.length > 0 && (
          <div className="audit-list">
            {audit.data.items.map((entry) => (
              <div className="audit-row" key={entry.audit_id}>
                <span className="audit-row__dot" aria-hidden="true" />
                <div className="audit-row__body">
                  <div className="audit-row__top">
                    <strong>{entry.action.replaceAll('_', ' ')}</strong>
                    <time>{formatDateTime(entry.created_at)}</time>
                  </div>
                  <p>
                    {entry.actor} · {entry.object_type}
                    {entry.object_id ? ` ${entry.object_id.slice(0, 8)}…` : ''}
                  </p>
                  <p className="audit-row__meta">{auditMetadata(entry.metadata)}</p>
                  {entry.request_id && <p className="audit-row__meta">request {entry.request_id}</p>}
                </div>
              </div>
            ))}
          </div>
        )}
        {audit.data?.has_more && (
          <p className="small text-muted">
            Showing the 25 most recent audit records. The API supports cursor pagination for older entries.
            {audit.data.next_cursor ? ' Cursor is available in the API response.' : ''}
          </p>
        )}
      </section>

      <section className="panel">
        <SectionHeading title="Operating vocabulary" detail="Words the interface uses and words it refuses to use." />
        <div className="vocabulary-grid">
          <div>
            <h3><IconCheck /> Permitted</h3>
            <ul>
              <li>Observed activity</li>
              <li>Review needed</li>
              <li>Item visibility lost</li>
              <li>Insufficient evidence</li>
              <li>Inconclusive</li>
              <li>Analysis unavailable</li>
            </ul>
          </div>
          <div>
            <h3><IconAlert /> Not used as conclusions</h3>
            <ul>
              <li>“Shoplifter” or “thief”</li>
              <li>“Guilty” or “caught”</li>
              <li>“Theft confirmed”</li>
              <li>Uncalibrated “risk score”</li>
              <li>Identity or intent inference</li>
              <li>Enforcement recommendation</li>
            </ul>
          </div>
        </div>
        <div className="privacy-note">
          <IconShield />
          <p>
            Shop is a review assistant. A human reviewer decides whether context supports a benign explanation, needs
            more review, or is inconclusive. No enforcement action is available in this build.
          </p>
        </div>
      </section>

      <details className="panel raw-config">
        <summary><IconSettings /> Raw redacted configuration</summary>
        <p className="small text-muted">
          This is the API-safe projection. Absolute storage paths and secret values are excluded.
        </p>
        {configData && <pre>{JSON.stringify(configData, null, 2)}</pre>}
        {capabilities.data && (
          <>
            <h3>Capabilities projection</h3>
            <pre>{JSON.stringify(capabilities.data, null, 2)}</pre>
          </>
        )}
      </details>

      {retentionOpen && (
        <Modal
          title="Apply evidence retention now?"
          onClose={() => setRetentionOpen(false)}
          footer={
            <>
              <button className="button button--ghost" type="button" onClick={() => setRetentionOpen(false)}>Cancel</button>
              <button className="button button--danger" type="button" onClick={() => void runRetention()} disabled={runningRetention}>
                {runningRetention ? 'Applying locally…' : 'Apply retention'}
              </button>
            </>
          }
        >
          <p>
            Clips whose retention deadline has already passed will be deleted from local storage. Event metadata stays
            as a tombstone and the action is recorded in the audit trail. This does not delete unexpired clips.
          </p>
          <p className="small text-muted">
            The API never silently extends retention and never uploads evidence anywhere.
          </p>
          {Boolean(retentionError) && <InlineNotice tone="danger">{errorMessage(retentionError)}</InlineNotice>}
        </Modal>
      )}
    </div>
  )
}

function MetricTile({ label, value, detail }: { label: string; value: string; detail: string }) {
  return (
    <div className="metric-tile">
      <span className="metric-tile__label">{label}</span>
      <strong>{value}</strong>
      <p>{detail}</p>
    </div>
  )
}
