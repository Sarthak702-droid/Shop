import { useApiResource } from '../lib/hooks'
import type { Capabilities, ModelsResponse, ModelStatus } from '../lib/types'
import {
  DataRow,
  EmptyState,
  ErrorPanel,
  InlineNotice,
  LoadingPanel,
  PageHeader,
  SectionHeading,
  StatusPill,
} from '../components/Common'
import {
  IconAlert,
  IconCheck,
  IconCpu,
  IconModel,
  IconShield,
} from '../components/Icons'

const ROLE_LABELS: Record<string, { title: string; purpose: string }> = {
  person_detector: {
    title: 'Person detector',
    purpose: 'Locates people to maintain session-local tracks. It does not classify behaviour, identity or intent.',
  },
  retail_detector: {
    title: 'Retail object detector',
    purpose: 'Would locate visible products, bags and baskets. No trained artifact exists in this build.',
  },
  pose_reference: {
    title: 'Pose reference',
    purpose: 'Would estimate keypoints for movement features. Reference only; not installed or enabled.',
  },
  temporal_activity_model: {
    title: 'Temporal activity model',
    purpose: 'Would classify a causal feature window into observable actions. Proposed architecture only.',
  },
}

function commercialTone(status: string): 'success' | 'review' | 'muted' | 'danger' {
  if (status === 'approved') return 'success'
  if (status === 'development_only') return 'review'
  if (status === 'pending') return 'review'
  if (status === 'not_approved') return 'muted'
  return 'danger'
}

function commercialLabel(status: string): string {
  return {
    approved: 'Commercial use approved',
    development_only: 'Development use only',
    pending: 'Approval pending',
    not_approved: 'Not approved',
    unknown: 'Rights unknown',
  }[status] ?? status
}

function detailValue(value: unknown): string {
  if (value === null || value === undefined || value === '') return '—'
  if (typeof value === 'boolean') return value ? 'Yes' : 'No'
  if (typeof value === 'string') return value
  if (typeof value === 'number') return Number.isFinite(value) ? String(value) : '—'
  if (Array.isArray(value)) {
    return value.every((item) => typeof item !== 'object')
      ? value.join(', ')
      : JSON.stringify(value)
  }
  return JSON.stringify(value)
}

function ModelCard({ status }: { status: ModelStatus }) {
  const labels = ROLE_LABELS[status.role] ?? { title: status.role, purpose: '' }
  const details = status.details ?? {}
  const capabilityTone = status.available ? 'success' : status.enabled ? 'review' : 'muted'
  const capabilityLabel = status.available
    ? 'Available on this host'
    : status.enabled
      ? 'Enabled but unavailable'
      : 'Disabled by configuration'
  return (
    <article className="model-card panel">
      <div className="model-card__head">
        <span className={`model-card__icon${status.available ? ' model-card__icon--on' : ''}`}>
          {status.available ? <IconModel /> : <IconShield />}
        </span>
        <div>
          <h3>{labels.title}</h3>
          <p>{labels.purpose}</p>
        </div>
        <StatusPill tone={capabilityTone} dot>{capabilityLabel}</StatusPill>
      </div>
      <div className="model-card__pills">
        <StatusPill tone={commercialTone(status.commercial_status)}>
          {commercialLabel(status.commercial_status)}
        </StatusPill>
        <StatusPill tone={status.available ? 'teal' : 'muted'}>
          Capability: {status.capability.replaceAll('_', ' ')}
        </StatusPill>
      </div>
      <InlineNotice tone={status.available ? 'neutral' : 'review'} title="Capability statement">
        {status.reason}
      </InlineNotice>
      <dl className="data-grid data-grid--compact">
        <DataRow label="Model ID"><code>{status.model_id}</code></DataRow>
        <DataRow label="Version">{status.version}</DataRow>
        <DataRow label="Architecture"><code className="breakable">{status.architecture ?? 'Not declared'}</code></DataRow>
        <DataRow label="Artifact SHA-256">
          {status.sha256 ? <code className="hash-value hash-value--short">{status.sha256}</code> : 'No approved artifact'}
        </DataRow>
        <DataRow label="Rights declaration">{commercialLabel(status.commercial_status)}</DataRow>
        <DataRow label="Evaluation on this host">{details.measured_on_this_host === true ? 'Reported as measured' : 'Not measured on this host'}</DataRow>
        <DataRow label="Feature schema">{detailValue(details.feature_schema)}</DataRow>
        {details.feature_count !== undefined && details.feature_count !== null && (
          <DataRow label="Feature contract">
            {detailValue(details.feature_count)} features · {detailValue(details.sample_rate_hz)} Hz · {detailValue(details.window_samples)} samples
          </DataRow>
        )}
        {Array.isArray(details.dilations) && (
          <DataRow label="Causal dilations">{detailValue(details.dilations)}</DataRow>
        )}
        {Boolean(details.class_order) && <DataRow label="Class order">{detailValue(details.class_order)}</DataRow>}
        {Boolean(details.weights_source) && <DataRow label="Weights source">{detailValue(details.weights_source)}</DataRow>}
      </dl>
      {typeof details.approval_notes === 'string' && (
        <div className="model-notes">
          <h4>Approval notes</h4>
          <p>{details.approval_notes}</p>
        </div>
      )}
      {Boolean(details.license) && typeof details.license === 'object' && (
        <details className="model-details">
          <summary>License and source rights</summary>
          <dl className="data-grid data-grid--compact">
            {Object.entries(details.license as Record<string, unknown>).map(([key, value]) => (
              <DataRow key={key} label={key.replaceAll('_', ' ')}>{detailValue(value)}</DataRow>
            ))}
          </dl>
        </details>
      )}
      {Boolean(details.output) && typeof details.output === 'object' && (
        <details className="model-details">
          <summary>Output contract</summary>
          <dl className="data-grid data-grid--compact">
            {Object.entries(details.output as Record<string, unknown>).map(([key, value]) => (
              <DataRow key={key} label={key.replaceAll('_', ' ')}>{detailValue(value)}</DataRow>
            ))}
          </dl>
        </details>
      )}
      {Boolean(details.preprocessing) && typeof details.preprocessing === 'object' && (
        <details className="model-details">
          <summary>Preprocessing contract</summary>
          <dl className="data-grid data-grid--compact">
            {Object.entries(details.preprocessing as Record<string, unknown>).map(([key, value]) => (
              <DataRow key={key} label={key.replaceAll('_', ' ')}>{detailValue(value)}</DataRow>
            ))}
          </dl>
        </details>
      )}
      {(details.evaluation_report === null || details.evaluation_report === undefined) && (
        <p className="model-footnote">
          No evaluation report is attached. The interface does not display accuracy or fairness claims for this
          artifact.
        </p>
      )}
    </article>
  )
}

export function ModelsPage() {
  const models = useApiResource<ModelsResponse>('/models')
  const capabilities = useApiResource<Capabilities>('/capabilities')

  if (models.loading && !models.data) return <LoadingPanel label="Inspecting local model manifests…" rows={4} />
  if (models.error && !models.data) {
    return <ErrorPanel error={models.error} onRetry={models.reload} title="Model manifests could not load" />
  }

  const data = models.data
  const list = data ? [data.person, data.retail, data.pose, data.temporal] : []
  const cpuChecks = capabilities.data?.cpu_verification.checks ?? []
  const features = capabilities.data?.features ?? {}

  return (
    <div className="page-stack">
      <PageHeader
        eyebrow="Model registry"
        title="Models and rights"
        subtitle="Every model is described by a local manifest with an artifact hash, rights declaration and approval gate. Disabled modules stay visibly disabled."
        actions={
          <button className="button button--ghost" type="button" onClick={() => { models.reload(); capabilities.reload() }}>
            Refresh manifests
          </button>
        }
      />

      <div className="provenance-line">
        <span className="provenance-tag">Manifest declaration</span>
        <span>Rights and commercial status are declarations from local files, not measured accuracy claims.</span>
      </div>

      {data && (
        <InlineNotice tone="review" title="Baseline capability boundary">
          <p><strong>{data.bundle}</strong></p>
          <p>{data.policy.detail}</p>
          <p className="small">
            The baseline event type is <code>{data.policy.baseline_event_type}</code>; review score semantics are{' '}
            <code>{data.policy.score_semantics ?? 'null'}</code>.
          </p>
        </InlineNotice>
      )}

      {list.length === 0 && (
        <EmptyState icon={<IconModel />} title="No model manifests loaded">
          <p>Start the local API with a valid configuration to inspect the model registry.</p>
        </EmptyState>
      )}

      <div className="model-grid">
        {list.map((status) => <ModelCard key={status.role} status={status} />)}
      </div>

      <div className="model-grid">
        <section className="panel">
          <SectionHeading
            title="Analysis feature contract"
            detail="Disabled modules still declare the contract they would consume. Nothing is inferred from an untrained model."
            action={<span className="provenance-tag">Specification declaration</span>}
          />
          <dl className="data-grid data-grid--compact">
            <DataRow label="Feature schema">{String((data?.temporal.details.feature_schema as string | undefined) ?? 'shop_features_v1')}</DataRow>
            <DataRow label="Feature count">{String((data?.temporal.details.feature_count as number | undefined) ?? 145)}</DataRow>
            <DataRow label="Sample rate">{String((data?.temporal.details.sample_rate_hz as number | undefined) ?? 5)} Hz</DataRow>
            <DataRow label="Window">{String((data?.temporal.details.window_samples as number | undefined) ?? 29)} samples</DataRow>
            <DataRow label="Temporal architecture">Causal TCN · dilations 1 / 2 / 4 · two convolutions per level</DataRow>
            <DataRow label="Runtime gate">Temporal model disabled until approved data, training, calibration and CPU evaluation exist.</DataRow>
          </dl>
          <div className="feature-boundaries">
            {Object.entries(features).map(([key, value]) => (
              <span key={key} className={`feature-chip${value ? ' feature-chip--on' : ''}`}>
                {value ? <IconCheck /> : <IconAlert />}
                {key.replaceAll('_', ' ')}
              </span>
            ))}
          </div>
          <p className="model-footnote">
            These values are the documented feature contract for future work. They are not predictions and no
            temporal window is scored in this build.
          </p>
        </section>

        <section className="panel">
          <SectionHeading
            title="CPU runtime verification"
            detail="Checks run against the live local process when this page requests capabilities."
            action={<span className="provenance-tag">Actual measured result</span>}
          />
          {capabilities.loading && !capabilities.data && <LoadingPanel rows={2} />}
          {capabilities.error && !capabilities.data && <ErrorPanel error={capabilities.error} onRetry={capabilities.reload} />}
          {capabilities.data && (
            <>
              <div className="runtime-head">
                <StatusPill tone={capabilities.data.cpu_verification.passed ? 'success' : 'danger'} dot>
                  {capabilities.data.cpu_verification.passed ? 'CPU-only checks pass' : 'CPU checks fail'}
                </StatusPill>
                <span>{capabilities.data.host.cpu_model}</span>
              </div>
              <div className="check-list">
                {cpuChecks.map((check) => (
                  <div className="check-row" key={check.name}>
                    <span className={`check-mark check-mark--${check.passed ? 'pass' : 'fail'}`} aria-hidden="true">
                      {check.passed ? <IconCheck /> : <IconAlert />}
                    </span>
                    <div>
                      <div className="check-row__top">
                        <strong>{check.name.replaceAll('_', ' ')}</strong>
                        <StatusPill tone={check.passed ? 'success' : 'review'}>{check.passed ? 'Pass' : 'Fail'}</StatusPill>
                      </div>
                      <p>Observed <code>{JSON.stringify(check.observed)}</code> · expectation <code>{check.expectation}</code></p>
                    </div>
                  </div>
                ))}
              </div>
              <dl className="data-grid data-grid--compact">
                <DataRow label="Torch">{capabilities.data.host.torch}</DataRow>
                <DataRow label="TorchVision">{capabilities.data.host.torchvision}</DataRow>
                <DataRow label="CUDA build">{capabilities.data.host.cuda_build ?? 'None'}</DataRow>
                <DataRow label="Device">{capabilities.data.host.device}</DataRow>
              </dl>
            </>
          )}
          <div className="profile-footnote">
            <IconCpu />
            <span>No CUDA toolkit is installed or required. Model inference runs on CPU in the local worker.</span>
          </div>
        </section>
      </div>
    </div>
  )
}
