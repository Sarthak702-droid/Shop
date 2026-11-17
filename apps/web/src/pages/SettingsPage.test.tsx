import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import { SettingsPage } from './SettingsPage'
import {
  CREATED_AT,
  makeAudit,
  makeCapabilities,
  makeConfig,
} from '../test/fixtures'
import { apiErrorBody, mockFetch, renderAt } from '../test/utils'

function setup(config = makeConfig()) {
  return mockFetch((url, init) => {
    if (url.includes('/config')) return { body: config }
    if (url.includes('/capabilities')) return { body: makeCapabilities() }
    if (url.includes('/audit')) {
      return {
        body: makeAudit([
          {
            audit_id: 'aud-1',
            actor: 'alex',
            action: 'review_recorded',
            object_type: 'event',
            object_id: 'evt-11111111-1111-4111-8111-111111111111',
            request_id: 'req-audit-1',
            metadata: { decision: 'benign' },
            created_at: CREATED_AT,
          },
        ]),
      }
    }
    if (url.includes('/maintenance/retention') && init.method === 'POST') {
      return {
        body: {
          clips_expired: 2,
          events_updated: 1,
          file_cleanup_failures: 0,
          applied_at: CREATED_AT,
          request_id: 'req-retention-1',
        },
      }
    }
    return { status: 404, body: apiErrorBody('NOT_FOUND', 'Not found.') }
  })
}

describe('SettingsPage', () => {
  it('renders redacted configuration and never the token value', async () => {
    setup()
    renderAt(<SettingsPage />, { route: '/settings', path: '/settings' })
    expect(await screen.findByText('Managed storage')).toBeInTheDocument()
    expect(screen.getByText('/managed/local/storage')).toBeInTheDocument()
    expect(screen.getByText('Disabled by policy')).toBeInTheDocument()
    expect(screen.getByText('Token required by server')).toBeInTheDocument()
    expect(screen.getByText('Worker ready · Worker heartbeat is fresh.')).toBeInTheDocument()
    expect(screen.getByText('alex, sam')).toBeInTheDocument()
    expect(screen.getByText(/Absolute storage paths and secret values are excluded/i)).toBeInTheDocument()
  })

  it('checks the reviewer against the server allowlist before storing it', async () => {
    setup()
    renderAt(<SettingsPage />, { route: '/settings', path: '/settings' })
    const input = await screen.findByLabelText(/Reviewer identifier/)
    await userEvent.type(input, 'mallory')
    expect(await screen.findByText('Not in local allowlist')).toBeInTheDocument()
    await userEvent.clear(input)
    await userEvent.type(input, 'alex')
    expect(await screen.findByText('Matches local allowlist')).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'Store in this browser' }))
    expect(screen.getByText('Local browser setting saved')).toBeInTheDocument()
    expect(localStorage.getItem('shop.reviewer')).toBe('alex')
  })

  it('stores the API token in this browser and clears the input', async () => {
    setup()
    renderAt(<SettingsPage />, { route: '/settings', path: '/settings' })
    await screen.findByText('Token required by server')
    await userEvent.type(screen.getByLabelText(/Bearer token/), 'secret-local-token')
    await userEvent.click(screen.getByRole('button', { name: 'Store token' }))
    expect(localStorage.getItem('shop.apiToken')).toBe('secret-local-token')
    expect(screen.getByText(/Local API token stored in this browser/)).toBeInTheDocument()
    expect(screen.queryByDisplayValue('secret-local-token')).not.toBeInTheDocument()

    await userEvent.click(screen.getByRole('button', { name: 'Clear stored token' }))
    expect(localStorage.getItem('shop.apiToken')).toBeNull()
    expect(screen.getByText(/Stored API token cleared/)).toBeInTheDocument()
  })

  it('requires explicit confirmation before applying retention', async () => {
    const { calls } = setup()
    renderAt(<SettingsPage />, { route: '/settings', path: '/settings' })
    await screen.findByText('Token required by server')
    await userEvent.click(screen.getByRole('button', { name: /Run retention maintenance/ }))
    expect(screen.getByRole('dialog', { name: 'Apply evidence retention now?' })).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'Apply retention' }))
    expect(await screen.findByText('Retention maintenance applied')).toBeInTheDocument()
    expect(screen.getByText(/2 expired clips removed/)).toBeInTheDocument()
    expect(screen.getByText(/1 event tombstone updated/)).toBeInTheDocument()
    const post = calls.find((call) => call.url.includes('/maintenance/retention'))
    expect(post?.init.method).toBe('POST')
  })

  it('renders the append-only audit trail without raw absolute paths', async () => {
    setup()
    renderAt(<SettingsPage />, { route: '/settings', path: '/settings' })
    expect(await screen.findByText('review recorded')).toBeInTheDocument()
    expect(screen.getByText(/alex · event evt-1111/)).toBeInTheDocument()
    expect(screen.getByText('decision: benign')).toBeInTheDocument()
    expect(screen.getByText('request req-audit-1')).toBeInTheDocument()
    expect(screen.queryByText(/No audit records have been written yet/)).not.toBeInTheDocument()
  })

  it('shows a config failure and offers retry', async () => {
    let attempts = 0
    mockFetch((url) => {
      if (url.includes('/config')) {
        attempts += 1
        if (attempts === 1) {
          return { status: 503, body: apiErrorBody('CONFIG_UNAVAILABLE', 'Config unavailable.') }
        }
        return { body: makeConfig() }
      }
      if (url.includes('/capabilities')) return { body: makeCapabilities() }
      return { body: makeAudit() }
    })
    renderAt(<SettingsPage />, { route: '/settings', path: '/settings' })
    expect(await screen.findByText('Retention configuration could not load')).toBeInTheDocument()
    await userEvent.click(screen.getAllByRole('button', { name: 'Retry' })[0])
    await waitFor(() => expect(screen.getByText('/managed/local/storage')).toBeInTheDocument())
  })
})
