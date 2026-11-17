import { fireEvent, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import { UploadPage } from './UploadPage'
import {
  makeCapabilities,
  makeConfig,
  makeUpload,
} from '../test/fixtures'
import { apiErrorBody, mockFetch, renderAt } from '../test/utils'

function setup() {
  return mockFetch((url) => {
    if (url.includes('/capabilities')) return { body: makeCapabilities() }
    if (url.includes('/config')) return { body: makeConfig() }
    if (url.includes('/uploads')) {
      return { body: { items: [makeUpload()], next_cursor: null } }
    }
    return { status: 404, body: apiErrorBody('NOT_FOUND', 'Not found.') }
  })
}

describe('UploadPage', () => {
  it('explains local validation, audio suppression and proposed limits', async () => {
    setup()
    renderAt(<UploadPage />, { route: '/upload', path: '/upload' })
    expect(await screen.findByText('Drop a recording here, or browse')).toBeInTheDocument()
    expect(screen.getByText(/Files never leave this host/i)).toBeInTheDocument()
    expect(screen.getByText('Audio is ignored and never analysed.')).toBeInTheDocument()
    expect(screen.getByText(/Byte limit: 2.0 GB/)).toBeInTheDocument()
    expect(screen.getByText(/Waiting for a validated upload/)).toBeInTheDocument()
  })

  it('rejects an unsupported container before contacting the server', async () => {
    const { calls } = setup()
    const { container } = renderAt(<UploadPage />, { route: '/upload', path: '/upload' })
    await screen.findByText('Drop a recording here, or browse')
    const input = container.querySelector('input[type="file"]') as HTMLInputElement
    const file = new File(['notes'], 'notes.txt', { type: 'text/plain' })
    fireEvent.change(input, { target: { files: [file] } })
    expect(await screen.findByText('Upload was not accepted')).toBeInTheDocument()
    expect(screen.getByText(/Choose a supported video container/i)).toBeInTheDocument()
    expect(calls.some((call) => call.init.method === 'POST')).toBe(false)
    expect(screen.getByRole('button', { name: /Upload and validate/ })).toBeDisabled()
  })

  it('accepts a supported file and enables local upload', async () => {
    setup()
    const { container } = renderAt(<UploadPage />, { route: '/upload', path: '/upload' })
    await screen.findByText('Drop a recording here, or browse')
    const input = container.querySelector('input[type="file"]') as HTMLInputElement
    const file = new File(['video'], 'entrance.mp4', { type: 'video/mp4' })
    fireEvent.change(input, { target: { files: [file] } })
    expect(await screen.findByText('entrance.mp4')).toBeInTheDocument()
    expect(screen.getByText(/5 B selected/)).toBeInTheDocument()
    const upload = screen.getByRole('button', { name: /Upload and validate/ })
    expect(upload).toBeEnabled()
  })

  it('reuses a previously validated upload without re-uploading it', async () => {
    const { calls } = setup()
    renderAt(<UploadPage />, { route: '/upload', path: '/upload' })
    const summary = await screen.findByText('Recently validated uploads (1)')
    await userEvent.click(summary)
    await userEvent.click(screen.getByRole('button', { name: 'Use' }))
    expect(await screen.findByText('File validated locally')).toBeInTheDocument()
    expect(screen.getByText('front-door.mp4', { selector: '.probe-card p' })).toBeInTheDocument()
    await waitFor(() => expect(screen.getByRole('button', { name: 'Queue CPU analysis' })).toBeEnabled())
    const analysis = screen.getByText(/reports confirmed person tracks as descriptive activity segments/i)
    expect(analysis).toBeInTheDocument()
    expect(calls.some((call) => call.init.method === 'POST')).toBe(false)
  })

  it('shows the upload limits failure without blocking the review queue', async () => {
    mockFetch((url) => {
      if (url.includes('/capabilities')) {
        return { status: 503, body: apiErrorBody('CAPABILITIES_UNAVAILABLE', 'Capabilities unavailable.') }
      }
      if (url.includes('/config')) return { body: makeConfig() }
      return { body: { items: [], next_cursor: null } }
    })
    renderAt(<UploadPage />, { route: '/upload', path: '/upload' })
    expect(await screen.findByText('Upload limits could not load')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Review queue' })).toBeInTheDocument()
  })
})
