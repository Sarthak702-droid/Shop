import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import { Route, Routes, MemoryRouter } from 'react-router-dom'
import { render } from '@testing-library/react'
import { EventDetailPanel } from './EventDetailPanel'
import {
  EVENT_ID,
  makeEvent,
  makeReview,
} from '../test/fixtures'
import { apiErrorBody, mockFetch, renderAt } from '../test/utils'

function detailEvent(overrides = {}) {
  return makeEvent(overrides)
}

describe('EventDetailPanel', () => {
  it('shows a loading state while evidence metadata is requested', () => {
    mockFetch(() => ({ body: { event: detailEvent() } }))
    renderAt(
      <EventDetailPanel eventId={EVENT_ID} />,
      { route: `/review/${EVENT_ID}`, path: '/review/:eventId' },
    )
    expect(screen.getByLabelText('Loading event evidence…')).toBeInTheDocument()
  })

  it('renders the event without inventing a score or a conclusion', async () => {
    mockFetch(() => ({ body: { event: detailEvent() } }))
    const { container } = renderAt(
      <EventDetailPanel eventId={EVENT_ID} />,
      { route: `/review/${EVENT_ID}`, path: '/review/:eventId' },
    )
    expect(await screen.findByText('Person activity observed')).toBeInTheDocument()
    expect(container.querySelector('.score-panel__value')).toHaveTextContent('Not scored')
    expect(container.querySelector('.score-explanation')).toHaveTextContent('null')
    expect(container.querySelector('.score-explanation')).toHaveTextContent('not zero')
    expect(screen.getByText(/does not establish intent, ownership, payment/i)).toBeInTheDocument()
    expect(screen.queryByText(/shoplift|guilty|theft confirmed/i)).not.toBeInTheDocument()
  })

  it('keeps an unavailable evidence clip reviewable and states why it is missing', async () => {
    mockFetch(() => ({ body: { event: detailEvent() } }))
    renderAt(
      <EventDetailPanel eventId={EVENT_ID} />,
      { route: `/review/${EVENT_ID}`, path: '/review/:eventId' },
    )
    expect(await screen.findByText('Evidence unavailable')).toBeInTheDocument()
    expect(screen.getByText('Evidence capture was disabled for this analysis.')).toBeInTheDocument()
    expect(screen.getByText(/event remains reviewable/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Save · Benign/ })).toBeInTheDocument()
  })

  it('shows the retained context window around the observed interval', async () => {
    mockFetch(() => ({
      body: {
        event: detailEvent({
          evidence: {
            status: 'available',
            clip_id: 'clip-1',
            reason: null,
            thumbnail_url: `/api/v1/events/${EVENT_ID}/thumbnail`,
            media_url: `/api/v1/events/${EVENT_ID}/media`,
            asset: {
              clip_id: 'clip-1',
              bytes: 1024,
              sha256: 'c'.repeat(64),
              mime_type: 'video/webm',
              width: 768,
              height: 576,
              fps: 10,
              duration_ms: 6200,
              frame_count: 62,
              encoder: 'opencv_vp8',
              media_interval_ms: { start: 2400, end: 8600 },
              retain_until: null,
            },
          },
        }),
      },
    }))
    renderAt(
      <EventDetailPanel eventId={EVENT_ID} />,
      { route: `/review/${EVENT_ID}`, path: '/review/:eventId' },
    )
    expect(await screen.findByText('Before 1.6s')).toBeInTheDocument()
    expect(screen.getByText('Observed interval 2.6s')).toBeInTheDocument()
    expect(screen.getByText('After 2.0s')).toBeInTheDocument()
    expect(screen.getByText('clip 0:02–0:08')).toBeInTheDocument()
    expect(screen.getByText(/1m 2s|6.2s/)).toBeInTheDocument()
  })

  it('shows append-only review history and correction language', async () => {
    mockFetch(() => ({
      body: {
        event: detailEvent({
          review_state: 'benign',
          reviews: [makeReview()],
        }),
      },
    }))
    renderAt(
      <EventDetailPanel eventId={EVENT_ID} />,
      { route: `/review/${EVENT_ID}`, path: '/review/:eventId' },
    )
    expect(await screen.findByText('Record a correction')).toBeInTheDocument()
    expect(screen.getByText('Customer returned the item to the shelf.')).toBeInTheDocument()
    expect(screen.getByText('Reviewed · benign')).toBeInTheDocument()
    expect(screen.getAllByText('Activity appears benign in context').length).toBeGreaterThan(0)
    expect(screen.getByRole('button', { name: 'Append correction' })).toBeInTheDocument()
    expect(screen.getByText(/earlier decisions remain visible/i)).toBeInTheDocument()
  })

  it('posts a review with reviewer identity and an idempotency key', async () => {
    localStorage.setItem('shop.reviewer', 'alex')
    const { calls } = mockFetch((url, init) => {
      if (init.method === 'POST' && url.includes('/reviews')) {
        return { body: { review: makeReview(), event: detailEvent({ review_state: 'benign' }) } }
      }
      return { body: { event: detailEvent() } }
    })
    renderAt(
      <EventDetailPanel eventId={EVENT_ID} />,
      { route: `/review/${EVENT_ID}`, path: '/review/:eventId' },
    )
    await screen.findByText('Person activity observed')
    expect(screen.getByText(/Signed as/)).toHaveTextContent('alex')
    await userEvent.click(screen.getByRole('button', { name: /Save · Benign/ }))
    expect(await screen.findByText('Review recorded')).toBeInTheDocument()
    const post = calls.find((call) => call.init.method === 'POST')
    expect(post).toBeDefined()
    expect(post?.url).toBe(`/api/v1/events/${EVENT_ID}/reviews`)
    const headers = new Headers(post?.init.headers)
    expect(headers.get('X-Shop-Reviewer')).toBe('alex')
    expect(headers.get('Idempotency-Key')).toMatch(/^[0-9a-f-]{36}$/)
    expect(JSON.parse(String(post?.init.body))).toMatchObject({
      decision: 'benign',
      reason_codes: ['activity_benign'],
      notes: '',
      supersedes_review_id: null,
    })
  })

  it('refuses to submit with no reason codes and no note', async () => {
    mockFetch(() => ({ body: { event: detailEvent() } }))
    renderAt(
      <EventDetailPanel eventId={EVENT_ID} />,
      { route: `/review/${EVENT_ID}`, path: '/review/:eventId' },
    )
    await screen.findByText('Person activity observed')
    await userEvent.click(screen.getByRole('checkbox', { name: /Activity appears benign in context/i }))
    await userEvent.click(screen.getByRole('button', { name: /Save · Benign/ }))
    expect(screen.getByText('Choose at least one reason or add a short note.')).toBeInTheDocument()
  })

  it('requires confirmation before deleting local evidence', async () => {
    const { calls } = mockFetch((url, init) => {
      if (init.method === 'DELETE') return { status: 204 }
      return { body: { event: detailEvent() } }
    })
    render(
      <MemoryRouter initialEntries={[`/review/${EVENT_ID}`]}>
        <Routes>
          <Route path="/review/:eventId" element={<EventDetailPanel eventId={EVENT_ID} />} />
          <Route path="/review" element={<p>Queue destination</p>} />
        </Routes>
      </MemoryRouter>,
    )
    await screen.findByText('Person activity observed')
    await userEvent.click(screen.getByRole('button', { name: /Delete/ }))
    expect(screen.getByRole('dialog', { name: 'Delete this event and its evidence?' })).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'Cancel' }))
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
    expect(calls.some((call) => call.init.method === 'DELETE')).toBe(false)

    await userEvent.click(screen.getByRole('button', { name: /Delete/ }))
    await userEvent.click(screen.getByRole('button', { name: 'Delete event' }))
    expect(await screen.findByText('Queue destination')).toBeInTheDocument()
    expect(calls.some((call) => call.init.method === 'DELETE')).toBe(true)
  })

  it('shows a safe error with a request reference when the event cannot load', async () => {
    mockFetch(() => ({
      status: 410,
      body: apiErrorBody('EVENT_DELETED', 'This event has been deleted.'),
      headers: { 'X-Request-ID': 'req-event-1' },
    }))
    renderAt(
      <EventDetailPanel eventId={EVENT_ID} />,
      { route: `/review/${EVENT_ID}`, path: '/review/:eventId' },
    )
    expect(await screen.findByText('Event could not load')).toBeInTheDocument()
    expect(screen.getByText(/This event has been deleted.*req-event-1/)).toBeInTheDocument()
  })
})
