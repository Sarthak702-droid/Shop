import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import { ReviewQueuePage } from './ReviewQueuePage'
import { makeEvent, makeEventList, makeMetrics } from '../test/fixtures'
import { apiErrorBody, callsTo, mockFetch, renderAt } from '../test/utils'

function mockQueue(
  events = makeEventList(),
  metrics = makeMetrics(),
) {
  return mockFetch((url) => {
    if (url.includes('/events')) return { body: events }
    if (url.includes('/metrics')) return { body: metrics }
    return { status: 404, body: apiErrorBody('NOT_FOUND', 'Not found.') }
  })
}

describe('ReviewQueuePage', () => {
  it('shows a loading skeleton before the first local response', () => {
    mockQueue()
    renderAt(<ReviewQueuePage />, { route: '/review', path: '/review' })
    expect(document.querySelector('.loading-panel')).toBeInTheDocument()
    expect(screen.getByText('Review queue')).toBeInTheDocument()
  })

  it('shows an honest empty state when nothing is awaiting review', async () => {
    mockQueue(makeEventList([]), makeMetrics({ events_total: 0, events_by_review_state: {} }))
    renderAt(<ReviewQueuePage />, { route: '/review', path: '/review' })
    expect(await screen.findByText('Nothing waiting for review')).toBeInTheDocument()
    expect(screen.getByText(/does not imply that no activity occurred/i)).toBeInTheDocument()
    expect(screen.queryByRole('link')).not.toBeInTheDocument()
  })

  it('renders real queue records with a separate review state and no theft claim', async () => {
    const { calls } = mockQueue()
    renderAt(<ReviewQueuePage />, { route: '/review', path: '/review' })
    expect(await screen.findByText('Front entrance camera')).toBeInTheDocument()
    expect(screen.getByText('Person activity observed')).toBeInTheDocument()
    expect(screen.getByText('Person track observed')).toBeInTheDocument()
    expect(screen.getAllByText('Unreviewed').length).toBeGreaterThan(0)
    expect(screen.queryByText(/shoplift|guilty|caught/i)).not.toBeInTheDocument()
    expect(screen.getByText(/none states that theft occurred/i)).toBeInTheDocument()
    expect(callsTo(calls, '/events?limit=30&reviewed=false')).toHaveLength(1)
    expect(callsTo(calls, '/events?limit=30&reviewed=true')).toHaveLength(0)
  })

  it('changes the server-side review-state filter', async () => {
    const { calls } = mockQueue(
      makeEventList([makeEvent({ review_state: 'benign', reviews: [] })]),
      makeMetrics({ events_by_review_state: { unreviewed: 0, benign: 1 } }),
    )
    renderAt(<ReviewQueuePage />, { route: '/review', path: '/review' })
    await screen.findByText('Front entrance camera')
    await userEvent.click(screen.getByRole('tab', { name: 'Reviewed' }))
    await waitFor(() => {
      expect(callsTo(calls, '/events?limit=30&reviewed=true')).toHaveLength(1)
    })
    expect(screen.getByRole('tab', { name: 'Reviewed' })).toHaveAttribute('aria-selected', 'true')
    expect(screen.queryByText('Nothing waiting for review')).not.toBeInTheDocument()
  })

  it('filters only records already loaded, without pretending to search the database', async () => {
    mockQueue()
    renderAt(<ReviewQueuePage />, { route: '/review', path: '/review' })
    await screen.findByText('Front entrance camera')
    const search = screen.getByRole('searchbox', { name: 'Filter loaded events' })
    await userEvent.type(search, 'forklift')
    expect(await screen.findByText('No matching loaded events')).toBeInTheDocument()
    expect(screen.getByText(/Search does not scan records that have not been fetched/i)).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'Clear filter' }))
    expect(await screen.findByText('Front entrance camera')).toBeInTheDocument()
  })

  it('shows the safe API error and retries the queue request', async () => {
    let attempts = 0
    const { calls } = mockFetch((url) => {
      if (url.includes('/events')) {
        attempts += 1
        if (attempts === 1) {
          return {
            status: 503,
            body: apiErrorBody('DATABASE_UNAVAILABLE', 'The local database is unavailable.', true),
            headers: { 'X-Request-ID': 'req-queue-1' },
          }
        }
        return { body: makeEventList() }
      }
      return { body: makeMetrics() }
    })
    renderAt(<ReviewQueuePage />, { route: '/review', path: '/review' })
    expect(await screen.findByText('Review queue could not load')).toBeInTheDocument()
    expect(screen.getByText(/The local database is unavailable.*req-queue-1/)).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'Retry' }))
    expect(await screen.findByText('Front entrance camera')).toBeInTheDocument()
    expect(callsTo(calls, '/events?limit=30')).toHaveLength(2)
  })
})
