import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { ApiError } from '../lib/api'
import {
  EmptyState,
  ErrorPanel,
  Field,
  InlineNotice,
  LoadingPanel,
  MetricCard,
  Modal,
  Spinner,
  StatusPill,
} from './Common'

describe('shared UI states', () => {
  it('exposes spinner and loading panel status to assistive technology', () => {
    const { rerender } = render(<Spinner label="Loading local events" />)
    expect(screen.getByRole('status')).toHaveTextContent('Loading local events')

    rerender(<LoadingPanel label="Loading evidence" rows={2} />)
    expect(screen.getByLabelText('Loading evidence')).toHaveAttribute('aria-busy', 'true')
    expect(screen.getAllByText('Loading evidence').length).toBeGreaterThan(0)
  })

  it('renders a calm error with code, request reference and retry action', async () => {
    const retry = vi.fn()
    const error = new ApiError(
      503,
      { code: 'WORKER_UNAVAILABLE', message: 'The local worker is offline.', retryable: true },
      'req-999',
    )
    render(<ErrorPanel error={error} onRetry={retry} />)
    expect(screen.getByRole('alert')).toHaveTextContent('The local worker is offline.')
    expect(screen.getByText('Code: WORKER_UNAVAILABLE')).toBeInTheDocument()
    expect(screen.getByText('Request: req-999')).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'Retry' }))
    expect(retry).toHaveBeenCalledTimes(1)
  })

  it('hides retry for permanent client errors such as a malformed event ID', () => {
    const error = new ApiError(
      422,
      { code: 'VALIDATION_ERROR', message: 'The request did not match the required schema.', retryable: false },
      'req-422',
    )
    render(<ErrorPanel error={error} onRetry={vi.fn()} />)
    expect(screen.getByRole('alert')).toHaveTextContent('Code: VALIDATION_ERROR')
    expect(screen.queryByRole('button', { name: 'Retry' })).not.toBeInTheDocument()
  })

  it('renders inline notices with the right urgency role', () => {
    const { rerender } = render(<InlineNotice tone="danger" title="Upload rejected">Try again.</InlineNotice>)
    expect(screen.getByRole('alert')).toHaveTextContent('Upload rejected')
    rerender(<InlineNotice tone="info" title="Local only">Nothing left this host.</InlineNotice>)
    expect(screen.getByRole('status')).toHaveTextContent('Nothing left this host.')
  })

  it('keeps empty, metric and status components presentational', () => {
    render(
      <>
        <EmptyState title="Nothing waiting for review">
          <p>Empty queues do not imply no activity occurred.</p>
        </EmptyState>
        <MetricCard label="Needs review" value={0} detail="No recorded review outcome." />
        <StatusPill tone="review" dot>Needs review</StatusPill>
      </>,
    )
    expect(screen.getByText('Nothing waiting for review')).toBeInTheDocument()
    expect(screen.getByText('Empty queues do not imply no activity occurred.')).toBeInTheDocument()
    expect(screen.getByText('Needs review', { selector: '.pill' })).toBeInTheDocument()
  })

  it('wires field hints and errors to their input', () => {
    render(
      <Field label="Reviewer identifier" htmlFor="reviewer" hint="Allowlisted locally." error="Not allowed.">
        <input id="reviewer" />
      </Field>,
    )
    const input = screen.getByLabelText(/Reviewer identifier/)
    expect(input).toHaveAttribute('id', 'reviewer')
    expect(screen.getByText('Allowlisted locally.')).toHaveAttribute('id', 'reviewer-hint')
    expect(screen.getByText('Not allowed.')).toHaveAttribute('id', 'reviewer-error')
  })

  it('closes the modal on Escape and on backdrop press, and labels the dialog', async () => {
    const close = vi.fn()
    const { container } = render(
      <Modal title="Delete this event?" onClose={close} footer={<button type="button">Delete event</button>}>
        <p>The audit tombstone remains.</p>
      </Modal>,
    )
    const dialog = screen.getByRole('dialog', { name: 'Delete this event?' })
    expect(dialog).toHaveAttribute('aria-modal', 'true')
    expect(dialog).toHaveTextContent('The audit tombstone remains.')

    await userEvent.keyboard('{Escape}')
    expect(close).toHaveBeenCalledTimes(1)

    const backdrop = container.querySelector('.modal-backdrop') as HTMLElement
    await userEvent.click(backdrop)
    expect(close).toHaveBeenCalledTimes(2)
  })
})
