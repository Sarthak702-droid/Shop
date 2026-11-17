import { beforeEach, describe, expect, it, vi } from 'vitest'
import {
  ApiError,
  apiBlob,
  apiGet,
  apiPost,
  apiRequest,
  authHeaders,
  errorMessage,
  getReviewer,
  getToken,
  newIdempotencyKey,
  setReviewer,
  setToken,
} from './api'
import { apiErrorBody, mockFetch } from '../test/utils'

describe('local API client', () => {
  beforeEach(() => {
    localStorage.clear()
  })

  it('stores token and reviewer identities in this browser only', () => {
    expect(getToken()).toBe('')
    expect(getReviewer()).toBe('')
    setToken('  local-secret  ')
    setReviewer('  alex  ')
    expect(getToken()).toBe('local-secret')
    expect(getReviewer()).toBe('alex')
    expect(authHeaders()).toEqual({
      Authorization: 'Bearer local-secret',
      'X-Shop-Reviewer': 'alex',
    })
    setToken('   ')
    setReviewer('')
    expect(localStorage.getItem('shop.apiToken')).toBeNull()
    expect(localStorage.getItem('shop.reviewer')).toBeNull()
  })

  it('sends JSON, bearer auth and reviewer headers on requests', async () => {
    localStorage.setItem('shop.apiToken', 'token-1')
    localStorage.setItem('shop.reviewer', 'sam')
    const { calls } = mockFetch(() => ({ body: { ok: true } }))
    await apiPost('/events/x/reviews', { decision: 'benign' })
    expect(calls).toHaveLength(1)
    expect(calls[0].url).toBe('/api/v1/events/x/reviews')
    const headers = new Headers(calls[0].init.headers)
    expect(headers.get('Authorization')).toBe('Bearer token-1')
    expect(headers.get('X-Shop-Reviewer')).toBe('sam')
    expect(headers.get('Content-Type')).toBe('application/json')
    expect(headers.get('Accept')).toBe('application/json')
    expect(calls[0].init.body).toBe(JSON.stringify({ decision: 'benign' }))
    expect(calls[0].init.credentials).toBe('same-origin')
  })

  it('surfaces the safe error envelope and request reference', async () => {
    const { calls } = mockFetch(() => ({
      status: 503,
      body: apiErrorBody('WORKER_UNAVAILABLE', 'The local worker is not running.', true),
      headers: { 'X-Request-ID': 'req-123' },
    }))
    const failure = apiGet('/health').catch((error: unknown) => error)
    const error = (await failure) as ApiError
    expect(error).toBeInstanceOf(ApiError)
    expect(error.status).toBe(503)
    expect(error.code).toBe('WORKER_UNAVAILABLE')
    expect(error.retryable).toBe(true)
    expect(error.requestId).toBe('req-123')
    expect(errorMessage(error)).toBe('The local worker is not running. Reference: req-123')
    expect(calls).toHaveLength(1)
  })

  it('falls back safely when the response is not a JSON error envelope', async () => {
    mockFetch(() => ({ status: 502, body: 'upstream failed' }))
    const error = (await apiGet('/health').catch((caught: unknown) => caught)) as ApiError
    expect(error).toBeInstanceOf(ApiError)
    expect(error.code).toBe('REQUEST_FAILED')
    expect(error.message).toContain('HTTP 502')
    expect(error.retryable).toBe(false)
  })

  it('dispatches an auth-required event on 401', async () => {
    const listener = vi.fn()
    window.addEventListener('shop-auth-required', listener)
    mockFetch(() => ({
      status: 401,
      body: apiErrorBody('AUTH_REQUIRED', 'A local API token is required.'),
    }))
    await expect(apiGet('/events')).rejects.toMatchObject({ code: 'AUTH_REQUIRED' })
    expect(listener).toHaveBeenCalledTimes(1)
    window.removeEventListener('shop-auth-required', listener)
  })

  it('returns undefined for 204 responses', async () => {
    mockFetch(() => ({ status: 204 }))
    await expect(apiRequest<void>('/events/x', { method: 'DELETE' })).resolves.toBeUndefined()
  })

  it('fetches evidence media with media accept headers and errors as ApiError', async () => {
    const { calls } = mockFetch(() => ({ body: 'media-bytes' }))
    const blob = await apiBlob('/events/x/evidence/thumbnail')
    expect(blob).toBeInstanceOf(Blob)
    expect(blob.size).toBe(11)
    const headers = new Headers(calls[0].init.headers)
    expect(headers.get('Accept')).toContain('video/webm')
    expect(headers.get('Accept')).toContain('image/jpeg')

    const absolute = mockFetch(() => ({ body: 'thumbnail-bytes' }))
    await apiBlob('/api/v1/events/x/thumbnail')
    expect(absolute.calls.at(-1)?.url).toBe('/api/v1/events/x/thumbnail')

    mockFetch(() => ({
      status: 410,
      body: apiErrorBody('EVIDENCE_EXPIRED', 'Evidence expired by retention policy.'),
    }))
    await expect(apiBlob('/events/x/evidence/media')).rejects.toBeInstanceOf(ApiError)
  })

  it('creates unique idempotency keys', () => {
    const first = newIdempotencyKey()
    const second = newIdempotencyKey()
    expect(first).toMatch(/^[0-9a-f-]{36}$/)
    expect(first).not.toBe(second)
  })
})
