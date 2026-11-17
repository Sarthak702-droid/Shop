import { render } from '@testing-library/react'
import type { ReactElement } from 'react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { vi } from 'vitest'

export interface MockResult {
  status?: number
  body?: unknown
  headers?: Record<string, string>
  networkError?: boolean
}

export type MockHandler = (url: string, init: RequestInit) => MockResult | Promise<MockResult>

function requestUrl(input: RequestInfo | URL): string {
  if (typeof input === 'string') return input
  if (input instanceof URL) return input.toString()
  return input.url
}

/**
 * Installs a fetch stub that serves JSON by path. It records calls so tests can
 * assert request shape, headers, and ordering.
 */
export function mockFetch(handler: MockHandler) {
  const calls: Array<{ url: string; init: RequestInit }> = []
  const stub = vi.fn(async (input: RequestInfo | URL, init: RequestInit = {}) => {
    const url = requestUrl(input)
    calls.push({ url, init })
    const result = await handler(url, init)
    if (result.networkError) throw new TypeError('Failed to fetch')
    const body =
      result.body === undefined || result.status === 204
        ? null
        : typeof result.body === 'string'
          ? result.body
          : JSON.stringify(result.body)
    return new Response(body, {
      status: result.status ?? 200,
      headers: { 'Content-Type': 'application/json', ...(result.headers ?? {}) },
    })
  })
  vi.stubGlobal('fetch', stub)
  return { stub, calls }
}

export function callsTo(calls: Array<{ url: string }>, fragment: string) {
  return calls.filter((call) => call.url.includes(fragment))
}

export function renderAt(
  element: ReactElement,
  { route = '/', path = '*' }: { route?: string; path?: string } = {},
) {
  return render(
    <MemoryRouter initialEntries={[route]}>
      <Routes>
        <Route path={path} element={element} />
      </Routes>
    </MemoryRouter>,
  )
}

export function jsonResponse(body: unknown, status = 200, headers: Record<string, string> = {}) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json', ...headers },
  })
}

export function apiErrorBody(code: string, message: string, retryable = false) {
  return { error: { code, message, retryable, details: {} } }
}
