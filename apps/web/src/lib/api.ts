const API_BASE = import.meta.env.VITE_API_BASE ?? '/api/v1'
const TOKEN_KEY = 'shop.apiToken'
const REVIEWER_KEY = 'shop.reviewer'

export interface ApiErrorPayload {
  code: string
  message: string
  retryable: boolean
  details?: Record<string, unknown>
}

export class ApiError extends Error {
  readonly status: number
  readonly code: string
  readonly retryable: boolean
  readonly details: Record<string, unknown>
  readonly requestId: string | null

  constructor(
    status: number,
    payload: ApiErrorPayload,
    requestId: string | null,
  ) {
    super(payload.message)
    this.name = 'ApiError'
    this.status = status
    this.code = payload.code
    this.retryable = payload.retryable
    this.details = payload.details ?? {}
    this.requestId = requestId
  }
}

export function getToken(): string {
  return localStorage.getItem(TOKEN_KEY) ?? ''
}

export function setToken(value: string): void {
  if (value.trim()) localStorage.setItem(TOKEN_KEY, value.trim())
  else localStorage.removeItem(TOKEN_KEY)
}

export function getReviewer(): string {
  return localStorage.getItem(REVIEWER_KEY) ?? ''
}

export function setReviewer(value: string): void {
  if (value.trim()) localStorage.setItem(REVIEWER_KEY, value.trim())
  else localStorage.removeItem(REVIEWER_KEY)
}

export function authHeaders(): Record<string, string> {
  const token = getToken()
  const reviewer = getReviewer()
  const headers: Record<string, string> = {}
  if (token) headers.Authorization = `Bearer ${token}`
  if (reviewer) headers['X-Shop-Reviewer'] = reviewer
  return headers
}

function apiPath(path: string): string {
  // The API returns media paths already rooted at /api/v1. Accept both that
  // form and the shorter paths used by client calls without double-prefixing.
  return path.startsWith('/api/') ? path : `${API_BASE}${path}`
}

function notifyAuthRequired(status: number): void {
  if (status === 401) {
    window.dispatchEvent(new CustomEvent('shop-auth-required'))
  }
}

async function parseError(response: Response): Promise<ApiError> {
  let body: Record<string, unknown> | null = null
  try {
    body = (await response.json()) as Record<string, unknown>
  } catch {
    body = null
  }
  const error = (body?.error ?? {}) as Record<string, unknown>
  const payload: ApiErrorPayload = {
    code: typeof error.code === 'string' ? error.code : 'REQUEST_FAILED',
    message:
      typeof error.message === 'string'
        ? error.message
        : `The local API returned HTTP ${response.status}.`,
    retryable: Boolean(error.retryable),
    details: (error.details as Record<string, unknown> | undefined) ?? {},
  }
  const requestId = response.headers.get('X-Request-ID')
  notifyAuthRequired(response.status)
  return new ApiError(response.status, payload, requestId)
}

export async function apiRequestWithHeaders<T>(
  path: string,
  init: RequestInit = {},
): Promise<{ data: T; headers: Headers }> {
  const headers = new Headers(init.headers)
  for (const [key, value] of Object.entries(authHeaders())) headers.set(key, value)
  if (init.body && !(init.body instanceof FormData) && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json')
  }
  headers.set('Accept', 'application/json')
  const response = await fetch(apiPath(path), {
    ...init,
    headers,
    credentials: 'same-origin',
  })
  if (!response.ok) throw await parseError(response)
  if (response.status === 204) return { data: undefined as T, headers: response.headers }
  return { data: (await response.json()) as T, headers: response.headers }
}

export async function apiRequest<T>(
  path: string,
  init: RequestInit = {},
): Promise<T> {
  const { data } = await apiRequestWithHeaders<T>(path, init)
  return data
}

export function apiGet<T>(path: string, signal?: AbortSignal): Promise<T> {
  return apiRequest<T>(path, { signal, method: 'GET' })
}

export function apiGetWithHeaders<T>(path: string): Promise<{ data: T; headers: Headers }> {
  return apiRequestWithHeaders<T>(path, { method: 'GET' })
}

export function apiPost<T>(path: string, body: unknown, headers?: Record<string, string>): Promise<T> {
  return apiRequest<T>(path, {
    method: 'POST',
    body: JSON.stringify(body),
    headers,
  })
}

export function apiPatch<T>(path: string, body: unknown, headers?: Record<string, string>): Promise<T> {
  return apiRequest<T>(path, {
    method: 'PATCH',
    body: JSON.stringify(body),
    headers,
  })
}

export function apiDelete<T>(path: string): Promise<T> {
  return apiRequest<T>(path, { method: 'DELETE' })
}

export async function apiBlob(path: string, signal?: AbortSignal): Promise<Blob> {
  const headers = new Headers(authHeaders())
  headers.set('Accept', 'video/webm,image/jpeg,image/*;q=0.9,*/*;q=0.5')
  const response = await fetch(apiPath(path), {
    headers,
    signal,
    credentials: 'same-origin',
  })
  if (!response.ok) throw await parseError(response)
  return response.blob()
}

export function newIdempotencyKey(): string {
  return crypto.randomUUID()
}

export function uploadVideo(
  file: File,
  onProgress: (fraction: number) => void,
  signal: AbortSignal,
): Promise<{ upload: import('./types').Upload; request_id: string }> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    const form = new FormData()
    form.append('file', file, file.name)
    xhr.open('POST', `${API_BASE}/uploads`)
    for (const [key, value] of Object.entries(authHeaders())) xhr.setRequestHeader(key, value)
    xhr.responseType = 'json'
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) onProgress(event.loaded / event.total)
    }
    xhr.onerror = () => reject(new ApiError(0, {
      code: 'NETWORK_ERROR',
      message: 'The local API could not be reached.',
      retryable: true,
    }, null))
    xhr.onabort = () => reject(new DOMException('Upload cancelled', 'AbortError'))
    xhr.onload = () => {
      const body = xhr.response as Record<string, unknown> | null
      if (xhr.status >= 200 && xhr.status < 300 && body) {
        resolve(body as unknown as { upload: import('./types').Upload; request_id: string })
        return
      }
      const error = (body?.error ?? {}) as Record<string, unknown>
      notifyAuthRequired(xhr.status)
      reject(new ApiError(
        xhr.status,
        {
          code: typeof error.code === 'string' ? error.code : 'UPLOAD_FAILED',
          message: typeof error.message === 'string'
            ? error.message
            : `The upload failed with HTTP ${xhr.status}.`,
          retryable: Boolean(error.retryable),
          details: (error.details as Record<string, unknown> | undefined) ?? {},
        },
        xhr.getResponseHeader('X-Request-ID'),
      ))
    }
    signal.addEventListener('abort', () => xhr.abort(), { once: true })
    xhr.send(form)
  })
}

export function errorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    return error.requestId ? `${error.message} Reference: ${error.requestId}` : error.message
  }
  if (error instanceof Error) return error.message
  return 'An unexpected error occurred.'
}
