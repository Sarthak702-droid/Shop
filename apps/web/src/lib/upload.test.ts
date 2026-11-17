import { beforeEach, describe, expect, it, vi } from 'vitest'
import { ApiError, uploadVideo } from './api'
import { makeUpload } from '../test/fixtures'

type ProgressHandler = (event: { lengthComputable: boolean; loaded: number; total: number }) => void

class FakeXMLHttpRequest {
  static last: FakeXMLHttpRequest | null = null
  upload: { onprogress: ProgressHandler | null } = { onprogress: null }
  status = 0
  response: unknown = null
  onerror: (() => void) | null = null
  onabort: (() => void) | null = null
  onload: (() => void) | null = null
  method = ''
  url = ''
  headers: Record<string, string> = {}
  body: unknown = null
  responseType = ''
  private abortListeners: Array<() => void> = []

  constructor() {
    FakeXMLHttpRequest.last = this
  }

  open(method: string, url: string) {
    this.method = method
    this.url = url
  }

  setRequestHeader(key: string, value: string) {
    this.headers[key] = value
  }

  getResponseHeader(name: string) {
    return name === 'X-Request-ID' ? 'req-upload-1' : null
  }

  addEventListener(type: string, listener: () => void) {
    if (type === 'abort') this.abortListeners.push(listener)
  }

  send(body: unknown) {
    this.body = body
  }

  abort() {
    this.onabort?.()
  }

  progress(loaded: number, total: number) {
    this.upload.onprogress?.({ lengthComputable: true, loaded, total })
  }

  respond(status: number, body: unknown) {
    this.status = status
    this.response = body
    this.onload?.()
  }
}

function xhr() {
  const instance = FakeXMLHttpRequest.last
  if (!instance) throw new Error('upload did not open an XMLHttpRequest')
  return instance
}

describe('uploadVideo', () => {
  beforeEach(() => {
    FakeXMLHttpRequest.last = null
    localStorage.setItem('shop.reviewer', 'alex')
    vi.stubGlobal('XMLHttpRequest', FakeXMLHttpRequest)
  })

  it('reports upload progress and resolves the validated upload', async () => {
    const controller = new AbortController()
    const progress = vi.fn()
    const pending = uploadVideo(
      new File(['video'], 'entrance.mp4', { type: 'video/mp4' }),
      progress,
      controller.signal,
    )
    const request = xhr()
    expect(request.method).toBe('POST')
    expect(request.url).toBe('/api/v1/uploads')
    expect(request.headers['X-Shop-Reviewer']).toBe('alex')
    request.progress(25, 100)
    expect(progress).toHaveBeenCalledWith(0.25)
    request.respond(201, { upload: makeUpload(), request_id: 'req-upload-1' })
    await expect(pending).resolves.toMatchObject({
      upload: { upload_id: 'upl-44444444-4444-4444-8444-444444444444' },
      request_id: 'req-upload-1',
    })
  })

  it('maps a server upload error to the safe API envelope', async () => {
    const controller = new AbortController()
    const pending = uploadVideo(new File(['x'], 'huge.mp4'), () => {}, controller.signal)
    xhr().respond(413, {
      error: {
        code: 'UPLOAD_TOO_LARGE',
        message: 'The file exceeds the configured local size limit.',
        retryable: false,
        details: {},
      },
    })
    const error = (await pending.catch((caught: unknown) => caught)) as ApiError
    expect(error).toBeInstanceOf(ApiError)
    expect(error.status).toBe(413)
    expect(error.code).toBe('UPLOAD_TOO_LARGE')
    expect(error.requestId).toBe('req-upload-1')
  })

  it('rejects with an AbortError when the reviewer cancels', async () => {
    const controller = new AbortController()
    const pending = uploadVideo(new File(['x'], 'entrance.mp4'), () => {}, controller.signal)
    controller.abort()
    await expect(pending).rejects.toMatchObject({ name: 'AbortError' })
  })

  it('reports a network failure as retryable', async () => {
    const controller = new AbortController()
    const pending = uploadVideo(new File(['x'], 'entrance.mp4'), () => {}, controller.signal)
    xhr().onerror?.()
    await expect(pending).rejects.toMatchObject({ code: 'NETWORK_ERROR', retryable: true })
  })
})
