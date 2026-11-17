import { useEffect, useRef, useState } from 'react'
import { apiBlob, apiGet, ApiError } from './api'

export interface ResourceState<T> {
  data: T | null
  error: ApiError | Error | null
  loading: boolean
  refreshing: boolean
  reload: () => void
}

export function useApiResource<T>(
  path: string | null,
  options: { pollMs?: number; enabled?: boolean } = {},
): ResourceState<T> {
  const { pollMs = 0, enabled = true } = options
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<ApiError | Error | null>(null)
  const [loading, setLoading] = useState(Boolean(path) && enabled)
  const [refreshing, setRefreshing] = useState(false)
  const [version, setVersion] = useState(0)
  const firstLoad = useRef(true)

  useEffect(() => {
    if (!path || !enabled) {
      setLoading(false)
      return
    }
    let disposed = false
    let timer: number | undefined
    const controller = new AbortController()

    const load = async () => {
      if (document.visibilityState === 'hidden' && !firstLoad.current) return
      if (!firstLoad.current) setRefreshing(true)
      try {
        const result = await apiGet<T>(path, controller.signal)
        if (disposed) return
        setData(result)
        setError(null)
        setLoading(false)
        setRefreshing(false)
        firstLoad.current = false
      } catch (caught) {
        if (disposed || (caught instanceof DOMException && caught.name === 'AbortError')) return
        setError(caught instanceof Error ? caught : new Error('The request failed.'))
        setLoading(false)
        setRefreshing(false)
      }
    }

    void load()
    if (pollMs > 0) {
      timer = window.setInterval(() => void load(), pollMs)
    }
    return () => {
      disposed = true
      controller.abort()
      if (timer) window.clearInterval(timer)
    }
  }, [path, pollMs, enabled, version])

  return {
    data,
    error,
    loading,
    refreshing,
    reload: () => {
      firstLoad.current = true
      setVersion((value) => value + 1)
    },
  }
}

export function useBlobUrl(
  path: string | null,
  options: { enabled?: boolean; version?: number } = {},
): { url: string | null; loading: boolean; error: Error | null } {
  const { enabled = true, version = 0 } = options
  const [url, setUrl] = useState<string | null>(null)
  const [loading, setLoading] = useState(Boolean(path) && enabled)
  const [error, setError] = useState<Error | null>(null)

  useEffect(() => {
    if (!path || !enabled) {
      setUrl(null)
      setLoading(false)
      return
    }
    let disposed = false
    let objectUrl: string | null = null
    const controller = new AbortController()
    setLoading(true)
    setError(null)
    apiBlob(path, controller.signal)
      .then((blob) => {
        if (disposed) return
        objectUrl = URL.createObjectURL(blob)
        setUrl(objectUrl)
        setLoading(false)
      })
      .catch((caught) => {
        if (disposed || (caught instanceof DOMException && caught.name === 'AbortError')) return
        setError(caught instanceof Error ? caught : new Error('Media could not be loaded.'))
        setUrl(null)
        setLoading(false)
      })
    return () => {
      disposed = true
      controller.abort()
      if (objectUrl) URL.revokeObjectURL(objectUrl)
    }
  }, [path, enabled, version])

  return { url, loading, error }
}

export function useDebouncedValue<T>(value: T, delayMs = 250): T {
  const [debounced, setDebounced] = useState(value)
  useEffect(() => {
    const timer = window.setTimeout(() => setDebounced(value), delayMs)
    return () => window.clearTimeout(timer)
  }, [value, delayMs])
  return debounced
}
