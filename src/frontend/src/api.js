/** Client for the LectureBuddy inference API (proxied via Vite to :8000). */

const API_BASE = import.meta.env.VITE_API_BASE || ''

export async function fetchHealth({ signal } = {}) {
  const res = await fetch(`${API_BASE}/health`, { signal })
  if (!res.ok) {
    throw new Error(`Health check failed (${res.status})`)
  }
  return res.json()
}

/**
 * Upload a lecture video + question to VideoChat3.
 * @param {File} videoFile
 * @param {string} question
 * @param {{ signal?: AbortSignal, maxNewTokens?: number, task?: string }} [opts]
 */
export async function askVideo(videoFile, question, opts = {}) {
  const form = new FormData()
  form.append('question', question)
  form.append('video', videoFile, videoFile.name)
  if (opts.task) {
    form.append('task', opts.task)
  }
  if (opts.maxNewTokens != null) {
    form.append('max_new_tokens', String(opts.maxNewTokens))
  }

  const res = await fetch(`${API_BASE}/v1/ask`, {
    method: 'POST',
    body: form,
    signal: opts.signal,
  })

  let payload = null
  try {
    payload = await res.json()
  } catch {
    payload = null
  }

  if (!res.ok) {
    const detail = payload?.detail
    const message =
      typeof detail === 'string'
        ? detail
        : Array.isArray(detail)
          ? detail.map((d) => d.msg || JSON.stringify(d)).join('; ')
          : `Inference request failed (${res.status})`
    throw new Error(message)
  }

  return payload
}

/**
 * Upload a lecture video + topic query for temporal grounding.
 * @param {File} videoFile
 * @param {string} query
 * @param {{ signal?: AbortSignal, maxNewTokens?: number }} [opts]
 */
export async function groundVideo(videoFile, query, opts = {}) {
  const form = new FormData()
  form.append('query', query)
  form.append('video', videoFile, videoFile.name)
  if (opts.maxNewTokens != null) {
    form.append('max_new_tokens', String(opts.maxNewTokens))
  }

  const res = await fetch(`${API_BASE}/v1/ground`, {
    method: 'POST',
    body: form,
    signal: opts.signal,
  })

  let payload = null
  try {
    payload = await res.json()
  } catch {
    payload = null
  }

  if (!res.ok) {
    const detail = payload?.detail
    const message =
      typeof detail === 'string'
        ? detail
        : Array.isArray(detail)
          ? detail.map((d) => d.msg || JSON.stringify(d)).join('; ')
          : `Grounding request failed (${res.status})`
    throw new Error(message)
  }

  return payload
}
