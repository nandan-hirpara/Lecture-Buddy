import { useEffect, useId, useRef, useState } from 'react'
import { askVideo, fetchHealth } from './api.js'
import { TASKS, mockReply } from './mock/responses.js'
import './App.css'

function formatTime(date) {
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

function describeApi(health) {
  if (!health) return { label: 'API offline', ready: false }
  if (health.error) return { label: `API error`, ready: false }
  if (!health.ready) return { label: 'API loading…', ready: false }
  if (health.mock) return { label: 'API mock', ready: true }
  return { label: 'VideoChat3 live', ready: true }
}

export default function App() {
  const fileInputId = useId()
  const chatEndRef = useRef(null)
  const videoRef = useRef(null)
  const abortRef = useRef(null)

  const [videoFile, setVideoFile] = useState(null)
  const [videoUrl, setVideoUrl] = useState(null)
  const [messages, setMessages] = useState([
    {
      id: 'welcome',
      role: 'assistant',
      text: 'Upload a lecture video, then use a study action or ask a question. The UI talks to the inference API at :8000 when it is available.',
      at: new Date(),
    },
  ])
  const [draft, setDraft] = useState('')
  const [busy, setBusy] = useState(false)
  const [apiHealth, setApiHealth] = useState(null)

  const api = describeApi(apiHealth)

  useEffect(() => {
    let cancelled = false
    const controller = new AbortController()

    async function refresh() {
      try {
        const health = await fetchHealth({ signal: controller.signal })
        if (!cancelled) setApiHealth(health)
      } catch {
        if (!cancelled) setApiHealth(null)
      }
    }

    void refresh()
    const id = setInterval(refresh, 8000)
    return () => {
      cancelled = true
      controller.abort()
      clearInterval(id)
    }
  }, [])

  useEffect(() => {
    return () => {
      if (videoUrl) URL.revokeObjectURL(videoUrl)
      abortRef.current?.abort()
    }
  }, [videoUrl])

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages, busy])

  function onPickVideo(file) {
    if (!file) return
    if (videoUrl) URL.revokeObjectURL(videoUrl)
    const url = URL.createObjectURL(file)
    setVideoFile(file)
    setVideoUrl(url)
    setMessages((prev) => [
      ...prev,
      {
        id: `sys-${Date.now()}`,
        role: 'system',
        text: `Loaded "${file.name}". ${
          api.ready
            ? 'Questions will go to the inference API.'
            : 'API offline — answers will use local mock until the server is up.'
        }`,
        at: new Date(),
      },
    ])
  }

  function clearVideo() {
    if (videoUrl) URL.revokeObjectURL(videoUrl)
    setVideoFile(null)
    setVideoUrl(null)
    if (videoRef.current) videoRef.current.removeAttribute('src')
  }

  async function resolveReply(userText, taskId) {
    if (!videoFile) {
      return mockReply(taskId, null, userText)
    }

    if (!api.ready) {
      return (
        mockReply(taskId, videoFile.name, userText) +
        '\n\n_(API offline — mock fallback. Start src/inference uvicorn on :8000.)_'
      )
    }

    const controller = new AbortController()
    abortRef.current = controller
    // Long videos on a 6GB GPU can take several minutes.
    const timeout = setTimeout(() => controller.abort(), 10 * 60 * 1000)
    try {
      const result = await askVideo(videoFile, userText, {
        signal: controller.signal,
        task: taskId && taskId !== 'chat' ? taskId : undefined,
      })
      const badge = result.mock ? ' [api-mock]' : ''
      return `${result.answer}${badge}`
    } finally {
      clearTimeout(timeout)
      if (abortRef.current === controller) abortRef.current = null
    }
  }

  async function pushExchange(userText, taskId = 'chat') {
    if (busy) return
    if (!videoFile) {
      setMessages((prev) => [
        ...prev,
        {
          id: `need-${Date.now()}`,
          role: 'assistant',
          text: 'Add a lecture video first, then ask or run a study action.',
          at: new Date(),
        },
      ])
      return
    }

    const userMsg = {
      id: `u-${Date.now()}`,
      role: 'user',
      text: userText,
      at: new Date(),
    }
    setMessages((prev) => [...prev, userMsg])
    setBusy(true)

    try {
      const reply = await resolveReply(userText, taskId)
      setMessages((prev) => [
        ...prev,
        {
          id: `a-${Date.now()}`,
          role: 'assistant',
          text: reply,
          at: new Date(),
        },
      ])
    } catch (err) {
      const message =
        err?.name === 'AbortError'
          ? 'Request timed out or was cancelled. Try a shorter clip or fewer frames.'
          : err?.message || 'Inference failed.'
      setMessages((prev) => [
        ...prev,
        {
          id: `err-${Date.now()}`,
          role: 'assistant',
          text: `Could not reach VideoChat3: ${message}`,
          at: new Date(),
        },
      ])
    } finally {
      setBusy(false)
    }
  }

  function onTask(task) {
    void pushExchange(task.chatText, task.id)
  }

  function onSubmit(event) {
    event.preventDefault()
    const text = draft.trim()
    if (!text) return
    setDraft('')
    void pushExchange(text, 'chat')
  }

  return (
    <div className="app">
      <div className="atmosphere" aria-hidden="true" />

      <header className="top">
        <div className="brand-block">
          <p className="brand">LectureBuddy</p>
          <p className="tagline">Your companion for lecture videos</p>
        </div>
        <div className="status-row">
          <p className="status-pill" title={apiHealth?.model_id || 'Inference API'}>
            {api.label}
            <span className="dot" data-ready={api.ready} />
          </p>
          <p className="status-pill">
            {videoFile ? videoFile.name : 'No video yet'}
            <span className="dot" data-ready={Boolean(videoFile)} />
          </p>
        </div>
      </header>

      <main className="workspace">
        <section className="stage" aria-label="Lecture video">
          {videoUrl ? (
            <div className="player-wrap">
              <video
                ref={videoRef}
                className="player"
                src={videoUrl}
                controls
                playsInline
              />
              <button type="button" className="ghost-btn" onClick={clearVideo}>
                Remove video
              </button>
            </div>
          ) : (
            <label className="dropzone" htmlFor={fileInputId}>
              <span className="dropzone-title">Drop a lecture here</span>
              <span className="dropzone-hint">MP4, WebM, or MOV · sent to VideoChat3 on ask</span>
              <span className="dropzone-cta">Choose file</span>
            </label>
          )}
          <input
            id={fileInputId}
            className="sr-only"
            type="file"
            accept="video/*"
            onChange={(e) => onPickVideo(e.target.files?.[0])}
          />
        </section>

        <section className="companion" aria-label="Study companion">
          <div className="tasks" role="toolbar" aria-label="Study actions">
            {TASKS.map((task) => (
              <button
                key={task.id}
                type="button"
                className="task-btn"
                disabled={busy}
                onClick={() => onTask(task)}
              >
                {task.label}
              </button>
            ))}
          </div>

          <div className="chat" role="log" aria-live="polite">
            {messages.map((msg) => (
              <article key={msg.id} className={`bubble bubble-${msg.role}`}>
                <header className="bubble-meta">
                  <span>
                    {msg.role === 'user'
                      ? 'You'
                      : msg.role === 'system'
                        ? 'Session'
                        : 'Buddy'}
                  </span>
                  <time dateTime={msg.at.toISOString()}>{formatTime(msg.at)}</time>
                </header>
                <div className="bubble-body">{msg.text}</div>
              </article>
            ))}
            {busy ? (
              <p className="thinking" aria-busy="true">
                {api.ready ? 'Asking VideoChat3…' : 'Thinking…'}
              </p>
            ) : null}
            <div ref={chatEndRef} />
          </div>

          <form className="composer" onSubmit={onSubmit}>
            <label className="sr-only" htmlFor="ask">
              Ask about the lecture
            </label>
            <input
              id="ask"
              type="text"
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              placeholder="Ask anything about the lecture…"
              disabled={busy}
              autoComplete="off"
            />
            <button type="submit" className="send-btn" disabled={busy || !draft.trim()}>
              Ask
            </button>
          </form>
        </section>
      </main>
    </div>
  )
}
