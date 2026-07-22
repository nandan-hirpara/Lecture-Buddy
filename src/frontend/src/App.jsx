import { useEffect, useId, useRef, useState } from 'react'
import { TASKS, mockReply, delay } from './mock/responses.js'
import './App.css'

function formatTime(date) {
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

export default function App() {
  const fileInputId = useId()
  const chatEndRef = useRef(null)
  const videoRef = useRef(null)

  const [videoFile, setVideoFile] = useState(null)
  const [videoUrl, setVideoUrl] = useState(null)
  const [messages, setMessages] = useState([
    {
      id: 'welcome',
      role: 'assistant',
      text: 'Upload a lecture video, then use a study action or ask a question. Responses are mocked until VideoChat3 is connected.',
      at: new Date(),
    },
  ])
  const [draft, setDraft] = useState('')
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    return () => {
      if (videoUrl) URL.revokeObjectURL(videoUrl)
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
        text: `Loaded “${file.name}”. Study actions are ready (mock).`,
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

  async function pushExchange(userText, taskId = 'chat') {
    if (busy) return
    if (!videoFile && taskId !== 'chat') {
      setMessages((prev) => [
        ...prev,
        {
          id: `need-${Date.now()}`,
          role: 'assistant',
          text: 'Add a lecture video first, then run a study action.',
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
    await delay()
    const reply = mockReply(taskId, videoFile?.name, userText)
    setMessages((prev) => [
      ...prev,
      {
        id: `a-${Date.now()}`,
        role: 'assistant',
        text: reply,
        at: new Date(),
      },
    ])
    setBusy(false)
  }

  function onTask(task) {
    void pushExchange(task.prompt, task.id)
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
        <p className="status-pill">
          {videoFile ? videoFile.name : 'No video yet'}
          <span className="dot" data-ready={Boolean(videoFile)} />
        </p>
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
              <span className="dropzone-hint">MP4, WebM, or MOV · mock mode</span>
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
                Thinking…
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
