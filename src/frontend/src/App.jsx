import { useEffect, useId, useRef, useState } from 'react'
import {
  askVideo,
  chaptersVideo,
  fetchHealth,
  flashcardsVideo,
  groundVideo,
  proactiveVideo,
  quizVideo,
} from './api.js'
import FlashcardDeck from './FlashcardDeck.jsx'
import QuizPanel from './QuizPanel.jsx'
import { TASKS, mockReply } from './mock/responses.js'
import './App.css'

function formatTime(date) {
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

function formatClock(seconds) {
  const total = Math.max(0, Math.round(Number(seconds) || 0))
  const h = Math.floor(total / 3600)
  const m = Math.floor((total % 3600) / 60)
  const s = total % 60
  if (h > 0) {
    return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
  }
  return `${m}:${String(s).padStart(2, '0')}`
}

/** Parse "3:42", "1:02:30", or plain seconds into a number. */
function parseClock(text) {
  const raw = String(text || '').trim()
  if (!raw) return 0
  if (/^\d+(\.\d+)?$/.test(raw)) return Math.max(0, Number(raw))
  const parts = raw.split(':').map((p) => Number(p))
  if (parts.some((n) => Number.isNaN(n))) return 0
  if (parts.length === 2) return Math.max(0, parts[0] * 60 + parts[1])
  if (parts.length === 3) return Math.max(0, parts[0] * 3600 + parts[1] * 60 + parts[2])
  return 0
}

function describeApi(health) {
  if (!health) return { label: 'API offline', ready: false, mock: false }
  if (health.error) return { label: `API error`, ready: false, mock: false }
  if (!health.ready) return { label: 'API loading…', ready: false, mock: Boolean(health.mock) }
  if (health.mock) return { label: 'API mock', ready: true, mock: true }
  return { label: 'VideoChat3 live', ready: true, mock: false }
}

function mockGrounding(query) {
  return {
    query,
    found: true,
    segments: [
      {
        start_sec: 12,
        end_sec: 45,
        label: 'First introduction (mock)',
        evidence: `Placeholder span for “${query}”.`,
      },
      {
        start_sec: 120,
        end_sec: 150,
        label: 'Follow-up mention (mock)',
        evidence: 'Secondary placeholder span.',
      },
    ],
    display: `**Temporal grounding** for “${query}”\n\nJump points (mock):\n- **0:12–0:45** — First introduction\n- **2:00–2:30** — Follow-up mention`,
    mock: true,
  }
}

function mockProactive(question, startSec = 0) {
  const offset = Math.max(0, Number(startSec) || 0)
  return {
    question,
    start_sec: offset,
    rounds: [
      { round_idx: 0, time_start: offset + 0, time_end: offset + 2, state: 'silence', high_res: false },
      { round_idx: 1, time_start: offset + 2, time_end: offset + 4, state: 'silence', high_res: false },
      { round_idx: 2, time_start: offset + 4, time_end: offset + 6, state: 'standby', high_res: false },
      { round_idx: 3, time_start: offset + 6, time_end: offset + 8, state: 'response', high_res: true },
    ],
    final_answer: `[mock] Enough evidence to answer: ${question}`,
    display: `**Proactive stream** for “${question}”${
      offset > 0 ? ` _(from ${offset}s)_` : ''
    }\n\n- **${offset}s–${offset + 2}s** [SILENCE]: </Silence>\n- **${offset + 2}s–${
      offset + 4
    }s** [SILENCE]: </Silence>\n- **${offset + 4}s–${offset + 6}s** [STANDBY]: </Standby>\n- **${
      offset + 6
    }s–${offset + 8}s** [RESPONSE] HIGH-RES: [mock] Enough evidence to answer: ${question}\n\n**Final answer:** [mock] Enough evidence to answer: ${question}`,
    mock: true,
  }
}

function mockChapters(videoName) {
  const title = `${(videoName || 'lecture').replace(/\.[^.]+$/, '')} (mock outline)`
  const chapters = [
    {
      index: 1,
      start_sec: 0,
      end_sec: 90,
      title: 'Opening and agenda',
      summary: 'Instructor sets motivation and lists what the lecture will cover.',
    },
    {
      index: 2,
      start_sec: 90,
      end_sec: 240,
      title: 'Core definitions',
      summary: 'Key terms and notation are introduced with board/slide visuals.',
    },
    {
      index: 3,
      start_sec: 240,
      end_sec: 420,
      title: 'Main method',
      summary: 'The central algorithm or derivation is walked through step by step.',
    },
    {
      index: 4,
      start_sec: 420,
      end_sec: 540,
      title: 'Worked example',
      summary: 'A concrete example applies the method and highlights common mistakes.',
    },
    {
      index: 5,
      start_sec: 540,
      end_sec: 600,
      title: 'Wrap-up',
      summary: 'Summary takeaways and suggested practice or next lecture preview.',
    },
  ]
  return {
    title,
    chapters,
    display: `**${title}**\n\n${chapters
      .map(
        (ch) =>
          `${ch.index}. **${formatClock(ch.start_sec)}–${formatClock(ch.end_sec)}** — ${ch.title}\n   ${ch.summary}`,
      )
      .join('\n')}`,
    mock: true,
  }
}

function mockFlashcards(videoName) {
  const topic = `${(videoName || 'lecture').replace(/\.[^.]+$/, '')} flashcards`
  const cards = [
    { id: 1, front: 'What problem does this lecture address?', back: 'The core learning goal introduced at the start.' },
    { id: 2, front: 'Key term from the opening', back: 'The main concept defined early in the talk.' },
    { id: 3, front: 'Why does the method matter?', back: 'It improves on the baseline approach discussed.' },
    { id: 4, front: 'Critical intermediate step', back: 'The hinge step before the worked example.' },
    { id: 5, front: 'When does the approach fail?', back: 'Edge cases or assumptions called out by the speaker.' },
    { id: 6, front: 'One takeaway to remember', back: 'The closing summary point to review later.' },
  ]
  return {
    topic,
    cards,
    display: `**${topic}**\n\n_${cards.length} cards — tap a card in the deck to flip._`,
    mock: true,
  }
}

function mockQuiz(videoName) {
  const title = `${(videoName || 'lecture').replace(/\.[^.]+$/, '')} quiz`
  const questions = [
    {
      id: 1,
      type: 'mcq',
      prompt: 'What is the main goal of this lecture?',
      choices: { A: 'Trivia list', B: 'Explain the core method', C: 'Hardware setup', D: 'Grading policy' },
      answer: 'B',
      explanation: 'The opening frames the central problem.',
    },
    {
      id: 2,
      type: 'mcq',
      prompt: 'Which step is most error-prone?',
      choices: {
        A: 'Title slide',
        B: 'Credits',
        C: 'Core derivation / algorithm step',
        D: 'Outro music',
      },
      answer: 'C',
      explanation: 'The speaker highlights a common pitfall mid-lecture.',
    },
    {
      id: 3,
      type: 'short',
      prompt: 'Name one key term defined in the lecture.',
      answer: 'main concept',
      explanation: 'Any clearly defined lecture term is acceptable in mock mode.',
      choices: {},
    },
    {
      id: 4,
      type: 'mcq',
      prompt: 'True or false framing: the method always runs in linear time.',
      choices: { A: 'True', B: 'False', C: 'Not discussed', D: 'Only on GPU' },
      answer: 'B',
      explanation: 'Complexity caveats are usually discussed near the end.',
    },
  ]
  return {
    title,
    questions,
    display: `**${title}**\n\n_${questions.length} questions — answer below, then Grade._`,
    mock: true,
  }
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
  const [liveStartSec, setLiveStartSec] = useState(0)
  const [liveStartInput, setLiveStartInput] = useState('0:00')
  const [followPlayhead, setFollowPlayhead] = useState(true)
  const [outline, setOutline] = useState(null)
  const [activeChapterIdx, setActiveChapterIdx] = useState(null)

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
    setLiveStartSec(0)
    setLiveStartInput('0:00')
    setFollowPlayhead(true)
    setOutline(null)
    setActiveChapterIdx(null)
    setMessages((prev) => [
      ...prev,
      {
        id: `sys-${Date.now()}`,
        role: 'system',
        text: `Loaded "${file.name}". ${
          api.ready
            ? 'Questions will go to the inference API.'
            : 'API offline — answers will use local mock until the server is up.'
        } Scrub the player or set “Live from” before clicking Live.`,
        at: new Date(),
      },
    ])
  }

  function clearVideo() {
    if (videoUrl) URL.revokeObjectURL(videoUrl)
    setVideoFile(null)
    setVideoUrl(null)
    setLiveStartSec(0)
    setLiveStartInput('0:00')
    setFollowPlayhead(true)
    setOutline(null)
    setActiveChapterIdx(null)
    if (videoRef.current) videoRef.current.removeAttribute('src')
  }

  function syncLiveStartFromPlayer() {
    const el = videoRef.current
    if (!el) return
    const t = Math.max(0, el.currentTime || 0)
    setLiveStartSec(t)
    setLiveStartInput(formatClock(t))
  }

  function onLiveStartCommit(raw) {
    const t = parseClock(raw)
    setFollowPlayhead(false)
    setLiveStartSec(t)
    setLiveStartInput(formatClock(t))
    const el = videoRef.current
    if (el && Number.isFinite(el.duration) && el.duration > 0) {
      el.currentTime = Math.min(t, el.duration)
    } else if (el) {
      el.currentTime = t
    }
  }

  async function resolveReply(userText, taskId, opts = {}) {
    const startSec = Math.max(0, Number(opts.startSec) || 0)
    if (!videoFile) {
      return { text: mockReply(taskId, null, userText), segments: null, rounds: null, chapters: null }
    }

    if (!api.ready) {
      if (taskId === 'find') {
        const ground = mockGrounding(userText)
        return {
          text:
            ground.display +
            '\n\n_(API offline — mock grounding. Start src/inference uvicorn on :8000.)_',
          segments: ground.segments,
          rounds: null,
          chapters: null,
        }
      }
      if (taskId === 'live') {
        const live = mockProactive(userText, startSec)
        return {
          text:
            live.display +
            '\n\n_(API offline — mock proactive. Start src/inference uvicorn on :8000.)_',
          segments: null,
          rounds: live.rounds,
          chapters: null,
        }
      }
      if (taskId === 'chapters') {
        const ch = mockChapters(videoFile.name)
        return {
          text:
            ch.display +
            '\n\n_(API offline — mock chapters. Start src/inference uvicorn on :8000.)_',
          segments: null,
          rounds: null,
          chapters: ch.chapters,
          chapterTitle: ch.title,
        }
      }
      if (taskId === 'flashcards') {
        const fc = mockFlashcards(videoFile.name)
        return {
          text:
            fc.display +
            '\n\n_(API offline — mock flashcards. Start src/inference uvicorn on :8000.)_',
          segments: null,
          rounds: null,
          chapters: null,
          cards: fc.cards,
          cardsTopic: fc.topic,
        }
      }
      if (taskId === 'quiz') {
        const qz = mockQuiz(videoFile.name)
        return {
          text:
            qz.display +
            '\n\n_(API offline — mock quiz. Start src/inference uvicorn on :8000.)_',
          segments: null,
          rounds: null,
          chapters: null,
          questions: qz.questions,
          quizTitle: qz.title,
        }
      }
      return {
        text:
          mockReply(taskId, videoFile.name, userText) +
          '\n\n_(API offline — mock fallback. Start src/inference uvicorn on :8000.)_',
        segments: null,
        rounds: null,
        chapters: null,
      }
    }

    const controller = new AbortController()
    abortRef.current = controller
    const timeout = setTimeout(() => controller.abort(), 10 * 60 * 1000)
    try {
      if (taskId === 'find') {
        const result = await groundVideo(videoFile, userText, { signal: controller.signal })
        const badge = result.mock ? ' [api-mock]' : ''
        return {
          text: `${result.display}${badge}`,
          segments: result.segments || [],
          rounds: null,
          chapters: null,
        }
      }

      if (taskId === 'live') {
        const result = await proactiveVideo(videoFile, userText, {
          signal: controller.signal,
          startSec,
        })
        const badge = result.mock ? ' [api-mock]' : ''
        return {
          text: `${result.display}${badge}`,
          segments: null,
          rounds: result.rounds || [],
          chapters: null,
        }
      }

      if (taskId === 'chapters') {
        const result = await chaptersVideo(videoFile, { signal: controller.signal })
        const badge = result.mock ? ' [api-mock]' : ''
        return {
          text: `${result.display}${badge}`,
          segments: null,
          rounds: null,
          chapters: result.chapters || [],
          chapterTitle: result.title,
        }
      }

      if (taskId === 'flashcards') {
        const result = await flashcardsVideo(videoFile, { signal: controller.signal })
        const badge = result.mock ? ' [api-mock]' : ''
        return {
          text: `${result.display}${badge}`,
          segments: null,
          rounds: null,
          chapters: null,
          cards: result.cards || [],
          cardsTopic: result.topic,
        }
      }

      if (taskId === 'quiz') {
        const result = await quizVideo(videoFile, { signal: controller.signal })
        const badge = result.mock ? ' [api-mock]' : ''
        return {
          text: `${result.display}${badge}`,
          segments: null,
          rounds: null,
          chapters: null,
          questions: result.questions || [],
          quizTitle: result.title,
        }
      }

      const result = await askVideo(videoFile, userText, {
        signal: controller.signal,
        task: taskId && taskId !== 'chat' ? taskId : undefined,
      })
      const badge = result.mock ? ' [api-mock]' : ''
      return { text: `${result.answer}${badge}`, segments: null, rounds: null, chapters: null }
    } finally {
      clearTimeout(timeout)
      if (abortRef.current === controller) abortRef.current = null
    }
  }

  function seekTo(seconds) {
    const el = videoRef.current
    if (!el) return
    const t = Math.max(0, Number(seconds) || 0)
    el.currentTime = t
    setFollowPlayhead(true)
    setLiveStartSec(t)
    setLiveStartInput(formatClock(t))
    void el.play?.()
  }

  async function pushExchange(userText, taskId = 'chat', opts = {}) {
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

    const startSec = Math.max(0, Number(opts.startSec) || 0)
    const displayText =
      taskId === 'live' && startSec > 0
        ? `Live from ${formatClock(startSec)}: ${userText}`
        : userText

    const userMsg = {
      id: `u-${Date.now()}`,
      role: 'user',
      text: displayText,
      at: new Date(),
    }
    setMessages((prev) => [...prev, userMsg])
    setBusy(true)

    try {
      const reply = await resolveReply(userText, taskId, { startSec })
      if (reply.chapters?.length) {
        setOutline({
          title: reply.chapterTitle || 'Lecture chapters',
          chapters: reply.chapters,
        })
        setActiveChapterIdx(null)
      }
      setMessages((prev) => [
        ...prev,
        {
          id: `a-${Date.now()}`,
          role: 'assistant',
          text: reply.text,
          segments: reply.segments,
          rounds: reply.rounds,
          chapters: reply.chapters,
          cards: reply.cards,
          cardsTopic: reply.cardsTopic,
          questions: reply.questions,
          quizTitle: reply.quizTitle,
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
    if (task.id === 'find' || task.id === 'live') {
      const topic = draft.trim() || task.chatText
      if (draft.trim()) setDraft('')
      if (task.id === 'live') {
        const startSec = followPlayhead
          ? Math.max(0, videoRef.current?.currentTime || liveStartSec || 0)
          : liveStartSec
        if (followPlayhead) {
          setLiveStartSec(startSec)
          setLiveStartInput(formatClock(startSec))
        }
        void pushExchange(topic, 'live', { startSec })
        return
      }
      void pushExchange(topic, task.id)
      return
    }
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
            <span className="dot" data-ready={api.ready} data-mock={api.mock ? '1' : '0'} />
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
                onTimeUpdate={() => {
                  if (followPlayhead) syncLiveStartFromPlayer()
                  const t = videoRef.current?.currentTime || 0
                  if (outline?.chapters?.length) {
                    const hit = outline.chapters.find(
                      (ch) => t >= ch.start_sec && t < (ch.end_sec > ch.start_sec ? ch.end_sec : ch.start_sec + 1),
                    )
                    const idx = hit?.index ?? null
                    setActiveChapterIdx((prev) => (prev === idx ? prev : idx))
                  }
                }}
                onSeeked={() => {
                  if (!followPlayhead) return
                  syncLiveStartFromPlayer()
                }}
              />
              <div className="live-from-row">
                <label className="live-from-label" htmlFor="live-from">
                  Live from
                </label>
                <input
                  id="live-from"
                  className="live-from-input"
                  type="text"
                  value={liveStartInput}
                  disabled={busy}
                  placeholder="m:ss"
                  title="Start timestamp for Live (m:ss or seconds)"
                  onChange={(e) => {
                    setFollowPlayhead(false)
                    setLiveStartInput(e.target.value)
                  }}
                  onBlur={(e) => onLiveStartCommit(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      e.preventDefault()
                      onLiveStartCommit(e.currentTarget.value)
                      e.currentTarget.blur()
                    }
                  }}
                />
                <button
                  type="button"
                  className="ghost-btn"
                  disabled={busy}
                  onClick={() => {
                    setFollowPlayhead(true)
                    syncLiveStartFromPlayer()
                  }}
                  title="Lock Live start to the video playhead"
                >
                  Use playhead
                </button>
                <span className="live-from-hint">
                  {followPlayhead ? 'follows scrubber' : 'fixed'} · {formatClock(liveStartSec)}
                </span>
              </div>
              {outline?.chapters?.length ? (
                <nav className="chapter-outline" aria-label="Lecture chapters">
                  <p className="chapter-outline-title">{outline.title || 'Chapters'}</p>
                  <ol className="chapter-list">
                    {outline.chapters.map((ch) => (
                      <li key={`outline-${ch.index}`}>
                        <button
                          type="button"
                          className="chapter-item"
                          data-active={activeChapterIdx === ch.index ? '1' : '0'}
                          onClick={() => {
                            setActiveChapterIdx(ch.index)
                            seekTo(ch.start_sec)
                          }}
                        >
                          <span className="chapter-time">
                            {formatClock(ch.start_sec)}–{formatClock(ch.end_sec)}
                          </span>
                          <span className="chapter-name">{ch.title}</span>
                          {ch.summary ? <span className="chapter-summary">{ch.summary}</span> : null}
                        </button>
                      </li>
                    ))}
                  </ol>
                </nav>
              ) : null}
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
                {msg.segments?.length ? (
                  <div className="seek-row" aria-label="Jump to timestamps">
                    {msg.segments.map((seg, idx) => (
                      <button
                        key={`${msg.id}-seg-${idx}`}
                        type="button"
                        className="seek-btn"
                        onClick={() => seekTo(seg.start_sec)}
                        title={seg.evidence || seg.label || 'Seek'}
                      >
                        ▶ {formatClock(seg.start_sec)}
                        {seg.end_sec != null ? `–${formatClock(seg.end_sec)}` : ''}
                      </button>
                    ))}
                  </div>
                ) : null}
                {msg.rounds?.length ? (
                  <div className="round-row" aria-label="Proactive rounds">
                    {msg.rounds.map((round, idx) => (
                      <button
                        key={`${msg.id}-round-${idx}`}
                        type="button"
                        className="round-chip"
                        data-state={round.state}
                        data-high-res={round.high_res ? '1' : '0'}
                        onClick={() => seekTo(round.time_start)}
                        title={`${round.state}${round.high_res ? ' · high-res' : ''}`}
                      >
                        {formatClock(round.time_start)} {String(round.state || '').toUpperCase()}
                        {round.high_res ? ' · HR' : ''}
                      </button>
                    ))}
                  </div>
                ) : null}
                {msg.chapters?.length ? (
                  <div className="seek-row" aria-label="Jump to chapters">
                    {msg.chapters.map((ch, idx) => (
                      <button
                        key={`${msg.id}-ch-${idx}`}
                        type="button"
                        className="seek-btn"
                        onClick={() => {
                          setActiveChapterIdx(ch.index)
                          seekTo(ch.start_sec)
                        }}
                        title={ch.summary || ch.title}
                      >
                        ▶ {formatClock(ch.start_sec)} {ch.title}
                      </button>
                    ))}
                  </div>
                ) : null}
                {msg.cards?.length ? (
                  <FlashcardDeck cards={msg.cards} topic={msg.cardsTopic} />
                ) : null}
                {msg.questions?.length ? (
                  <QuizPanel
                    questions={msg.questions}
                    title={msg.quizTitle}
                    messageId={msg.id}
                  />
                ) : null}
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
              placeholder="Ask… Find topic, or scrub + Live from a timestamp"
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
