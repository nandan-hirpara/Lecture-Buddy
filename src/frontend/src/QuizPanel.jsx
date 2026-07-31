import { useMemo, useState } from 'react'
import { gradeQuiz } from './api.js'

/**
 * Interactive quiz with client/API grading.
 * @param {{ questions: Array<object>, title?: string, messageId: string }} props
 */
export default function QuizPanel({ questions, title, messageId }) {
  const [responses, setResponses] = useState({})
  const [grade, setGrade] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  const resultById = useMemo(() => {
    const map = {}
    for (const item of grade?.results || []) map[item.id] = item
    return map
  }, [grade])

  if (!questions?.length) return null

  function setAnswer(id, value) {
    setGrade(null)
    setError(null)
    setResponses((prev) => ({ ...prev, [String(id)]: value }))
  }

  async function onGrade() {
    setBusy(true)
    setError(null)
    try {
      const payload = await gradeQuiz(questions, responses)
      setGrade(payload)
    } catch (err) {
      // Local fallback if API grade endpoint is offline
      const results = questions.map((q) => {
        const user = String(responses[String(q.id)] || '').trim()
        let correct = false
        if (q.type === 'mcq') {
          correct = user.toUpperCase().slice(0, 1) === String(q.answer || '').toUpperCase().slice(0, 1)
        } else {
          const expected = String(q.answer || '').trim().toLowerCase()
          const got = user.toLowerCase()
          correct = Boolean(expected) && (expected.includes(got) || got.includes(expected))
        }
        return {
          id: q.id,
          correct,
          user_answer: user,
          expected: q.answer,
          explanation: q.explanation || '',
        }
      })
      const score = results.filter((r) => r.correct).length
      setGrade({
        score,
        total: questions.length,
        percent: Math.round((1000 * score) / Math.max(1, questions.length)) / 10,
        results,
      })
      if (err?.message) setError(`Graded locally (${err.message})`)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="quiz-panel" aria-label={title || 'Quiz'}>
      <p className="quiz-panel-label">{title || 'Quiz'}</p>
      <ol className="quiz-list">
        {questions.map((q) => {
          const marked = resultById[q.id]
          return (
            <li key={`${messageId}-q-${q.id}`} className="quiz-item" data-correct={marked ? (marked.correct ? '1' : '0') : undefined}>
              <p className="quiz-prompt">
                {q.id}. {q.prompt}
              </p>
              {q.type === 'mcq' && q.choices ? (
                <div className="quiz-choices" role="radiogroup" aria-label={`Choices for question ${q.id}`}>
                  {Object.keys(q.choices)
                    .sort()
                    .map((letter) => (
                      <label key={`${messageId}-${q.id}-${letter}`} className="quiz-choice">
                        <input
                          type="radio"
                          name={`${messageId}-q-${q.id}`}
                          value={letter}
                          checked={responses[String(q.id)] === letter}
                          onChange={() => setAnswer(q.id, letter)}
                          disabled={busy}
                        />
                        <span>
                          <strong>{letter}.</strong> {q.choices[letter]}
                        </span>
                      </label>
                    ))}
                </div>
              ) : (
                <input
                  className="quiz-short"
                  type="text"
                  value={responses[String(q.id)] || ''}
                  onChange={(e) => setAnswer(q.id, e.target.value)}
                  placeholder="Your answer"
                  disabled={busy}
                />
              )}
              {marked ? (
                <p className="quiz-feedback">
                  {marked.correct ? 'Correct' : `Expected: ${marked.expected}`}
                  {marked.explanation ? ` — ${marked.explanation}` : ''}
                </p>
              ) : null}
            </li>
          )
        })}
      </ol>
      <div className="quiz-actions">
        <button type="button" className="send-btn" disabled={busy} onClick={() => void onGrade()}>
          {busy ? 'Grading…' : 'Grade quiz'}
        </button>
        {grade ? (
          <span className="quiz-score">
            Score {grade.score}/{grade.total} ({grade.percent}%)
          </span>
        ) : null}
      </div>
      {error ? <p className="quiz-error">{error}</p> : null}
    </div>
  )
}
