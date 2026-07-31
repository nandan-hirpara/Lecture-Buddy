/** Study actions + mock replies for the lecture companion UI. */

const TASKS = [
  {
    id: 'explain',
    label: 'Explain',
    chatText: 'Explain this lecture.',
  },
  {
    id: 'notes',
    label: 'Notes',
    chatText: 'Generate study notes.',
  },
  {
    id: 'flashcards',
    label: 'Flashcards',
    chatText: 'Generate flashcards.',
  },
  {
    id: 'quiz',
    label: 'Quiz',
    chatText: 'Create a quiz.',
  },
  {
    id: 'chapters',
    label: 'Chapters',
    chatText: 'Summarize each chapter.',
  },
  {
    id: 'formulas',
    label: 'Formulas',
    chatText: 'List important formulas.',
  },
  {
    id: 'find',
    label: 'Find topic',
    chatText: 'the main concept',
  },
  {
    id: 'live',
    label: 'Live',
    chatText: 'Watch for the key concept and answer when ready.',
  },
]

function mockReply(taskId, videoName, userText) {
  const name = videoName || 'your lecture'
  const replies = {
    explain: `**Mock explanation** for “${name}”

This lecture walks through the core idea step by step: definitions first, then a worked example, then common pitfalls. Once VideoChat3 is connected, this answer will cite on-screen visuals and spoken context from the video.`,
    notes: `**Mock notes** — ${name}

1. Setup and motivation
2. Key definitions and notation
3. Main algorithm / derivation
4. Worked example
5. Takeaways and practice prompts

_(Stub output — real notes will be grounded in the uploaded lecture.)_`,
    flashcards: `**Mock flashcards**

| Front | Back |
| --- | --- |
| What is the main problem? | … |
| Key term from the lecture | … |
| When does the method fail? | … |

Five cards will be generated from lecture evidence in a later task.`,
    quiz: `**Mock quiz** (3 questions)

1. What problem does this lecture solve?
2. Which step is most error-prone?
3. True/False: the method always runs in linear time.

Answers appear after VideoChat3 grounding is enabled.`,
    chapters: `**Mock chapter summaries** — ${name}

- **0:00–…** Opening and agenda
- **…** Core concept introduction
- **…** Worked example
- **…** Wrap-up and next steps

Chapter boundaries will later come from temporal segmentation.`,
    formulas: `**Mock formulas**

- Definition / identity highlighted on the board
- Main recurrence or closed form
- Complexity / bound used in the lecture

OCR + video QA will replace this stub.`,
    find: `**Mock temporal grounding**

Query: “${userText || 'main concept'}”

→ **Approx. 12:40–14:05** (placeholder)

VideoChat3’s temporal grounding will return real timestamps in a later task.`,
    live: `**Mock proactive stream** for “${userText || 'key concept'}”

- **0s–2s** [SILENCE]: </Silence>
- **2s–4s** [SILENCE]: </Silence>
- **4s–6s** [STANDBY]: </Standby>
- **6s–8s** [RESPONSE] HIGH-RES: Enough evidence to answer.

_(Stub — real Silence/Standby/Response comes from /v1/proactive.)_`,
    chat: `Mock reply to: “${userText}”

I can discuss the lecture once a video is attached and the inference service is live. For now this is a UI stub.`,
  }

  return replies[taskId] || replies.chat
}

function delay(ms = 550) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

export { TASKS, mockReply, delay }
