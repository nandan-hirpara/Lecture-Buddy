import { useState } from 'react'

/**
 * Horizontally scrolling flip cards: front = question, click to reveal answer.
 * @param {{ cards: Array<{ id: number, front: string, back: string, hint?: string }>, topic?: string }} props
 */
export default function FlashcardDeck({ cards, topic }) {
  const [flipped, setFlipped] = useState(() => ({}))

  if (!cards?.length) return null

  function toggle(id) {
    setFlipped((prev) => ({ ...prev, [id]: !prev[id] }))
  }

  function onKeyDown(event, id) {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault()
      toggle(id)
    }
  }

  return (
    <div className="flash-deck" aria-label={topic || 'Flashcards'}>
      <p className="flash-deck-label">
        {topic || 'Flashcards'} · tap a card to flip · scroll sideways
      </p>
      <div className="flash-scroller">
        {cards.map((card) => {
          const isFlipped = Boolean(flipped[card.id])
          return (
            <div
              key={card.id}
              className={`flash-card${isFlipped ? ' is-flipped' : ''}`}
              role="button"
              tabIndex={0}
              aria-pressed={isFlipped}
              onClick={() => toggle(card.id)}
              onKeyDown={(e) => onKeyDown(e, card.id)}
              aria-label={
                isFlipped
                  ? `Answer: ${card.back}. Activate to hide.`
                  : `Question: ${card.front}. Activate to reveal answer.`
              }
            >
              <div className="flash-card-inner">
                <div className="flash-face flash-front">
                  <span className="flash-kicker">Question {card.id}</span>
                  <p className="flash-text">{card.front}</p>
                  {card.hint ? <span className="flash-hint">Hint: {card.hint}</span> : null}
                  <span className="flash-cue">Tap to reveal</span>
                </div>
                <div className="flash-face flash-back">
                  <span className="flash-kicker">Answer</span>
                  <p className="flash-text">{card.back}</p>
                  <span className="flash-cue">Tap to hide</span>
                </div>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
