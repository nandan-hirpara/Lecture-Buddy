"""Quick parser / grade checks for structured study outputs."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.study import (  # noqa: E402
    grade_quiz,
    mock_flashcards,
    mock_quiz,
    parse_flashcards_answer,
    parse_quiz_answer,
)


def main() -> int:
    sample_cards = """
```json
{
  "topic": "Recursion",
  "cards": [
    {"id": 1, "front": "What is a base case?", "back": "The terminating condition."},
    {"id": 2, "front": "What is recursion?", "back": "A function calling itself."}
  ]
}
```
"""
    topic, cards = parse_flashcards_answer(sample_cards)
    assert topic == "Recursion"
    assert len(cards) == 2
    assert cards[0].front.startswith("What is a base")

    sample_quiz = """
```json
{
  "title": "Recursion quiz",
  "questions": [
    {
      "id": 1,
      "type": "mcq",
      "prompt": "Pick the base case role",
      "choices": {"A": "Loop forever", "B": "Stop recursion", "C": "Allocate GPU", "D": "Sort"},
      "answer": "B",
      "explanation": "Stops the call chain."
    },
    {
      "id": 2,
      "type": "short",
      "prompt": "Name the call structure",
      "answer": "call stack",
      "explanation": "Frames push/pop."
    }
  ]
}
```
"""
    title, questions = parse_quiz_answer(sample_quiz)
    assert title == "Recursion quiz"
    assert len(questions) == 2
    graded = grade_quiz(questions, {"1": "B", "2": "the call stack grows"})
    assert graded["score"] == 2

    _, mock_cards, _ = mock_flashcards("demo.mp4")
    _, mock_qs, _ = mock_quiz("demo.mp4")
    assert len(mock_cards) >= 4
    assert len(mock_qs) >= 3
    print("study parse ok", len(cards), len(questions), graded["percent"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
