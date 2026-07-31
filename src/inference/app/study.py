"""Structured study outputs: flashcards + quiz (JSON schemas + grading)."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from .prompts import GROUNDING_INSTRUCTION

logger = logging.getLogger(__name__)

FLASHCARDS_PROMPT = """You are LectureBuddy. Create flashcards for active recall from this lecture video.

Respond with ONLY a fenced JSON block (exact schema):
```json
{{
  "topic": "short lecture topic",
  "cards": [
    {{
      "id": 1,
      "front": "clear question or term",
      "back": "short accurate answer",
      "hint": "optional tiny hint"
    }}
  ]
}}
```
Rules:
- Make 6–8 cards.
- Prefer definitions, why/how, and comparisons over trivia.
- Keep front/back concise (one sentence each when possible).
- Ground every card in the lecture. {grounding}
""".format(grounding=GROUNDING_INSTRUCTION)

QUIZ_PROMPT = """You are LectureBuddy. Write a short graded quiz from this lecture video.

Respond with ONLY a fenced JSON block (exact schema):
```json
{{
  "title": "short quiz title",
  "questions": [
    {{
      "id": 1,
      "type": "mcq",
      "prompt": "question text",
      "choices": {{"A": "…", "B": "…", "C": "…", "D": "…"}},
      "answer": "B",
      "explanation": "one-sentence why"
    }},
    {{
      "id": 2,
      "type": "short",
      "prompt": "short-answer question",
      "answer": "expected key phrase",
      "explanation": "one-sentence why"
    }}
  ]
}}
```
Rules:
- Exactly 4 questions: mix 2–3 multiple-choice (type "mcq") and 1–2 short-answer (type "short").
- MCQ answer must be a single letter A–D present in choices.
- Short-answer "answer" is a concise key phrase for grading (case-insensitive contains match).
- Cover core ideas only. {grounding}
""".format(grounding=GROUNDING_INSTRUCTION)

_JSON_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
_TABLE_ROW = re.compile(
    r"^\s*\|\s*(?P<front>[^|]+?)\s*\|\s*(?P<back>[^|]+?)\s*\|\s*$",
    re.MULTILINE,
)


@dataclass
class Flashcard:
    id: int
    front: str
    back: str
    hint: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "front": self.front,
            "back": self.back,
            "hint": self.hint,
        }


@dataclass
class QuizQuestion:
    id: int
    type: Literal["mcq", "short"]
    prompt: str
    answer: str
    explanation: str = ""
    choices: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "type": self.type,
            "prompt": self.prompt,
            "answer": self.answer,
            "explanation": self.explanation,
        }
        if self.choices:
            payload["choices"] = self.choices
        return payload


def build_flashcards_prompt() -> str:
    return FLASHCARDS_PROMPT


def build_quiz_prompt() -> str:
    return QUIZ_PROMPT


def _try_load_json(blob: str) -> dict[str, Any] | None:
    try:
        data = json.loads(blob)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass
    repaired = blob.strip()
    last = repaired.rfind("}")
    if last < 0:
        return None
    repaired = re.sub(r",\s*$", "", repaired[: last + 1])
    if '"cards"' in repaired or '"questions"' in repaired:
        if not repaired.rstrip().endswith("]"):
            repaired += "]"
        if not repaired.rstrip().endswith("}"):
            repaired += "}"
    try:
        data = json.loads(repaired)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def _json_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    for fence in _JSON_FENCE.finditer(text or ""):
        blob = fence.group(1).strip()
        if blob.startswith("{"):
            candidates.append(blob)
    if not candidates:
        start = (text or "").find("{")
        end = (text or "").rfind("}")
        if start >= 0 and end > start:
            candidates.append(text[start : end + 1])
        elif start >= 0:
            candidates.append(text[start:])
    return candidates


def _normalize_flashcard(item: dict[str, Any], fallback_id: int) -> Flashcard | None:
    front = str(item.get("front") or item.get("question") or item.get("term") or "").strip()
    back = str(item.get("back") or item.get("answer") or item.get("definition") or "").strip()
    if not front or not back:
        return None
    try:
        cid = int(item.get("id") or fallback_id)
    except (TypeError, ValueError):
        cid = fallback_id
    hint = str(item.get("hint") or "").strip()
    return Flashcard(id=cid, front=front, back=back, hint=hint)


def parse_flashcards_answer(text: str) -> tuple[str | None, list[Flashcard]]:
    for blob in _json_candidates(text or ""):
        data = _try_load_json(blob)
        if not data:
            continue
        raw = data.get("cards") or data.get("flashcards") or []
        if not isinstance(raw, list):
            continue
        cards: list[Flashcard] = []
        for i, item in enumerate(raw, start=1):
            if isinstance(item, dict):
                card = _normalize_flashcard(item, i)
                if card is not None:
                    cards.append(card)
        if cards:
            topic = str(data.get("topic") or data.get("title") or "").strip() or None
            return topic, cards[:10]
    # Markdown table fallback: | Front | Back |
    cards = []
    for i, match in enumerate(_TABLE_ROW.finditer(text or ""), start=1):
        front = match.group("front").strip()
        back = match.group("back").strip()
        if front.lower() in {"front", "---", ":---", ":---:"}:
            continue
        if back.lower() in {"back", "---", ":---", ":---:"}:
            continue
        if front and back:
            cards.append(Flashcard(id=i, front=front, back=back))
    return None, cards[:10]


def _normalize_choices(raw: Any) -> dict[str, str]:
    choices: dict[str, str] = {}
    if isinstance(raw, dict):
        for key, val in raw.items():
            letter = str(key).strip().upper()[:1]
            if letter in "ABCD" and str(val).strip():
                choices[letter] = str(val).strip()
    elif isinstance(raw, list):
        letters = "ABCD"
        for i, val in enumerate(raw[:4]):
            if str(val).strip():
                choices[letters[i]] = str(val).strip()
    return choices


def _normalize_quiz_question(item: dict[str, Any], fallback_id: int) -> QuizQuestion | None:
    prompt = str(item.get("prompt") or item.get("question") or "").strip()
    answer = str(item.get("answer") or item.get("correct") or "").strip()
    if not prompt or not answer:
        return None
    try:
        qid = int(item.get("id") or fallback_id)
    except (TypeError, ValueError):
        qid = fallback_id
    qtype_raw = str(item.get("type") or "").strip().lower()
    choices = _normalize_choices(item.get("choices") or item.get("options"))
    if qtype_raw not in {"mcq", "short"}:
        qtype_raw = "mcq" if choices else "short"
    if qtype_raw == "mcq":
        letter = answer.strip().upper()[:1]
        if letter in choices:
            answer = letter
        elif letter in "ABCD" and not choices:
            answer = letter
        elif answer.upper()[:1] in choices:
            answer = answer.upper()[:1]
    explanation = str(item.get("explanation") or item.get("rationale") or "").strip()
    return QuizQuestion(
        id=qid,
        type="mcq" if qtype_raw == "mcq" else "short",  # type: ignore[arg-type]
        prompt=prompt,
        answer=answer,
        explanation=explanation,
        choices=choices,
    )


def parse_quiz_answer(text: str) -> tuple[str | None, list[QuizQuestion]]:
    for blob in _json_candidates(text or ""):
        data = _try_load_json(blob)
        if not data:
            continue
        raw = data.get("questions") or data.get("quiz") or []
        if not isinstance(raw, list):
            continue
        questions: list[QuizQuestion] = []
        for i, item in enumerate(raw, start=1):
            if isinstance(item, dict):
                q = _normalize_quiz_question(item, i)
                if q is not None:
                    questions.append(q)
        if questions:
            title = str(data.get("title") or data.get("topic") or "").strip() or None
            return title, questions[:8]
    return None, []


def grade_quiz(
    questions: list[QuizQuestion],
    responses: dict[str, str] | dict[int, str],
) -> dict[str, Any]:
    """Grade submitted answers. ``responses`` maps question id → user answer."""
    results = []
    correct_n = 0
    for q in questions:
        raw = responses.get(str(q.id), responses.get(q.id, ""))  # type: ignore[arg-type]
        user = str(raw or "").strip()
        ok = False
        if q.type == "mcq":
            ok = user.upper()[:1] == q.answer.upper()[:1]
        else:
            expected = q.answer.strip().lower()
            got = user.lower()
            ok = bool(expected) and (expected in got or got in expected)
        if ok:
            correct_n += 1
        results.append(
            {
                "id": q.id,
                "correct": ok,
                "user_answer": user,
                "expected": q.answer,
                "explanation": q.explanation,
            }
        )
    total = len(questions) or 1
    return {
        "score": correct_n,
        "total": len(questions),
        "percent": round(100.0 * correct_n / total, 1),
        "results": results,
    }


def format_flashcards_markdown(topic: str | None, cards: list[Flashcard]) -> str:
    heading = topic or "Flashcards"
    lines = [f"**{heading}**", "", f"_{len(cards)} cards — tap a card in the deck to flip._", ""]
    for card in cards:
        lines.append(f"{card.id}. **Q:** {card.front}")
        lines.append(f"   **A:** {card.back}")
    if not cards:
        lines.append("Could not parse flashcards from the model output.")
    return "\n".join(lines)


def format_quiz_markdown(title: str | None, questions: list[QuizQuestion]) -> str:
    heading = title or "Lecture quiz"
    lines = [f"**{heading}**", "", f"_{len(questions)} questions — answer below, then Grade._", ""]
    for q in questions:
        lines.append(f"{q.id}. {q.prompt}")
        if q.type == "mcq" and q.choices:
            for letter in sorted(q.choices):
                lines.append(f"   {letter}. {q.choices[letter]}")
        else:
            lines.append("   _(short answer)_")
    if not questions:
        lines.append("Could not parse quiz questions from the model output.")
    return "\n".join(lines)


def mock_flashcards(video_name: str | None = None) -> tuple[str, list[Flashcard], str]:
    topic = f"{(video_name or 'lecture').rsplit('.', 1)[0]} flashcards"
    cards = [
        Flashcard(1, "What problem does this lecture address?", "The core learning goal introduced at the start."),
        Flashcard(2, "Key term from the opening", "The main concept defined early in the talk."),
        Flashcard(3, "Why does the method matter?", "It improves on the baseline approach discussed."),
        Flashcard(4, "Critical intermediate step", "The hinge step before the worked example."),
        Flashcard(5, "When does the approach fail?", "Edge cases or assumptions called out by the speaker."),
        Flashcard(6, "One takeaway to remember", "The closing summary point to review later."),
    ]
    answer = "```json\n" + json.dumps(
        {"topic": topic, "cards": [c.as_dict() for c in cards]}, indent=2
    ) + "\n```"
    return topic, cards, answer


def mock_quiz(video_name: str | None = None) -> tuple[str, list[QuizQuestion], str]:
    title = f"{(video_name or 'lecture').rsplit('.', 1)[0]} quiz"
    questions = [
        QuizQuestion(
            1,
            "mcq",
            "What is the main goal of this lecture?",
            "B",
            "The opening frames the central problem.",
            {"A": "Trivia list", "B": "Explain the core method", "C": "Hardware setup", "D": "Grading policy"},
        ),
        QuizQuestion(
            2,
            "mcq",
            "Which step is most error-prone?",
            "C",
            "The speaker highlights a common pitfall mid-lecture.",
            {"A": "Title slide", "B": "Credits", "C": "Core derivation / algorithm step", "D": "Outro music"},
        ),
        QuizQuestion(
            3,
            "short",
            "Name one key term defined in the lecture.",
            "main concept",
            "Any clearly defined lecture term is acceptable in mock mode.",
        ),
        QuizQuestion(
            4,
            "mcq",
            "True or false framing: the method always runs in linear time.",
            "B",
            "Complexity caveats are usually discussed near the end.",
            {"A": "True", "B": "False", "C": "Not discussed", "D": "Only on GPU"},
        ),
    ]
    answer = "```json\n" + json.dumps(
        {"title": title, "questions": [q.as_dict() for q in questions]}, indent=2
    ) + "\n```"
    return title, questions, answer
