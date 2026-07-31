"""Lecture study prompts layered on top of VideoChat3 video QA."""

from __future__ import annotations

from typing import Final

# Known study actions (must stay in sync with frontend TASKS ids).
STUDY_TASKS: Final[frozenset[str]] = frozenset(
    {
        "explain",
        "notes",
        "flashcards",
        "quiz",
        "chapters",
        "formulas",
        "find",
    }
)

GROUNDING_INSTRUCTION = (
    "Ground every claim in what is said or shown in this lecture video. "
    "If something is unclear or not covered, say so instead of inventing details."
)

LECTURE_PROMPTS: Final[dict[str, str]] = {
    "explain": f"""You are LectureBuddy, a study companion for recorded lectures.
Watch the video carefully, then explain the main topic to a student who just watched it.

Structure your answer as:
1. One-sentence overview
2. Key definitions and intuition (plain language)
3. The central method, argument, or derivation
4. A concrete example from the lecture if one appears
5. Common pitfalls or caveats mentioned

{GROUNDING_INSTRUCTION}""",
    "notes": f"""You are LectureBuddy. Produce concise study notes for this lecture video.

Format as markdown:
- Title (inferred topic)
- Learning objectives (3–5 bullets)
- Outline with numbered sections and short bullets under each
- Key terms glossary (term — definition)
- Takeaways (3 bullets)

Keep notes skimmable and faithful to the video. {GROUNDING_INSTRUCTION}""",
    "flashcards": f"""You are LectureBuddy. Create flashcards for active recall from this lecture video.

Output a markdown table with columns: Front | Back
- Make 6–8 cards
- Front: a clear question, term, or prompt
- Back: a short accurate answer grounded in the lecture
- Prefer definitions, why/how questions, and comparisons over trivia

{GROUNDING_INSTRUCTION}""",
    "quiz": f"""You are LectureBuddy. Write a short quiz to check understanding of this lecture video.

Requirements:
- 4 questions mixing multiple-choice (A–D) and short-answer
- Cover core ideas, not obscure asides
- After all questions, add an **Answer key** with brief justifications

{GROUNDING_INSTRUCTION}""",
    "chapters": f"""You are LectureBuddy. Segment this lecture into logical chapters and summarize each.

For each chapter provide:
- Approximate time range if you can infer it (MM:SS–MM:SS), otherwise order only (Chapter 1, 2, …)
- Short title
- 2–4 sentence summary of what is taught

Aim for 3–6 chapters covering the full video with MM:SS–MM:SS ranges when possible.
Prefer the dedicated /v1/chapters JSON schema when available. {GROUNDING_INSTRUCTION}""",
    "formulas": f"""You are LectureBuddy. Extract important formulas, equations, and formal identities from this lecture video.

For each item list:
- Name or role (e.g. definition, recurrence, complexity bound)
- The formula in plain text or LaTeX-like notation
- When / why it is used in the lecture

If the board or slides show a derivation, include the final result and one intermediate form if emphasized. {GROUNDING_INSTRUCTION}""",
    "find": f"""You are LectureBuddy. Locate where the following topic is introduced or explained in this lecture video:

Topic: {{topic}}

Reply with:
1. Best timestamp range you can infer (MM:SS–MM:SS), or "unknown time" if timing is unclear
2. What is said or shown at that moment
3. Brief before/after context

If the topic never appears, say so and point to the closest related segment. {GROUNDING_INSTRUCTION}""",
}


def build_lecture_prompt(task: str | None, question: str) -> str:
    """
    Map a study-task id + user question into a full video-QA prompt.

    - ``chat`` / unknown / missing task → use the question as-is
    - ``find`` → fill the topic slot from the question text
    - other study tasks → use the lecture template (question is only UI label)
    """
    q = (question or "").strip()
    key = (task or "").strip().lower() or None

    if not key or key == "chat" or key not in LECTURE_PROMPTS:
        if not q:
            raise ValueError("Question must be non-empty.")
        return q

    template = LECTURE_PROMPTS[key]

    if key == "find":
        topic = q or "the main concept of this lecture"
        return template.format(topic=topic)

    return template
