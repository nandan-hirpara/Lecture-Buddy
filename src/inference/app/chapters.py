"""Chapter segmentation: logical lecture chapters with times + summaries."""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .grounding import format_timestamp, parse_timestamp
from .prompts import GROUNDING_INSTRUCTION

logger = logging.getLogger(__name__)

CHAPTERS_PROMPT = """You are LectureBuddy. Label fixed time windows of this lecture video.

Video duration: {duration_label} ({duration_sec:.0f} seconds).

The chapter TIME RANGES are already chosen. Do NOT invent new times and do NOT skip windows.
Fill a title + one-sentence summary for EVERY window below ({num_chapters} items):

{slot_list}

Respond with ONLY this JSON fence (compact):
```json
{{
  "title": "short lecture title",
  "chapters": [
    {{"index": 1, "title": "short title", "summary": "one sentence"}},
    {{"index": 2, "title": "short title", "summary": "one sentence"}}
  ]
}}
```
Rules:
- Include exactly {num_chapters} objects, indices 1..{num_chapters}.
- Titles: 3–8 words. Summary: ONE short sentence grounded in that window of the video.
- Cover later windows too (not only the opening).
- Do not output start_sec/end_sec (times are fixed server-side).

{grounding}"""

_JSON_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
_CHAPTER_HEAD = re.compile(
    r"(?:^|\n)\s*(?:#{1,3}\s*)?(?:chapter\s*)?(?P<idx>\d+)[\.\):\-–—]\s*"
    r"(?P<title>[^\n(]+?)"
    r"(?:\s*[\(\[]\s*(?P<span>[^\)\]]+)[\)\]])?"
    r"[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
_RANGE = re.compile(
    r"(?P<s>(?:\d{1,2}:)?\d{1,2}:\d{2}(?:\.\d+)?|\d+(?:\.\d+)?\s*s(?:ec(?:onds?)?)?)"
    r"\s*[-–—to]+\s*"
    r"(?P<e>(?:\d{1,2}:)?\d{1,2}:\d{2}(?:\.\d+)?|\d+(?:\.\d+)?\s*s(?:ec(?:onds?)?)?)",
    re.IGNORECASE,
)


@dataclass
class Chapter:
    index: int
    start_sec: float
    end_sec: float
    title: str
    summary: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "title": self.title,
            "summary": self.summary,
        }


def probe_video_duration(video_path: str | Path) -> float | None:
    """Return duration in seconds, or None if unreadable."""
    try:
        import cv2
    except ImportError:
        return None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        if fps > 0 and frames > 0:
            return frames / fps
        msec = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
        return msec / 1000.0 if msec > 0 else None
    finally:
        cap.release()


def chapter_budget(duration_sec: float | None) -> tuple[int, int]:
    """Suggested (min, max) chapter counts from duration."""
    if not duration_sec or duration_sec <= 0:
        return 3, 6
    minutes = duration_sec / 60.0
    # ~1 chapter per ~2 minutes, clamped for laptop inference cost.
    min_c = max(3, min(5, int(math.ceil(minutes / 2.5))))
    max_c = max(min_c, min(6, int(math.ceil(minutes / 2.0))))
    return min_c, max_c


def build_time_slots(duration_sec: float | None, count: int | None = None) -> list[tuple[float, float]]:
    """Evenly spaced (start, end) windows covering the full video."""
    dur = float(duration_sec) if duration_sec and duration_sec > 0 else 600.0
    min_c, max_c = chapter_budget(dur)
    n = int(count) if count is not None else min_c
    n = max(min_c, min(max_c, n))
    step = dur / n
    slots: list[tuple[float, float]] = []
    for i in range(n):
        start = round(i * step, 1)
        end = round(dur if i == n - 1 else (i + 1) * step, 1)
        if end <= start:
            end = min(dur, start + step)
        slots.append((start, end))
    return slots


def build_chapters_prompt(
    duration_sec: float | None = None,
    slots: list[tuple[float, float]] | None = None,
) -> str:
    dur = float(duration_sec) if duration_sec and duration_sec > 0 else 600.0
    windows = slots or build_time_slots(dur)
    slot_lines = []
    for i, (start, end) in enumerate(windows, start=1):
        slot_lines.append(
            f"{i}. {format_timestamp(start)}–{format_timestamp(end)} "
            f"({start:.0f}s–{end:.0f}s)"
        )
    return CHAPTERS_PROMPT.format(
        grounding=GROUNDING_INSTRUCTION,
        duration_sec=dur,
        duration_label=format_timestamp(dur),
        num_chapters=len(windows),
        slot_list="\n".join(slot_lines),
    )


def apply_time_slots(
    chapters: list[Chapter],
    slots: list[tuple[float, float]],
) -> list[Chapter]:
    """Force full-video coverage by binding model labels onto fixed time windows."""
    if not slots:
        return chapters

    by_index: dict[int, Chapter] = {}
    for ch in chapters:
        by_index[int(ch.index)] = ch

    ordered = sorted(chapters, key=lambda c: (c.start_sec, c.index))
    filled: list[Chapter] = []
    for i, (start, end) in enumerate(slots, start=1):
        src = by_index.get(i)
        if src is None and i - 1 < len(ordered):
            src = ordered[i - 1]
        title = (src.title if src else "").strip() or f"Section {i}"
        summary = (src.summary if src else "").strip()
        if not summary and src is None:
            summary = "No model label for this window — scrub here to review."
        filled.append(
            Chapter(
                index=i,
                start_sec=start,
                end_sec=end,
                title=title,
                summary=summary,
            )
        )
    return filled


def _normalize_chapter(item: dict[str, Any], fallback_index: int) -> Chapter | None:
    start = item.get("start_sec", item.get("start", item.get("start_time")))
    end = item.get("end_sec", item.get("end", item.get("end_time")))
    try:
        start_f = float(start) if start is not None else float(fallback_index - 1)
        end_f = float(end) if end is not None else start_f + 1.0
    except (TypeError, ValueError):
        start_f = float(fallback_index - 1)
        end_f = start_f + 1.0
    if end_f < start_f:
        start_f, end_f = end_f, start_f

    try:
        index = int(item.get("index") or item.get("id") or fallback_index)
    except (TypeError, ValueError):
        index = fallback_index

    title = str(item.get("title") or item.get("label") or item.get("name") or "").strip()
    if not title:
        title = f"Chapter {index}"
    summary = str(
        item.get("summary") or item.get("description") or item.get("evidence") or ""
    ).strip()
    return Chapter(
        index=index,
        start_sec=start_f,
        end_sec=end_f,
        title=title,
        summary=summary,
    )


def _try_load_json(blob: str) -> dict[str, Any] | None:
    try:
        data = json.loads(blob)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass

    # Truncated generation: keep complete chapter objects, close the array/object.
    repaired = blob.strip()
    if '"chapters"' not in repaired and '"sections"' not in repaired:
        return None
    # Cut after the last complete `{...}` object inside the chapters array.
    last_obj_end = repaired.rfind("}")
    if last_obj_end < 0:
        return None
    repaired = repaired[: last_obj_end + 1]
    # Drop a trailing comma after the last object.
    repaired = re.sub(r",\s*$", "", repaired)
    if not repaired.rstrip().endswith("]"):
        repaired += "]"
    if not repaired.rstrip().endswith("}"):
        repaired += "}"
    try:
        data = json.loads(repaired)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        logger.debug("Could not repair truncated chapters JSON")
        return None


def _parse_json_chapters(text: str) -> tuple[str | None, list[Chapter]]:
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
            # Truncated mid-JSON (no closing brace yet)
            candidates.append(text[start:])

    for blob in candidates:
        data = _try_load_json(blob)
        if not data:
            continue
        raw = data.get("chapters") or data.get("sections") or data.get("segments") or []
        if not isinstance(raw, list):
            continue
        chapters: list[Chapter] = []
        for i, item in enumerate(raw, start=1):
            if isinstance(item, dict):
                ch = _normalize_chapter(item, i)
                if ch is not None:
                    chapters.append(ch)
        if chapters:
            title = str(data.get("title") or data.get("lecture_title") or "").strip() or None
            return title, chapters
    return None, []


def _parse_markdown_chapters(text: str) -> list[Chapter]:
    chapters: list[Chapter] = []
    matches = list(_CHAPTER_HEAD.finditer(text or ""))
    for i, match in enumerate(matches):
        idx = int(match.group("idx"))
        title = (match.group("title") or "").strip(" -:–—")
        start_sec, end_sec = 0.0, 0.0
        span = match.group("span")
        if span:
            rm = _RANGE.search(span)
            if rm:
                s = parse_timestamp(rm.group("s"))
                e = parse_timestamp(rm.group("e"))
                if s is not None and e is not None:
                    start_sec, end_sec = s, e
                    if end_sec < start_sec:
                        start_sec, end_sec = end_sec, start_sec
        block_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = (text or "")[match.end() : block_end].strip()
        # Drop a leading time range line if present
        body = _RANGE.sub("", body, count=1).strip()
        summary = " ".join(body.split())[:400]
        chapters.append(
            Chapter(
                index=idx,
                start_sec=start_sec,
                end_sec=end_sec,
                title=title or f"Chapter {idx}",
                summary=summary,
            )
        )
    return chapters


def parse_chapters_answer(text: str) -> tuple[str | None, list[Chapter]]:
    """Extract lecture title + ordered chapters from a model answer."""
    title, chapters = _parse_json_chapters(text or "")
    if chapters:
        return title, chapters[:8]
    md = _parse_markdown_chapters(text or "")
    if md:
        return title, md[:8]
    return title, []


def normalize_chapters(
    chapters: list[Chapter],
    duration_sec: float | None = None,
) -> list[Chapter]:
    """Sort, reindex, and clamp chapter times to the video duration."""
    if not chapters:
        return []
    ordered = sorted(chapters, key=lambda c: (c.start_sec, c.end_sec))
    dur = float(duration_sec) if duration_sec and duration_sec > 0 else None
    fixed: list[Chapter] = []
    for i, ch in enumerate(ordered, start=1):
        start = max(0.0, float(ch.start_sec))
        end = max(start, float(ch.end_sec))
        if dur is not None:
            start = min(start, dur)
            end = min(max(end, start), dur)
        fixed.append(
            Chapter(
                index=i,
                start_sec=start,
                end_sec=end,
                title=ch.title,
                summary=ch.summary,
            )
        )
    if dur is not None and fixed:
        # Stretch the last chapter to the end if it already covers most of the tail.
        if fixed[-1].end_sec < dur and (dur - fixed[-1].end_sec) <= max(30.0, dur * 0.08):
            fixed[-1] = Chapter(
                index=fixed[-1].index,
                start_sec=fixed[-1].start_sec,
                end_sec=dur,
                title=fixed[-1].title,
                summary=fixed[-1].summary,
            )
    return fixed


def coverage_ratio(chapters: list[Chapter], duration_sec: float | None) -> float:
    if not chapters or not duration_sec or duration_sec <= 0:
        return 0.0
    return min(1.0, max(ch.end_sec for ch in chapters) / duration_sec)


def format_chapters_markdown(
    title: str | None,
    chapters: list[Chapter],
    answer: str,
    *,
    duration_sec: float | None = None,
) -> str:
    heading = title.strip() if title else "Lecture chapters"
    lines = [f"**{heading}**", ""]
    if duration_sec and duration_sec > 0:
        lines.append(f"_Video length: {format_timestamp(duration_sec)}_")
        lines.append("")
    if not chapters:
        lines.append("Could not parse chapter boundaries. See model notes below.")
    else:
        for ch in chapters:
            span = f"{format_timestamp(ch.start_sec)}–{format_timestamp(ch.end_sec)}"
            lines.append(f"{ch.index}. **{span}** — {ch.title}")
            if ch.summary:
                lines.append(f"   {ch.summary}")
        cov = coverage_ratio(chapters, duration_sec)
        if duration_sec and cov < 0.85:
            lines.extend(
                [
                    "",
                    f"_Note: outline only covers ~{cov:.0%} of the video "
                    f"(through {format_timestamp(max(c.end_sec for c in chapters))})._",
                ]
            )
    raw = (answer or "").strip()
    if raw and not chapters:
        lines.extend(["", "Model notes:", raw])
    return "\n".join(lines)


def mock_chapters(video_name: str | None = None) -> tuple[str, list[Chapter], str]:
    """Deterministic stub when LECTUREBUDDY_MOCK=1."""
    name = (video_name or "lecture").rsplit(".", 1)[0]
    title = f"{name} (mock outline)"
    chapters = [
        Chapter(
            1,
            0.0,
            90.0,
            "Opening and agenda",
            "Instructor sets motivation and lists what the lecture will cover.",
        ),
        Chapter(
            2,
            90.0,
            240.0,
            "Core definitions",
            "Key terms and notation are introduced with board/slide visuals.",
        ),
        Chapter(
            3,
            240.0,
            420.0,
            "Main method",
            "The central algorithm or derivation is walked through step by step.",
        ),
        Chapter(
            4,
            420.0,
            540.0,
            "Worked example",
            "A concrete example applies the method and highlights common mistakes.",
        ),
        Chapter(
            5,
            540.0,
            600.0,
            "Wrap-up",
            "Summary takeaways and suggested practice or next lecture preview.",
        ),
    ]
    payload = {
        "title": title,
        "chapters": [c.as_dict() for c in chapters],
    }
    answer = (
        "```json\n"
        + json.dumps(payload, indent=2)
        + "\n```\n\n"
        "[mock] Replace with VideoChat3 chapter segmentation when LECTUREBUDDY_MOCK=0."
    )
    return title, chapters, answer
