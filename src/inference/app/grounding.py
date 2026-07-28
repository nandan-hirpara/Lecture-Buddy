"""Temporal grounding: locate lecture topics → timestamp segments."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .prompts import GROUNDING_INSTRUCTION

# Prefer structured JSON from the model; fall back to MM:SS regex parsing.
GROUNDING_PROMPT = """You are LectureBuddy performing temporal grounding on a lecture video.

Locate when this topic is introduced or explained:
"{query}"

Respond with a fenced JSON block first (exact schema):
```json
{{
  "found": true,
  "segments": [
    {{
      "start_sec": 12.5,
      "end_sec": 48.0,
      "label": "short title for this moment",
      "evidence": "what is said or shown"
    }}
  ]
}}
```
Rules:
- Times are seconds from the start of the video (floats allowed).
- Use 1–3 segments max; prefer the first clear introduction.
- If the topic is missing, set "found": false and "segments": [].
- After the JSON fence, you may add one short paragraph of context.

{grounding}"""


_JSON_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
_RANGE = re.compile(
    r"(?P<s>(?:\d{1,2}:)?\d{1,2}:\d{2}(?:\.\d+)?|\d+(?:\.\d+)?\s*s(?:ec(?:onds?)?)?)"
    r"\s*[-–—to]+\s*"
    r"(?P<e>(?:\d{1,2}:)?\d{1,2}:\d{2}(?:\.\d+)?|\d+(?:\.\d+)?\s*s(?:ec(?:onds?)?)?)",
    re.IGNORECASE,
)
_CLOCK = re.compile(r"^(?:(\d{1,2}):)?(\d{1,2}):(\d{2}(?:\.\d+)?)$")
_SECONDS = re.compile(r"^(\d+(?:\.\d+)?)\s*s(?:ec(?:onds?)?)?$", re.IGNORECASE)


@dataclass
class GroundSegment:
    start_sec: float
    end_sec: float
    label: str = ""
    evidence: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "label": self.label,
            "evidence": self.evidence,
        }


def build_grounding_prompt(query: str) -> str:
    q = (query or "").strip()
    if not q:
        raise ValueError("query must be non-empty")
    return GROUNDING_PROMPT.format(query=q, grounding=GROUNDING_INSTRUCTION)


def parse_timestamp(token: str) -> float | None:
    """Parse HH:MM:SS, MM:SS, or '12.5s' into seconds."""
    raw = (token or "").strip()
    if not raw:
        return None

    m = _SECONDS.match(raw)
    if m:
        return float(m.group(1))

    m = _CLOCK.match(raw)
    if m:
        hours = int(m.group(1) or 0)
        minutes = int(m.group(2))
        seconds = float(m.group(3))
        return hours * 3600 + minutes * 60 + seconds

    try:
        return float(raw)
    except ValueError:
        return None


def format_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"


def _normalize_segment(item: dict[str, Any]) -> GroundSegment | None:
    start = item.get("start_sec", item.get("start", item.get("start_time")))
    end = item.get("end_sec", item.get("end", item.get("end_time")))
    try:
        start_f = float(start)
        end_f = float(end) if end is not None else start_f
    except (TypeError, ValueError):
        return None
    if end_f < start_f:
        start_f, end_f = end_f, start_f
    label = str(item.get("label") or item.get("title") or "").strip()
    evidence = str(item.get("evidence") or item.get("description") or "").strip()
    return GroundSegment(start_sec=start_f, end_sec=end_f, label=label, evidence=evidence)


def _parse_json_segments(text: str) -> tuple[bool | None, list[GroundSegment]]:
    candidates: list[str] = []
    for fence in _JSON_FENCE.finditer(text or ""):
        blob = fence.group(1).strip()
        if blob.startswith("{"):
            candidates.append(blob)
    # Fallback: first {...} span that contains "segments"
    if not candidates:
        start = (text or "").find("{")
        end = (text or "").rfind("}")
        if start >= 0 and end > start:
            candidates.append(text[start : end + 1])

    for blob in candidates:
        try:
            data = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        found = data.get("found")
        raw_segs = data.get("segments") or data.get("intervals") or []
        segments: list[GroundSegment] = []
        if isinstance(raw_segs, list):
            for item in raw_segs:
                if isinstance(item, dict):
                    seg = _normalize_segment(item)
                    if seg is not None:
                        segments.append(seg)
        if isinstance(found, bool):
            found_flag: bool | None = found
        else:
            found_flag = bool(segments) if segments else None
        if found_flag is not None or segments:
            return found_flag, segments
    return None, []


def _parse_regex_segments(text: str) -> list[GroundSegment]:
    segments: list[GroundSegment] = []
    for match in _RANGE.finditer(text):
        start = parse_timestamp(match.group("s"))
        end = parse_timestamp(match.group("e"))
        if start is None or end is None:
            continue
        if end < start:
            start, end = end, start
        tail = text[match.end() : match.end() + 120].strip(" \n:-—")
        evidence = re.split(r"[\n.]", tail, maxsplit=1)[0].strip()[:160]
        segments.append(
            GroundSegment(
                start_sec=start,
                end_sec=end,
                label=f"{format_timestamp(start)}–{format_timestamp(end)}",
                evidence=evidence,
            )
        )
    return segments


def parse_grounding_answer(text: str) -> tuple[bool, list[GroundSegment]]:
    """Extract found flag + segments from a model answer."""
    found_json, segments = _parse_json_segments(text or "")
    if segments:
        return True if found_json is None else found_json, segments[:3]
    regex_segs = _parse_regex_segments(text or "")
    if regex_segs:
        return True, regex_segs[:3]
    if found_json is False:
        return False, []
    lowered = (text or "").lower()
    if any(p in lowered for p in ("not found", "never appears", "could not find", "no mention")):
        return False, []
    return False, []


def format_grounding_markdown(
    query: str, found: bool, segments: list[GroundSegment], answer: str
) -> str:
    """Human-readable chat body (also keeps raw model text)."""
    lines = [f'**Temporal grounding** for “{query}”', ""]
    if not found or not segments:
        lines.append("No confident timestamp match. See model notes below.")
    else:
        lines.append("Jump points:")
        for seg in segments:
            span = f"{format_timestamp(seg.start_sec)}–{format_timestamp(seg.end_sec)}"
            label = seg.label or "segment"
            bit = f"- **{span}** — {label}"
            if seg.evidence:
                bit += f": {seg.evidence}"
            lines.append(bit)
    raw = (answer or "").strip()
    if raw:
        lines.extend(["", "Model notes:", raw])
    return "\n".join(lines)


def mock_grounding(query: str) -> tuple[bool, list[GroundSegment], str]:
    """Deterministic stub when LECTUREBUDDY_MOCK=1."""
    segments = [
        GroundSegment(
            start_sec=12.0,
            end_sec=45.0,
            label="First introduction (mock)",
            evidence=f'Placeholder span for “{query}”.',
        ),
        GroundSegment(
            start_sec=120.0,
            end_sec=150.0,
            label="Follow-up mention (mock)",
            evidence="Secondary placeholder span.",
        ),
    ]
    answer = (
        "```json\n"
        + json.dumps(
            {"found": True, "segments": [s.as_dict() for s in segments]},
            indent=2,
        )
        + "\n```\n\n"
        "[mock] Replace with VideoChat3 temporal grounding when LECTUREBUDDY_MOCK=0."
    )
    return True, segments, answer
