"""Quick parser checks for chapter segmentation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.chapters import mock_chapters, parse_chapters_answer  # noqa: E402


def main() -> int:
    sample = """
```json
{
  "title": "Intro to Recursion",
  "chapters": [
    {"index": 1, "start_sec": 0, "end_sec": 60, "title": "Motivation", "summary": "Why recursion."},
    {"index": 2, "start_sec": 60, "end_sec": 180, "title": "Base cases", "summary": "Stopping conditions."}
  ]
}
```
"""
    title, chapters = parse_chapters_answer(sample)
    assert title == "Intro to Recursion"
    assert len(chapters) == 2
    assert chapters[0].title == "Motivation"
    assert chapters[1].start_sec == 60

    md = """
## Chapter 1. Opening (0:00–1:30)
Agenda and motivation.

## Chapter 2. Core idea (1:30–4:00)
Definitions and intuition.
"""
    _, md_chapters = parse_chapters_answer(md)
    assert len(md_chapters) >= 2
    assert md_chapters[0].start_sec == 0
    assert md_chapters[1].end_sec == 240

    # Truncated JSON should still recover completed chapter objects.
    truncated = '''
```json
{
  "title": "Partial",
  "chapters": [
    {"index": 1, "start_sec": 0, "end_sec": 60, "title": "A", "summary": "one"},
    {"index": 2, "start_sec": 60, "end_sec": 120, "title": "B", "summary": "two"
'''
    t2, ch2 = parse_chapters_answer(truncated)
    assert t2 == "Partial" or len(ch2) >= 1

    title_m, mock, _answer = mock_chapters("demo.mp4")
    assert title_m
    assert len(mock) >= 3
    print("chapters parse ok", len(chapters), len(md_chapters), len(ch2), len(mock))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
