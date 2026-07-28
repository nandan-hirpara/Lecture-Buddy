"""Quick checks for temporal grounding helpers."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.grounding import format_timestamp, mock_grounding, parse_grounding_answer, parse_timestamp


def main() -> None:
    assert parse_timestamp("1:02:03") == 3723
    assert parse_timestamp("12:45") == 12 * 60 + 45
    assert format_timestamp(65) == "1:05"

    fenced = """```json
{"found": true, "segments": [{"start_sec": 12.5, "end_sec": 40, "label": "intro", "evidence": "x"}]}
```"""
    found, segs = parse_grounding_answer(fenced)
    assert found and len(segs) == 1 and segs[0].start_sec == 12.5

    found2, segs2 = parse_grounding_answer("See 3:10-4:20 for the proof.")
    assert found2 and abs(segs2[0].start_sec - 190) < 0.01

    found3, segs3 = parse_grounding_answer('{"found": false, "segments": []}')
    assert found3 is False and segs3 == []

    mf, ms, _ = mock_grounding("recursion")
    assert mf and len(ms) == 2
    print("grounding parser ok")


if __name__ == "__main__":
    main()
