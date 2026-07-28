"""CLI: temporal grounding for a local lecture video."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.grounding import (  # noqa: E402
    build_grounding_prompt,
    format_grounding_markdown,
    format_timestamp,
    mock_grounding,
    parse_grounding_answer,
)
from app.model import engine  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Locate a topic in a lecture video")
    parser.add_argument("video", type=Path, help="Path to lecture video")
    parser.add_argument("query", type=str, help="Topic / event to find")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--json", action="store_true", help="Print JSON")
    args = parser.parse_args()

    query = args.query.strip()
    if settings.mock:
        found, segments, answer = mock_grounding(query)
        device = "mock"
        model_id = settings.model_id
        mock = True
    else:
        engine.load()
        prompt = build_grounding_prompt(query)
        tokens = args.max_new_tokens or min(settings.max_new_tokens, 384)
        result = engine.ask(args.video, prompt, max_new_tokens=tokens)
        found, segments = parse_grounding_answer(result.answer)
        answer = result.answer
        device = result.device
        model_id = result.model_id
        mock = result.mock

    display = format_grounding_markdown(query, found, segments, answer)
    payload = {
        "query": query,
        "found": found,
        "segments": [s.as_dict() for s in segments],
        "answer": answer,
        "display": display,
        "model_id": model_id,
        "mock": mock,
        "device": device,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(display)
        if segments:
            print()
            for seg in segments:
                print(
                    f"  seek {format_timestamp(seg.start_sec)} "
                    f"({seg.start_sec:.1f}s – {seg.end_sec:.1f}s)"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
