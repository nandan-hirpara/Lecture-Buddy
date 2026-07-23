"""CLI helper: ask VideoChat3 about a local lecture video."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow `python scripts/ask_cli.py` from src/inference
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.model import engine  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask VideoChat3-4B about a local video")
    parser.add_argument("video", type=Path, help="Path to lecture video")
    parser.add_argument("question", type=str, help="Question about the video")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--json", action="store_true", help="Print JSON instead of plain text")
    args = parser.parse_args()

    if not settings.mock:
        engine.load()

    result = engine.ask(args.video, args.question, max_new_tokens=args.max_new_tokens)
    if args.json:
        print(
            json.dumps(
                {
                    "answer": result.answer,
                    "model_id": result.model_id,
                    "mock": result.mock,
                    "device": result.device,
                },
                indent=2,
            )
        )
    else:
        print(result.answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
