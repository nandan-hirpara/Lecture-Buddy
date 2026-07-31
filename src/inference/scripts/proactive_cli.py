"""CLI: proactive Silence/Standby/Response streaming over a lecture video."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.model import engine  # noqa: E402
from app.streaming import format_proactive_markdown  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run VideoChat3 proactive streaming (Silence/Standby/Response)"
    )
    parser.add_argument("video", type=Path, help="Path to lecture video")
    parser.add_argument("question", type=str, help="What to watch for / answer")
    parser.add_argument("--fps", type=float, default=None, help="Sampling fps")
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument(
        "--start-sec",
        type=float,
        default=None,
        help="Absolute start time in seconds (skip intro)",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if not settings.mock:
        engine.load()

    result = engine.proactive(
        args.video,
        args.question,
        target_fps=args.fps,
        max_rounds=args.max_rounds,
        max_seconds=args.max_seconds,
        max_new_tokens=args.max_new_tokens,
        start_sec=args.start_sec,
    )
    payload = {
        **result.as_dict(),
        "display": format_proactive_markdown(result),
        "model_id": settings.model_id,
        "device": engine.status.get("device"),
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(payload["display"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
