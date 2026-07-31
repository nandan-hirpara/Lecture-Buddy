"""CLI: chapter segmentation for a local lecture video."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.chapters import (  # noqa: E402
    apply_time_slots,
    build_chapters_prompt,
    build_time_slots,
    format_chapters_markdown,
    mock_chapters,
    parse_chapters_answer,
    probe_video_duration,
)
from app.config import settings  # noqa: E402
from app.grounding import format_timestamp  # noqa: E402
from app.model import engine  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Segment a lecture into timed chapters")
    parser.add_argument("video", type=Path, help="Path to lecture video")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--json", action="store_true", help="Print JSON")
    args = parser.parse_args()

    video_name = args.video.name
    duration = probe_video_duration(args.video)
    slots = build_time_slots(duration)
    if settings.mock:
        title, chapters, answer = mock_chapters(video_name)
        device = "mock"
        model_id = settings.model_id
        mock = True
    else:
        engine.load()
        prompt = build_chapters_prompt(duration, slots=slots)
        tokens = args.max_new_tokens or max(settings.max_new_tokens, 512)
        result = engine.ask(
            args.video,
            prompt,
            max_new_tokens=tokens,
            max_frames=settings.max_frames,
        )
        title, chapters = parse_chapters_answer(result.answer)
        answer = result.answer
        device = result.device
        model_id = result.model_id
        mock = result.mock

    chapters = apply_time_slots(chapters, slots)
    display = format_chapters_markdown(
        title, chapters, answer, duration_sec=duration
    )
    payload = {
        "title": title,
        "chapters": [c.as_dict() for c in chapters],
        "answer": answer,
        "display": display,
        "model_id": model_id,
        "mock": mock,
        "device": device,
        "video_name": video_name,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(display)
        if chapters:
            print()
            for ch in chapters:
                print(
                    f"  seek {format_timestamp(ch.start_sec)} "
                    f"({ch.start_sec:.1f}s – {ch.end_sec:.1f}s) {ch.title}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
