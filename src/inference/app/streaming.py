"""Proactive streaming loop (Silence / Standby / Response + adaptive resolution).

Adapted from the official HF ``demo_vc3_proactive.py`` + ``inference_fast_vc3.py``
pattern for VideoChat3-4B.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

logger = logging.getLogger(__name__)

# Paper §2.3 / official demo defaults
B_LOW = 224 * 224
B_HIGH = 448 * 448

SYSTEM = """
You are a helpful assistant specializing in streaming video analysis.
You will receive input frame by frame, each labeled with absolute time intervals
in the exact format <Xs-Ys> (e.g., <0s-1s>). Follow these rules precisely:

1. Use </Silence> when:
   - No relevant event has started, OR
   - The current input is irrelevant to the given question.

2. Use </Standby> when:
   - An event is in progress but has not yet completed, OR
   - The current input is relevant but the question cannot yet be answered.

3. Use </Response> only when:
   - An event has fully concluded, OR
   - The available information is sufficient to fully answer the question.
   Provide a complete description at this point.

Do not provide partial answers or speculate beyond the given information.
Whenever you deliver an answer, begin with </Response>.
""".strip()

_END_TOKENS = ("<|im_end|>", "<|endoftext|>")
_STATE_RE = re.compile(r"</(Silence|Standby|Response)>")


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 28 * 28,
    max_pixels: int = B_LOW,
    force_resize: bool = False,
) -> tuple[int, int]:
    """Isotropic resize so H*W fits the pixel quota (official demo helper)."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError("absolute aspect ratio must be smaller than 200")
    if force_resize:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
        return h_bar, w_bar
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def resize_frame(frame: Any, max_pixels: int) -> Any:
    from PIL import Image

    if not isinstance(frame, Image.Image):
        frame = Image.fromarray(frame)
    w, h = frame.size
    h_bar, w_bar = smart_resize(h, w, min_pixels=28 * 28, max_pixels=max_pixels, force_resize=True)
    return frame.resize((w_bar, h_bar))


def strip_end_tokens(text: str) -> str:
    for end_tok in _END_TOKENS:
        while text.rstrip().endswith(end_tok):
            text = text.rstrip()[: -len(end_tok)]
    return text.strip()


def parse_state(answer: str) -> str:
    match = _STATE_RE.search(answer or "")
    if not match:
        return "unknown"
    return match.group(1).lower()


@dataclass
class ProactiveRound:
    round_idx: int
    time_start: float
    time_end: float
    state: str
    high_res: bool
    max_pixels: int
    raw: str
    answer_text: str = ""


@dataclass
class ProactiveResult:
    question: str
    rounds: list[ProactiveRound] = field(default_factory=list)
    final_answer: str | None = None
    mock: bool = False
    start_sec: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "rounds": [asdict(r) for r in self.rounds],
            "final_answer": self.final_answer,
            "mock": self.mock,
            "start_sec": self.start_sec,
        }


class VideoFrameExtractor:
    """Extract frames at target fps (official decode-order pattern)."""

    def __init__(
        self,
        video_path: str | Path,
        target_fps: float = 1.0,
        start_sec: float = 0.0,
    ):
        import cv2

        self.video_path = str(video_path)
        self.target_fps = float(target_fps)
        self.start_sec = max(0.0, float(start_sec or 0.0))
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")

        self.original_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.duration = (
            self.total_frames / self.original_fps if self.original_fps > 0 else 0.0
        )
        if self.start_sec > 0 and self.duration > 0 and self.start_sec >= self.duration:
            raise ValueError(
                f"start_sec ({self.start_sec:g}s) is past video end ({self.duration:g}s)"
            )
        self.frame_interval = max(int(round(self.original_fps / self.target_fps)), 1) if self.original_fps > 0 else 1
        self.actual_fps = (
            self.original_fps / self.frame_interval if self.original_fps > 0 else self.target_fps
        )
        self.start_frame = (
            int(round(self.start_sec * self.original_fps)) if self.original_fps > 0 else 0
        )
        self.start_frame = max(0, min(self.start_frame, max(0, self.total_frames - 1)))
        remaining = max(0, self.total_frames - self.start_frame)
        self.num_extracted_frames = (
            (remaining + self.frame_interval - 1) // self.frame_interval
            if remaining > 0
            else 0
        )
        if self.start_frame > 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        self._iter: Iterator[Any] = self._sequential_iter()

    def _sequential_iter(self):
        import cv2
        from PIL import Image

        local_idx, extracted = 0, 0
        while extracted < self.num_extracted_frames:
            ret, frame = self.cap.read()
            if not ret:
                break
            if local_idx % self.frame_interval == 0:
                yield Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                extracted += 1
            local_idx += 1

    def get_frame_at_round(self, round_num: int) -> Any:
        img = next(self._iter, None)
        if img is None:
            raise StopIteration(f"No more frames at round {round_num}")
        return img

    def get_total_rounds(self) -> int:
        return self.num_extracted_frames

    def close(self) -> None:
        if getattr(self, "cap", None) is not None:
            self.cap.release()
            self.cap = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass


class StreamingSession:
    """Stateful sliding-window chat for frame-by-frame proactive inference."""

    def __init__(
        self,
        infer_fn: Callable[..., str],
        *,
        system: str = SYSTEM,
        question: str | None = None,
        question_time: int = 0,
        max_rounds: int = 16,
        global_question: bool = True,
        max_tokens: int = 128,
    ):
        self.infer_fn = infer_fn
        self.system = system
        self.question = question
        self.question_time = question_time
        self.max_rounds = max_rounds
        self.global_question = global_question
        self.max_tokens = max_tokens

        self._messages: list[dict] = []
        self._last_answer: str | None = None
        self._window_start_round = 0

    @staticmethod
    def _text_only(text: str) -> list[dict]:
        return [{"type": "text", "text": text}]

    def _user_content(
        self,
        frame: Any,
        round_idx: int,
        include_question: bool,
        frame_max_pixels: int | None = None,
        time_start: float | None = None,
        time_end: float | None = None,
    ) -> list[dict]:
        if time_start is None:
            time_start = float(round_idx)
        if time_end is None:
            time_end = time_start + 1.0
        time_tag = f"<{time_start:g}s-{time_end:g}s>"
        text = time_tag
        if include_question and self.question:
            text = f"{self.question}\n{text}"
        frames = frame if isinstance(frame, (list, tuple)) else [frame]
        content: list[dict] = []
        for one_frame in frames:
            image_item: dict = {"type": "image", "image": one_frame}
            if frame_max_pixels is not None:
                image_item["min_pixels"] = 28 * 28
                image_item["max_pixels"] = frame_max_pixels
            content.append(image_item)
        content.append({"type": "text", "text": text})
        return content

    def _inject_question(self, msg: dict) -> None:
        if not self.question:
            return
        content = msg.get("content", [])
        for i, item in enumerate(content):
            if item.get("type") == "text" and self.question not in item.get("text", ""):
                new_item = dict(item)
                new_item["text"] = self.question + "\n" + new_item["text"]
                msg["content"] = list(content)
                msg["content"][i] = new_item
                break

    def _append_turn(
        self,
        frame: Any,
        round_idx: int,
        frame_max_pixels: int | None = None,
        time_start: float | None = None,
        time_end: float | None = None,
    ) -> None:
        if not self._messages:
            if self.system:
                self._messages.append({"role": "system", "content": self._text_only(self.system)})
            include_q = self.global_question or (round_idx == self.question_time)
            self._messages.append(
                {
                    "role": "user",
                    "content": self._user_content(
                        frame, round_idx, include_q, frame_max_pixels, time_start, time_end
                    ),
                }
            )
            self._window_start_round = round_idx
            return

        if self._last_answer is not None:
            self._messages.append({"role": "assistant", "content": self._text_only(self._last_answer)})

        include_q = round_idx == self.question_time
        self._messages.append(
            {
                "role": "user",
                "content": self._user_content(
                    frame, round_idx, include_q, frame_max_pixels, time_start, time_end
                ),
            }
        )

        user_count = sum(1 for m in self._messages if m.get("role") == "user")
        if user_count > self.max_rounds:
            rounds_to_remove = user_count - self.max_rounds
            sys_offset = 1 if self._messages and self._messages[0].get("role") == "system" else 0
            for _ in range(rounds_to_remove):
                if sys_offset < len(self._messages) and self._messages[sys_offset].get("role") == "user":
                    del self._messages[sys_offset]
                if (
                    sys_offset < len(self._messages)
                    and self._messages[sys_offset].get("role") == "assistant"
                ):
                    del self._messages[sys_offset]
            self._window_start_round += rounds_to_remove
            new_first_idx = next(
                (i for i, m in enumerate(self._messages) if m.get("role") == "user"), None
            )
            if new_first_idx is not None:
                include_q_start = self.global_question or (
                    self.question_time == self._window_start_round
                )
                if include_q_start:
                    self._inject_question(self._messages[new_first_idx])

    def step(
        self,
        frame: Any,
        round_idx: int,
        frame_max_pixels: int | None = None,
        time_start: float | None = None,
        time_end: float | None = None,
    ) -> str:
        self._append_turn(
            frame,
            round_idx,
            frame_max_pixels=frame_max_pixels,
            time_start=time_start,
            time_end=time_end,
        )
        if round_idx < self.question_time:
            answer = "</Silence>"
        else:
            answer = self.infer_fn(self._messages, max_tokens=self.max_tokens)
        self._last_answer = answer
        return answer


def mock_proactive(
    question: str,
    duration_hint: float = 8.0,
    start_sec: float = 0.0,
) -> ProactiveResult:
    """Deterministic Silence → Standby → Response timeline for mock mode."""
    offset = max(0.0, float(start_sec or 0.0))
    rounds = [
        ProactiveRound(0, offset + 0.0, offset + 2.0, "silence", False, B_LOW, "</Silence>"),
        ProactiveRound(1, offset + 2.0, offset + 4.0, "silence", False, B_LOW, "</Silence>"),
        ProactiveRound(2, offset + 4.0, offset + 6.0, "standby", False, B_LOW, "</Standby>"),
        ProactiveRound(
            3,
            offset + 6.0,
            offset + min(8.0, duration_hint),
            "response",
            True,
            B_HIGH,
            f"</Response> [mock] Enough evidence to answer: {question}",
            answer_text=f"[mock] Enough evidence to answer: {question}",
        ),
    ]
    return ProactiveResult(
        question=question,
        rounds=rounds,
        final_answer=rounds[-1].answer_text,
        mock=True,
        start_sec=offset,
    )


def format_proactive_markdown(result: ProactiveResult) -> str:
    start = getattr(result, "start_sec", 0.0) or 0.0
    header = f'**Proactive stream** for “{result.question}”'
    if start > 0:
        header += f" _(from {start:g}s)_"
    lines = [header, ""]
    for r in result.rounds:
        tag = " HIGH-RES" if r.high_res else ""
        span = f"{r.time_start:g}s–{r.time_end:g}s"
        state = r.state.upper()
        body = r.answer_text or r.raw
        lines.append(f"- **{span}** [{state}]{tag}: {body}")
    if result.final_answer:
        lines.extend(["", f"**Final answer:** {result.final_answer}"])
    return "\n".join(lines)


def run_proactive_loop(
    *,
    video_path: str | Path,
    question: str,
    infer_fn: Callable[..., str],
    target_fps: float = 0.5,
    max_rounds: int = 8,
    max_seconds: float | None = 45.0,
    max_new_tokens: int = 128,
    start_sec: float = 0.0,
    low_pixels: int = B_LOW,
    high_pixels: int = B_HIGH,
) -> ProactiveResult:
    """
    Official adaptive-resolution controller:

    - default windows at ``low_pixels`` (224²)
    - after ``</Standby>``, next window uses ``high_pixels`` (448²)
    - after ``</Response>``, return to low budget
    """
    q = (question or "").strip()
    if not q:
        raise ValueError("question must be non-empty")

    offset = max(0.0, float(start_sec or 0.0))
    extractor = VideoFrameExtractor(video_path, target_fps=target_fps, start_sec=offset)
    session = StreamingSession(
        infer_fn,
        system=SYSTEM,
        question=q,
        question_time=0,
        max_rounds=max_rounds,
        max_tokens=max_new_tokens,
    )

    actual_fps = extractor.actual_fps if extractor.actual_fps > 0 else target_fps
    # Official demo groups ~1s of frames per round when target_fps≈video_fps sampling.
    frames_per_round = max(1, round(actual_fps / max(target_fps, 1e-6)))
    # With our extractor already at target_fps, one extracted frame ≈ one step.
    frames_per_round = 1

    total = extractor.get_total_rounds()
    if max_seconds is not None and actual_fps > 0:
        total = min(total, max(1, int(max_seconds * actual_fps)))
    total = min(total, max_rounds)

    standby_remaining = 0
    rounds: list[ProactiveRound] = []
    final_answer: str | None = None

    try:
        for round_idx in range(total):
            raw = extractor.get_frame_at_round(round_idx)
            high_res = standby_remaining > 0
            if standby_remaining > 0:
                standby_remaining -= 1
            max_px = high_pixels if high_res else low_pixels
            frames = [resize_frame(raw, max_px)]

            rel_start = round_idx / actual_fps if actual_fps > 0 else float(round_idx)
            rel_end = (round_idx + 1) / actual_fps if actual_fps > 0 else rel_start + 1.0
            time_start = offset + rel_start
            time_end = offset + rel_end


            answer = session.step(
                frames,
                round_idx=round_idx,
                frame_max_pixels=max_px,
                time_start=time_start,
                time_end=time_end,
            )
            state = parse_state(answer)
            if state == "standby":
                standby_remaining = 1

            answer_text = _STATE_RE.sub("", answer).strip()
            if state == "response" and answer_text:
                final_answer = answer_text

            rounds.append(
                ProactiveRound(
                    round_idx=round_idx,
                    time_start=time_start,
                    time_end=time_end,
                    state=state,
                    high_res=high_res,
                    max_pixels=max_px,
                    raw=answer,
                    answer_text=answer_text,
                )
            )
            logger.info(
                "proactive round=%s t=%.2f-%.2f state=%s high_res=%s",
                round_idx,
                time_start,
                time_end,
                state,
                high_res,
            )
            if state == "response":
                break
    finally:
        extractor.close()

    return ProactiveResult(
        question=q,
        rounds=rounds,
        final_answer=final_answer,
        mock=False,
        start_sec=offset,
    )
