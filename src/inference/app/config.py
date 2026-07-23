"""Configuration for the LectureBuddy VideoChat3 inference service."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    model_id: str = os.getenv("LECTUREBUDDY_MODEL_ID", "MCG-NJU/VideoChat3-4B")
    host: str = os.getenv("LECTUREBUDDY_HOST", "127.0.0.1")
    port: int = int(os.getenv("LECTUREBUDDY_PORT", "8000"))
    mock: bool = _env_bool("LECTUREBUDDY_MOCK", False)
    # auto | 4bit | 8bit  — 4bit recommended on ~6GB GPUs
    load_mode: str = os.getenv("LECTUREBUDDY_LOAD_MODE", "auto").strip().lower()
    max_new_tokens: int = int(os.getenv("LECTUREBUDDY_MAX_NEW_TOKENS", "256"))
    # Soft cap for qwen_vl_utils video sampling (frames / pixels).
    # Defaults are conservative for ~6GB GPUs; raise if you have more VRAM.
    # max_pixels must be >= qwen video min (~100352 at patch 14).
    video_fps: float = float(os.getenv("LECTUREBUDDY_VIDEO_FPS", "0.5"))
    max_frames: int = int(os.getenv("LECTUREBUDDY_MAX_FRAMES", "8"))
    max_pixels: int = int(os.getenv("LECTUREBUDDY_MAX_PIXELS", "100352"))
    upload_dir: Path = Path(
        os.getenv(
            "LECTUREBUDDY_UPLOAD_DIR",
            str(Path(__file__).resolve().parents[3] / "data" / "uploads"),
        )
    )
    preload: bool = _env_bool("LECTUREBUDDY_PRELOAD", True)


settings = Settings()
