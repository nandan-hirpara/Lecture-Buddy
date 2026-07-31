"""Pydantic schemas for the inference API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    ready: bool
    mock: bool
    model_id: str
    load_mode: str
    device: str | None = None
    loaded: bool
    error: str | None = None


class AskPathRequest(BaseModel):
    video_path: str = Field(..., description="Absolute or relative path to a local video file")
    question: str = Field(..., min_length=1)
    task: str | None = Field(
        default=None,
        description="Optional study task id: explain, notes, flashcards, quiz, chapters, formulas, find",
    )
    max_new_tokens: int | None = Field(default=None, ge=1, le=4096)


class AskResponse(BaseModel):
    answer: str
    model_id: str
    mock: bool
    device: str | None = None
    video_name: str | None = None


class GroundSegmentModel(BaseModel):
    start_sec: float = Field(..., ge=0)
    end_sec: float = Field(..., ge=0)
    label: str = ""
    evidence: str = ""


class GroundPathRequest(BaseModel):
    video_path: str
    query: str = Field(..., min_length=1, description="Topic or event to localize in the video")
    max_new_tokens: int | None = Field(default=None, ge=1, le=4096)


class GroundResponse(BaseModel):
    query: str
    found: bool
    segments: list[GroundSegmentModel]
    answer: str = Field(..., description="Raw model text")
    display: str = Field(..., description="Markdown summary for the chat UI")
    model_id: str
    mock: bool
    device: str | None = None
    video_name: str | None = None


class ProactiveRoundModel(BaseModel):
    round_idx: int
    time_start: float
    time_end: float
    state: str
    high_res: bool
    max_pixels: int
    raw: str
    answer_text: str = ""


class ProactivePathRequest(BaseModel):
    video_path: str
    question: str = Field(..., min_length=1)
    target_fps: float | None = Field(default=None, gt=0)
    max_rounds: int | None = Field(default=None, ge=1, le=64)
    max_seconds: float | None = Field(default=None, gt=0)
    max_new_tokens: int | None = Field(default=None, ge=1, le=1024)
    start_sec: float | None = Field(default=None, ge=0)


class ProactiveResponse(BaseModel):
    question: str
    rounds: list[ProactiveRoundModel]
    final_answer: str | None = None
    display: str
    model_id: str
    mock: bool
    device: str | None = None
    video_name: str | None = None
    start_sec: float = 0.0


class ChapterModel(BaseModel):
    index: int = Field(..., ge=1)
    start_sec: float = Field(..., ge=0)
    end_sec: float = Field(..., ge=0)
    title: str
    summary: str = ""


class ChaptersPathRequest(BaseModel):
    video_path: str
    max_new_tokens: int | None = Field(default=None, ge=1, le=4096)


class ChaptersResponse(BaseModel):
    title: str | None = None
    chapters: list[ChapterModel]
    answer: str = Field(..., description="Raw model text")
    display: str = Field(..., description="Markdown outline for the chat UI")
    model_id: str
    mock: bool
    device: str | None = None
    video_name: str | None = None
