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
