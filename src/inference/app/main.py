"""FastAPI service wrapping VideoChat3-4B for offline video QA."""

from __future__ import annotations

import logging
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .chapters import (
    build_chapters_prompt,
    format_chapters_markdown,
    mock_chapters,
    parse_chapters_answer,
)
from .config import settings
from .grounding import (
    build_grounding_prompt,
    format_grounding_markdown,
    mock_grounding,
    parse_grounding_answer,
)
from .model import engine
from .prompts import build_lecture_prompt
from .schemas import (
    AskPathRequest,
    AskResponse,
    ChapterModel,
    ChaptersPathRequest,
    ChaptersResponse,
    GroundPathRequest,
    GroundResponse,
    GroundSegmentModel,
    HealthResponse,
    ProactivePathRequest,
    ProactiveResponse,
    ProactiveRoundModel,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("lecturebuddy.inference")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    if settings.preload:
        try:
            engine.load()
        except Exception as exc:  # noqa: BLE001
            # Keep the API up so /health can report the failure.
            logger.error("Preload failed: %s", exc)
    yield


app = FastAPI(
    title="LectureBuddy Inference",
    description="Offline video QA via MCG-NJU/VideoChat3-4B",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    st = engine.status
    if st.get("error"):
        status = "error"
    elif st["ready"]:
        status = "ok"
    else:
        status = "starting"
    return HealthResponse(
        status=status,
        ready=bool(st["ready"]),
        mock=bool(st["mock"]),
        model_id=str(st["model_id"]),
        load_mode=str(st["load_mode"]),
        device=st.get("device"),
        loaded=bool(st["loaded"]),
        error=st.get("error"),
    )


@app.post("/v1/ask", response_model=AskResponse)
async def ask_upload(
    question: str = Form(...),
    video: UploadFile = File(...),
    task: str | None = Form(default=None),
    max_new_tokens: int | None = Form(default=None),
) -> AskResponse:
    if not question.strip():
        raise HTTPException(status_code=400, detail="question must be non-empty")

    try:
        prompt = build_lecture_prompt(task, question)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    suffix = Path(video.filename or "upload.mp4").suffix or ".mp4"
    dest = settings.upload_dir / f"{uuid.uuid4().hex}{suffix}"
    try:
        with dest.open("wb") as out:
            shutil.copyfileobj(video.file, out)
        result = engine.ask(dest, prompt, max_new_tokens=max_new_tokens)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        try:
            dest.unlink(missing_ok=True)
        except OSError:
            pass

    return AskResponse(
        answer=result.answer,
        model_id=result.model_id,
        mock=result.mock,
        device=result.device,
        video_name=video.filename,
    )


@app.post("/v1/ask_path", response_model=AskResponse)
def ask_path(body: AskPathRequest) -> AskResponse:
    try:
        prompt = build_lecture_prompt(body.task, body.question)
        result = engine.ask(body.video_path, prompt, max_new_tokens=body.max_new_tokens)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return AskResponse(
        answer=result.answer,
        model_id=result.model_id,
        mock=result.mock,
        device=result.device,
        video_name=Path(body.video_path).name,
    )


def _run_grounding(video_path: str | Path, query: str, max_new_tokens: int | None) -> GroundResponse:
    q = query.strip()
    if not q:
        raise ValueError("query must be non-empty")

    if settings.mock:
        found, segments, answer = mock_grounding(q)
        display = format_grounding_markdown(q, found, segments, answer)
        return GroundResponse(
            query=q,
            found=found,
            segments=[GroundSegmentModel(**s.as_dict()) for s in segments],
            answer=answer,
            display=display,
            model_id=settings.model_id,
            mock=True,
            device="mock",
            video_name=Path(video_path).name,
        )

    prompt = build_grounding_prompt(q)
    # Grounding answers are usually short JSON + a paragraph.
    tokens = max_new_tokens or min(settings.max_new_tokens, 384)
    result = engine.ask(video_path, prompt, max_new_tokens=tokens)
    found, segments = parse_grounding_answer(result.answer)
    display = format_grounding_markdown(q, found, segments, result.answer)
    return GroundResponse(
        query=q,
        found=found,
        segments=[GroundSegmentModel(**s.as_dict()) for s in segments],
        answer=result.answer,
        display=display,
        model_id=result.model_id,
        mock=result.mock,
        device=result.device,
        video_name=Path(video_path).name,
    )


@app.post("/v1/ground", response_model=GroundResponse)
async def ground_upload(
    query: str = Form(...),
    video: UploadFile = File(...),
    max_new_tokens: int | None = Form(default=None),
) -> GroundResponse:
    if not query.strip():
        raise HTTPException(status_code=400, detail="query must be non-empty")

    suffix = Path(video.filename or "upload.mp4").suffix or ".mp4"
    dest = settings.upload_dir / f"{uuid.uuid4().hex}{suffix}"
    try:
        with dest.open("wb") as out:
            shutil.copyfileobj(video.file, out)
        return _run_grounding(dest, query, max_new_tokens)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Grounding failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        try:
            dest.unlink(missing_ok=True)
        except OSError:
            pass


@app.post("/v1/ground_path", response_model=GroundResponse)
def ground_path(body: GroundPathRequest) -> GroundResponse:
    try:
        return _run_grounding(body.video_path, body.query, body.max_new_tokens)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Grounding failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _run_chapters(video_path: str | Path, max_new_tokens: int | None) -> ChaptersResponse:
    from .chapters import (
        apply_time_slots,
        build_time_slots,
        coverage_ratio,
        normalize_chapters,
        probe_video_duration,
    )

    video_name = Path(video_path).name
    duration = probe_video_duration(video_path)
    slots = build_time_slots(duration)

    if settings.mock:
        title, chapters, answer = mock_chapters(video_name)
        chapters = apply_time_slots(chapters, slots) if duration else normalize_chapters(
            chapters, duration
        )
        display = format_chapters_markdown(
            title, chapters, answer, duration_sec=duration
        )
        return ChaptersResponse(
            title=title,
            chapters=[ChapterModel(**c.as_dict()) for c in chapters],
            answer=answer,
            display=display,
            model_id=settings.model_id,
            mock=True,
            device="mock",
            video_name=video_name,
        )

    prompt = build_chapters_prompt(duration, slots=slots)
    # Compact title/summary JSON for ~5 slots; keep runtime closer to normal asks.
    tokens = max_new_tokens or max(settings.max_new_tokens, 512)
    frames = settings.max_frames

    result = engine.ask(
        video_path,
        prompt,
        max_new_tokens=tokens,
        max_frames=frames,
    )
    title, chapters = parse_chapters_answer(result.answer)
    # Always bind labels onto the full-duration scaffold so seek rail covers 0→end.
    chapters = apply_time_slots(chapters, slots)
    cov = coverage_ratio(chapters, duration)
    logger.info(
        "chapters parsed=%s slots=%s coverage=%.0f%% duration=%s tokens=%s frames=%s",
        len(chapters),
        len(slots),
        cov * 100,
        f"{duration:.0f}s" if duration else "?",
        tokens,
        frames,
    )
    display = format_chapters_markdown(
        title, chapters, result.answer, duration_sec=duration
    )
    return ChaptersResponse(
        title=title,
        chapters=[ChapterModel(**c.as_dict()) for c in chapters],
        answer=result.answer,
        display=display,
        model_id=result.model_id,
        mock=result.mock,
        device=result.device,
        video_name=video_name,
    )


@app.post("/v1/chapters", response_model=ChaptersResponse)
async def chapters_upload(
    video: UploadFile = File(...),
    max_new_tokens: int | None = Form(default=None),
) -> ChaptersResponse:
    suffix = Path(video.filename or "upload.mp4").suffix or ".mp4"
    dest = settings.upload_dir / f"{uuid.uuid4().hex}{suffix}"
    try:
        with dest.open("wb") as out:
            shutil.copyfileobj(video.file, out)
        return _run_chapters(dest, max_new_tokens)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Chapter segmentation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        try:
            dest.unlink(missing_ok=True)
        except OSError:
            pass


@app.post("/v1/chapters_path", response_model=ChaptersResponse)
def chapters_path(body: ChaptersPathRequest) -> ChaptersResponse:
    try:
        return _run_chapters(body.video_path, body.max_new_tokens)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Chapter segmentation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _run_proactive(
    video_path: str | Path,
    question: str,
    *,
    target_fps: float | None = None,
    max_rounds: int | None = None,
    max_seconds: float | None = None,
    max_new_tokens: int | None = None,
    start_sec: float | None = None,
) -> ProactiveResponse:
    from .streaming import format_proactive_markdown

    result = engine.proactive(
        video_path,
        question,
        target_fps=target_fps,
        max_rounds=max_rounds,
        max_seconds=max_seconds,
        max_new_tokens=max_new_tokens,
        start_sec=start_sec,
    )
    rounds = [
        ProactiveRoundModel(
            round_idx=r.round_idx,
            time_start=r.time_start,
            time_end=r.time_end,
            state=r.state,
            high_res=r.high_res,
            max_pixels=r.max_pixels,
            raw=r.raw,
            answer_text=r.answer_text,
        )
        for r in result.rounds
    ]
    return ProactiveResponse(
        question=result.question,
        rounds=rounds,
        final_answer=result.final_answer,
        display=format_proactive_markdown(result),
        model_id=settings.model_id,
        mock=result.mock,
        device=engine.status.get("device"),
        video_name=Path(video_path).name,
        start_sec=result.start_sec,
    )


@app.post("/v1/proactive", response_model=ProactiveResponse)
async def proactive_upload(
    question: str = Form(...),
    video: UploadFile = File(...),
    target_fps: float | None = Form(default=None),
    max_rounds: int | None = Form(default=None),
    max_seconds: float | None = Form(default=None),
    max_new_tokens: int | None = Form(default=None),
    start_sec: float | None = Form(default=None),
) -> ProactiveResponse:
    if not question.strip():
        raise HTTPException(status_code=400, detail="question must be non-empty")

    suffix = Path(video.filename or "upload.mp4").suffix or ".mp4"
    dest = settings.upload_dir / f"{uuid.uuid4().hex}{suffix}"
    try:
        with dest.open("wb") as out:
            shutil.copyfileobj(video.file, out)
        return _run_proactive(
            dest,
            question,
            target_fps=target_fps,
            max_rounds=max_rounds,
            max_seconds=max_seconds,
            max_new_tokens=max_new_tokens,
            start_sec=start_sec,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Proactive streaming failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        try:
            dest.unlink(missing_ok=True)
        except OSError:
            pass


@app.post("/v1/proactive_path", response_model=ProactiveResponse)
def proactive_path(body: ProactivePathRequest) -> ProactiveResponse:
    try:
        return _run_proactive(
            body.video_path,
            body.question,
            target_fps=body.target_fps,
            max_rounds=body.max_rounds,
            max_seconds=body.max_seconds,
            max_new_tokens=body.max_new_tokens,
            start_sec=body.start_sec,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Proactive streaming failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
