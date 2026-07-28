"""FastAPI service wrapping VideoChat3-4B for offline video QA."""

from __future__ import annotations

import logging
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .model import engine
from .prompts import build_lecture_prompt
from .schemas import AskPathRequest, AskResponse, HealthResponse

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
