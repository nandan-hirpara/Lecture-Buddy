# LectureBuddy VideoChat3 inference service

Offline video QA API wrapping [`MCG-NJU/VideoChat3-4B`](https://huggingface.co/MCG-NJU/VideoChat3-4B).

## Setup

```bash
cd src/inference
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

On an RTX 3060 Laptop (6GB), make sure PyTorch sees CUDA, then install 4-bit support:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# If cuda is False, reinstall GPU wheels:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install bitsandbytes
```

> The HF load report line `lm_head.weight | MISSING` is usually harmless — the head is tied to the input embeddings.

## Run (mock — no GPU / no weight download)

```powershell
$env:LECTUREBUDDY_MOCK="1"
uvicorn app.main:app --app-dir . --host 127.0.0.1 --port 8000
```

## Run (real model)

```powershell
$env:LECTUREBUDDY_MOCK="0"
$env:LECTUREBUDDY_LOAD_MODE="4bit"   # recommended on 6GB VRAM
uvicorn app.main:app --app-dir . --host 127.0.0.1 --port 8000
```

First real start downloads the HF weights (several GB).

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Readiness / device / mock flag |
| POST | `/v1/ask` | Multipart: `video` + `question` (+ optional `task`, `max_new_tokens`) |
| POST | `/v1/ask_path` | JSON: `{ "video_path", "question", "task?", "max_new_tokens?" }` |
| POST | `/v1/ground` | Multipart temporal grounding: `video` + `query` → timestamps |
| POST | `/v1/ground_path` | JSON grounding: `{ "video_path", "query", "max_new_tokens?" }` |
| POST | `/v1/proactive` | Multipart proactive stream: `video` + `question` → Silence/Standby/Response rounds |
| POST | `/v1/proactive_path` | JSON proactive: `{ "video_path", "question", "target_fps?", "max_rounds?", "max_seconds?", "start_sec?" }` |
| POST | `/v1/chapters` | Multipart chapter segmentation: `video` → timed chapters + summaries |
| POST | `/v1/chapters_path` | JSON chapters: `{ "video_path", "max_new_tokens?" }` |

Optional `task` values: `explain`, `notes`, `flashcards`, `quiz`, `chapters`, `formulas`, `find`.  
These expand into lecture study prompts in `app/prompts.py` before VideoChat3 runs. Free-form chat omits `task`.

`/v1/ground` asks VideoChat3 for JSON timestamp segments, parses them in `app/grounding.py`, and returns jump points for the UI.

`/v1/proactive` runs the official Silence / Standby / Response loop (`app/streaming.py`): low-res windows by default, high-res (448²) after `</Standby>`, stop on `</Response>`. Caps for 6GB: `LECTUREBUDDY_PROACTIVE_FPS`, `LECTUREBUDDY_PROACTIVE_MAX_ROUNDS`, `LECTUREBUDDY_PROACTIVE_MAX_SECONDS`. Optional `start_sec` skips the intro so Live can begin mid-lecture.

`/v1/chapters` asks VideoChat3 for a JSON chapter outline (`app/chapters.py`), parses timestamps + summaries, and powers the under-player chapter rail in the UI.

### Example

```bash
curl -s http://127.0.0.1:8000/health

curl -s -X POST http://127.0.0.1:8000/v1/ask ^
  -F "question=Generate study notes." ^
  -F "task=notes" ^
  -F "video=@..\..\data\sample.mp4"

curl -s -X POST http://127.0.0.1:8000/v1/ground ^
  -F "query=recursion" ^
  -F "video=@..\..\data\sample.mp4"

curl -s -X POST http://127.0.0.1:8000/v1/proactive ^
  -F "question=Watch for the key concept and answer when ready." ^
  -F "video=@..\..\data\sample.mp4"

curl -s -X POST http://127.0.0.1:8000/v1/chapters ^
  -F "video=@..\..\data\sample.mp4"
```

### CLI

```powershell
$env:LECTUREBUDDY_MOCK="1"
python scripts/ask_cli.py ..\..\data\sample.mp4 "Summarize this lecture."
python scripts/ask_cli.py ..\..\data\sample.mp4 "recursion" --task find
python scripts/ask_cli.py ..\..\data\sample.mp4 "notes" --task notes
python scripts/ground_cli.py ..\..\data\sample.mp4 "recursion"
python scripts/proactive_cli.py ..\..\data\sample.mp4 "Watch for the key concept."
python scripts/proactive_cli.py ..\..\data\sample.mp4 "Watch for the key concept." --start-sec 180
python scripts/chapters_cli.py ..\..\data\sample.mp4
```

## Notes

- Follows the official HF `demo_vc3.py` preprocessing (`process_vision_info` + `video_metadata`).
- Proactive mode follows `demo_vc3_proactive.py` / `inference_fast_vc3.py` (image frames + state tokens + adaptive pixels).
- Chapter mode asks for structured JSON chapter outlines (`app/chapters.py`) and falls back to markdown headings.
- CORS allows the Vite frontend (`localhost:5173`) for Task 8.
- Uploaded files are written under `data/uploads/` and deleted after each request.
