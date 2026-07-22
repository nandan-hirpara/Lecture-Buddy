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

On an RTX 3060 Laptop (6GB), also install 4-bit support:

```bash
pip install bitsandbytes
```

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
| POST | `/v1/ask` | Multipart: `video` + `question` (+ optional `max_new_tokens`) |
| POST | `/v1/ask_path` | JSON: `{ "video_path", "question", "max_new_tokens?" }` |

### Example

```bash
curl -s http://127.0.0.1:8000/health

curl -s -X POST http://127.0.0.1:8000/v1/ask ^
  -F "question=What is the main topic of this lecture?" ^
  -F "video=@..\..\data\sample.mp4"
```

### CLI

```powershell
$env:LECTUREBUDDY_MOCK="1"
python scripts/ask_cli.py ..\..\data\sample.mp4 "Summarize this lecture."
```

## Notes

- Follows the official HF `demo_vc3.py` preprocessing (`process_vision_info` + `video_metadata`).
- CORS allows the Vite frontend (`localhost:5173`) for Task 8.
- Uploaded files are written under `data/uploads/` and deleted after each request.
