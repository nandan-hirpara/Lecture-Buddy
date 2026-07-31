# LectureBuddy

A lecture companion built on top of **VideoChat3** ([paper](https://arxiv.org/abs/2607.14935), [GitHub](https://github.com/MCG-NJU/VideoChat3), [HF model](https://huggingface.co/MCG-NJU/VideoChat3-4B)).

Upload a lecture video, then ask:

- Explain this topic
- Generate notes
- Generate flashcards
- Create quizzes
- Summarize each chapter
- List important formulas
- Find where a concept was introduced (temporal grounding)
- Live proactive watch (Silence / Standby / Response)

## Quick start

### Frontend (React)

```bash
cd src/frontend
npm install
npm run dev
```

Open http://localhost:5173

With the inference API running on `:8000`, the UI polls `/health` and sends uploaded videos to `POST /v1/ask`. If the API is down, study actions fall back to local mock replies.

### Inference API (VideoChat3-4B)

```powershell
cd src/inference
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Dry-run (no weight download)
$env:LECTUREBUDDY_MOCK="1"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Real model (your machine has an RTX 3060 6GB — use 4-bit):

```powershell
pip install bitsandbytes
$env:LECTUREBUDDY_MOCK="0"
$env:LECTUREBUDDY_LOAD_MODE="4bit"
$env:LECTUREBUDDY_MAX_FRAMES="8" 
$env:LECTUREBUDDY_VIDEO_FPS="0.5" 
# Optional if you still hit CUDA OOM on long/high-res lectures:
# $env:LECTUREBUDDY_MAX_FRAMES="4"
# $env:LECTUREBUDDY_VIDEO_FPS="0.25"
# Keep max_pixels >= 100352 (qwen video minimum); lowering it further errors.
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Details: `src/inference/README.md`

| Endpoint | Purpose |
|---|---|
| `GET /health` | Ready / mock / device |
| `POST /v1/ask` | Multipart video + question |
| `POST /v1/ask_path` | Local path + question (JSON) |
| `POST /v1/ground` | Temporal grounding → timestamps |
| `POST /v1/proactive` | Proactive Silence/Standby/Response stream |

## Project layout

```
LectureBuddy/
├── paper/
│   └── VideoChat3.pdf
├── notes/
│   ├── summary.md
│   └── checklist.md
├── src/
│   ├── frontend/       # React + Vite UI
│   └── inference/      # FastAPI + VideoChat3-4B wrapper
├── data/               # uploads / sample videos
└── README.md
```

## Strategy

1. **Do not retrain VideoChat3 from scratch first.** Official weights are public; training code is not released yet.
2. **Use VideoChat3-4B for video understanding** (QA, grounding, optional streaming).
3. **Extension:** React study companion + structured lecture outputs (notes, flashcards, quizzes).
4. **One task at a time** — see `notes/checklist.md`.

## Paper vs this project

| Paper component | In LectureBuddy |
|---|---|
| I3D-ViT + Adaptive Frame Resolution | Via released VideoChat3-4B weights (not reimplemented yet) |
| Academic2M / LV116K / OL617K | Optional later; not required for inference MVP |
| 4-stage training + state-transition mask | Blocked until training code release; documented in notes |
| Lecture study UX | Meaningful extension beyond the paper |

Full inventory: `notes/summary.md`.

## Status

- [x] Scaffold + paper notes
- [x] React frontend shell with mock study actions
- [x] VideoChat3 inference API (`src/inference`)
- [x] Wire React ↔ inference API
- [x] Lecture study prompts (notes / flashcards / quiz / formulas / …)
- [x] Temporal grounding endpoint (`/v1/ground` → timestamps + seek UI)
- [x] Streaming / proactive loop (`/v1/proactive` + Live button)
- [ ] Chapter segmentation UX (next)

## References

- Paper: https://arxiv.org/abs/2607.14935
- Papers with Code: https://paperswithcode.co/paper/2607.14935
- Hugging Face paper: https://huggingface.co/papers/2607.14935
- Homepage: https://mcg-nju.github.io/VideoChat3/
