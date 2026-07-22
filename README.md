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

## Quick start (frontend shell)

```bash
cd src/frontend
npm install
npm run dev
```

Open http://localhost:5173 — upload a video and try study actions. Replies are **mocked** until the inference service is wired.

## Project layout

```
LectureBuddy/
├── paper/
│   └── VideoChat3.pdf
├── notes/
│   ├── summary.md      # algorithm + component inventory
│   └── checklist.md    # implement one task at a time (easy → hard)
├── src/
│   └── frontend/       # React + Vite UI (Task 3 done)
├── data/               # sample videos / session artifacts
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
- [ ] VideoChat3 inference API (next)

## References

- Paper: https://arxiv.org/abs/2607.14935
- Papers with Code: https://paperswithcode.co/paper/2607.14935
- Hugging Face paper: https://huggingface.co/papers/2607.14935
- Homepage: https://mcg-nju.github.io/VideoChat3/
