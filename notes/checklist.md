# LectureBuddy — Implementation Checklist

Ordered **easiest → hardest**. Do **one task at a time**.

Legend: `[P]` = paper/algorithm component · `[L]` = LectureBuddy product extension · `[R]` = reproducibility / comparison

---

## Tier A — Scaffold and docs (no model)

1. [x] `[R]` Repo layout: `paper/`, `notes/`, `src/`, `data/`, `README.md`
2. [x] `[R]` Paper summary + component inventory (`notes/summary.md`)
3. [ ] `[R]` Expand README with setup, goals, paper comparison section, extension notes

---

## Tier B — React frontend shell (mock / no GPU)

4. [x] `[L]` Vite + React app under `src/frontend` (upload video, chat UI, task buttons)
5. [x] `[L]` Wire task prompts: Explain / Notes / Flashcards / Quiz / Chapter summary / Formulas / Find topic
6. [x] `[L]` Mock backend responses so UI is demoable without the model

---

## Tier C — Inference integration (use released weights)

7. [x] `[P]` Python inference service wrapping [VideoChat3-4B](https://huggingface.co/MCG-NJU/VideoChat3-4B) (offline video QA)
8. [x] `[L]` Connect React upload + chat to the inference API
9. [ ] `[L]` Implement lecture prompts on top of video QA (notes, flashcards, quizzes, formulas)
10. [ ] `[P]` Temporal grounding endpoint ("Find where recursion was introduced" → timestamps)
11. [ ] `[P]` Streaming / proactive loop (Silence / Standby / Response + adaptive resolution) via official proactive demo pattern

---

## Tier D — Lecture-specific extension (meaningful beyond paper)

12. [ ] `[L]` Chapter segmentation UX (scene/time chapters + per-chapter summary)
13. [ ] `[L]` Structured study outputs (JSON schemas for flashcards / quiz grading)
14. [ ] `[L]` Persist sessions under `data/` (video metadata + chat history)

---

## Tier E — Paper fidelity / training (hard; optional until training code lands)

15. [ ] `[P]` Reproduce I3D-ViT module from paper (chunk T=4, temporal PE, pool, 2x2 merge) — hard without training code
16. [ ] `[P]` Adaptive Frame Resolution controller exactly as §2.3
17. [ ] `[P]` Data pipelines: Academic2M rewrite/judge, LV116K segment ledger, OL617K streaming construction
18. [ ] `[P]` Four-stage training + Stage-3 state-transition masking
19. [ ] `[R]` Side-by-side paper comparison doc (what matches / differs / missing)
20. [ ] `[R]` Full reproducibility (env lockfile, sample lecture video, eval script)

---

## Next recommended task

**Task 9:** Implement lecture prompts on top of video QA (notes, flashcards, quizzes, formulas).

---

## Final deliverables tracker

- [ ] GitHub repository (push when ready)
- [ ] Reproducible implementation (inference path first)
- [ ] Comparison with the paper (`notes/` + README section)
- [ ] One meaningful extension (lecture study companion UI + study tools)
- [ ] Well-written README
