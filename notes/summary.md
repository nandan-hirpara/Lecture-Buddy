# VideoChat3 to LectureBuddy: Paper Summary and Implementation Map

**Paper:** [VideoChat3](https://arxiv.org/abs/2607.14935) (arXiv:2607.14935)
**Code:** [MCG-NJU/VideoChat3](https://github.com/MCG-NJU/VideoChat3)
**Weights:** [MCG-NJU/VideoChat3-4B](https://huggingface.co/MCG-NJU/VideoChat3-4B)
**Data:** Academic2M, LV116K, OL617K on Hugging Face
**Status:** weights + data released; training code not yet released (GitHub TODO)

---

## 1. Algorithm summary

VideoChat3 is a 4B Video MLLM (ViT -> MLP projector -> LLM) designed for efficient, generalist video understanding across:

- Fine-grained motion / temporal perception
- Long-video reasoning
- Temporal grounding (localize when something happens)
- Streaming / proactive response (decide when to answer)

### Core idea

1. Compress video early in the vision encoder (I3D-ViT) so the LLM sees about 16x fewer visual tokens.
2. Adapt spatial resolution over time (Adaptive Frame Resolution) so routine moments stay cheap and important moments get more pixels.
3. Train with curated curricula on three datasets plus a 4-stage schedule.

---

## 2. Architecture components (paper Section 2)

### 2.1 Inflated 3D Vision Transformer (I3D-ViT)

Initialized from image-pretrained MoonViT, then inflated to video:

1. Chunked frame grouping: contiguous chunks of up to T frames (default T=4).
2. Temporal positional encoding: keep spatial PEs; add learned temporal embeddings for indices 0..T-1.
3. Native-resolution spatiotemporal attention inside each chunk.
4. Chunk-wise temporal pooling: T-fold token reduction.
5. 2x2 spatial merge (pixel shuffle): another 4x reduction.

Total compression approx 4T = 16x at T=4.

### 2.2 Adaptive Frame Resolution (streaming)

Closed loop between LLM state tokens and the visual tokenizer:

| State | Meaning | Next-window pixel quota |
|---|---|---|
| Silence | No task-relevant evidence | B_low = 224^2 |
| Standby | Potential evidence; need more frames | B_high = 448^2 |
| Response | Enough evidence; generate answer | then back to B_low |

Frames are isotropically resized to fit the quota. I3D-ViT supports variable spatial sizes.

### 2.3 Full stack

Video frames -> I3D-ViT -> MLP Projector -> LLM (Qwen3-4B family) -> text / state tokens

---

## 3. Data construction components (paper Section 3)

### 3.1 VideoChat3-Academic2M (~2.27M)

1. Aggregate academic sources (LLaVA-Video, Spoken-MIT, Vript, etc.).
2. Normalize queries (strip option-letter-only formatting).
3. Evidence-grounded rewrite via Qwen3-VL-235B (preserve original answer semantics).
4. Consistency judge filters hallucinations.

### 3.2 VideoChat3-LV116K (~116K long-video rows)

1. Collect / filter long videos.
2. Boundary-aware segmentation (PySceneDetect + duration heuristics).
3. Segment captions + quality filter.
4. Assemble evidence ledger; synthesize temporal grounding, timelines, long-video QA.

### 3.3 VideoChat3-OL617K (~617K streaming)

1. Localize visual clue intervals for offline QA.
2. Verify clues by cropping intervals and re-asking.
3. Build interleaved streams labeled Silence / Standby / Response + answer.

---

## 4. Training components (paper Section 4)

| Stage | Purpose | Trainable | Notes |
|---|---|---|---|
| 0 | Visual tokenizer pre-training | Proj then All | Temp LLM discarded after; keep ViT |
| 1 | Video-language alignment | Proj then All | Caption-centric; final LLM |
| 2 | Video instruction tuning | All | ~50B tokens; Academic + image/text mix |
| 3 | Long and streaming | Proj and LLM | State-transition masking; adaptive pixel budget |

State-transition masking (Stage 3): keep loss on all state changes; sample an equal number of keep positions so the model does not collapse to always predicting Silence.

---

## 5. Official resources vs LectureBuddy

| Resource | Official status | LectureBuddy approach |
|---|---|---|
| Paper + PDF | Available | paper/VideoChat3.pdf |
| Model weights | VideoChat3-4B on HF | Prefer inference from HF over re-training |
| Inference demos | demo_vc3.py, demo_vc3_proactive.py | Wrap for lecture tasks |
| Datasets | Academic2M / LV / OL on HF | Optional for fine-tuning / reproducibility |
| Training code | Not released yet | Document gap; do not block product |
| Lecture UX | Not in paper | Meaningful extension: React companion |

---

## 6. LectureBuddy product mapping

| User feature | Paper capability used |
|---|---|
| Explain this topic | General video QA / reasoning |
| Generate notes / chapter summaries | Long-video captioning + segment-style prompting |
| Flashcards / quizzes | Instruction-following over lecture evidence |
| List important formulas | Fine-grained perception + OCR-style detail |
| Find where X was introduced | Temporal grounding |
| Live lecture help (future) | Streaming Silence/Standby/Response |

---

## 7. Paper comparison checklist (fill during Review)

- [ ] I3D-ViT chunking T=4 + temporal PE + temporal pool + 2x2 merge
- [ ] Adaptive resolution controller (224^2 / 448^2 tied to state tokens)
- [ ] Silence / Standby / Response streaming loop
- [ ] Stage 0-3 training recipe (or documented weights-only shortcut)
- [ ] Academic2M / LV116K / OL617K usage or documented omission
- [ ] State-transition masking (if training streaming)
- [ ] Extension: lecture study tools + React UI (not in paper)

---

## 8. One-task-at-a-time rule

Implement one checklist item, verify it works, then pick the next.
