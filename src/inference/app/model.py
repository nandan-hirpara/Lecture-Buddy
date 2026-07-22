"""VideoChat3-4B inference wrapper (official HF demo pattern)."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import settings

logger = logging.getLogger(__name__)


@dataclass
class AskResult:
    answer: str
    model_id: str
    mock: bool
    device: str | None = None


class VideoChat3Engine:
    """Lazy-loaded singleton around MCG-NJU/VideoChat3-4B."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model = None
        self._processor = None
        self._device: str | None = None
        self._loaded = False
        self._load_error: str | None = None

    @property
    def ready(self) -> bool:
        if settings.mock:
            return True
        return self._loaded and self._model is not None and self._processor is not None

    @property
    def status(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "mock": settings.mock,
            "model_id": settings.model_id,
            "load_mode": settings.load_mode,
            "device": self._device,
            "loaded": self._loaded,
            "error": self._load_error,
        }

    def load(self) -> None:
        if settings.mock:
            self._loaded = True
            self._device = "mock"
            logger.info("Mock mode enabled — model weights will not be loaded.")
            return

        with self._lock:
            if self._loaded and self._model is not None:
                return
            try:
                self._load_model_unlocked()
            except Exception as exc:  # noqa: BLE001 — surface to API/health
                self._load_error = str(exc)
                self._loaded = False
                logger.exception("Failed to load VideoChat3")
                raise

    def _load_model_unlocked(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        logger.info(
            "Loading %s (load_mode=%s). First run downloads weights from Hugging Face.",
            settings.model_id,
            settings.load_mode,
        )

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "device_map": "auto",
        }

        if settings.load_mode == "4bit":
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif settings.load_mode == "8bit":
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            model_kwargs["torch_dtype"] = "auto"

        self._model = AutoModelForCausalLM.from_pretrained(settings.model_id, **model_kwargs)
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(
            settings.model_id,
            trust_remote_code=True,
        )

        try:
            self._device = str(next(self._model.parameters()).device)
        except StopIteration:
            self._device = "unknown"

        self._loaded = True
        self._load_error = None
        logger.info("VideoChat3 ready on device=%s", self._device)

    def ask(
        self,
        video_path: str | Path,
        question: str,
        *,
        max_new_tokens: int | None = None,
    ) -> AskResult:
        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(f"Video not found: {path}")
        if not question.strip():
            raise ValueError("Question must be non-empty.")

        if settings.mock:
            return AskResult(
                answer=(
                    f"[mock] Would ask VideoChat3 about '{path.name}':\n\n"
                    f"{question.strip()}\n\n"
                    "Set LECTUREBUDDY_MOCK=0 and install requirements to run the real model."
                ),
                model_id=settings.model_id,
                mock=True,
                device="mock",
            )

        if not self.ready:
            self.load()

        assert self._model is not None
        assert self._processor is not None

        from qwen_vl_utils import process_vision_info

        tokens = max_new_tokens or settings.max_new_tokens
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": str(path.resolve()),
                        "fps": settings.video_fps,
                        "max_frames": settings.max_frames,
                        "max_pixels": settings.max_pixels,
                    },
                    {"type": "text", "text": question.strip()},
                ],
            }
        ]

        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        images, videos, video_kwargs = process_vision_info(
            messages,
            image_patch_size=14,
            return_video_kwargs=True,
            return_video_metadata=True,
        )

        video_metadatas = None
        if videos is not None:
            videos, video_metadatas = zip(*videos)
            videos, video_metadatas = list(videos), list(video_metadatas)

        inputs = self._processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=video_metadatas,
            do_resize=False,
            return_tensors="pt",
            **(video_kwargs or {}),
        )
        inputs = inputs.to(self._model.device)
        if hasattr(self._model, "dtype"):
            inputs = inputs.to(self._model.dtype)

        generated_ids = self._model.generate(
            **inputs,
            max_new_tokens=tokens,
            do_sample=False,
        )
        trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        answer = self._processor.tokenizer.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        return AskResult(
            answer=answer,
            model_id=settings.model_id,
            mock=False,
            device=self._device,
        )


engine = VideoChat3Engine()
