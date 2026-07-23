"""VideoChat3-4B inference wrapper (official HF demo pattern)."""

from __future__ import annotations

import logging
import threading
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import settings

logger = logging.getLogger(__name__)


def _is_unset_sentinel(value: Any) -> bool:
    """transformers / hub leave bare `object()` instances for unset optional fields."""
    return type(value) is object


def _strip_unset_sentinels(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in kwargs.items() if not _is_unset_sentinel(value)}


def _patch_prepare_inputs_for_generation(model: Any) -> None:
    """
    VideoChat3 remote code needs two transformers-compat fixes during generate():

    1. Newer GenerationMixin may omit `inputs_embeds` from model_inputs → KeyError
       if remote code indexes it with [].
    2. Newer generate() no longer passes `cache_position` into prepare_inputs; it is
       synthesized inside the parent instead. The hub code only looks at the *argument*,
       so it never clears pixel_values_videos on decode steps. That re-runs the vision
       encoder on a 1-token prompt with 0 <|video_pad|> tokens →
       "Videos features and video tokens do not match: tokens: 0, ...".
    """
    if getattr(model, "_lecturebuddy_prepare_inputs_patch", False):
        return

    parent_prepare = None
    for cls in type(model).__mro__[1:]:
        cand = cls.__dict__.get("prepare_inputs_for_generation")
        if cand is not None:
            parent_prepare = cand
            break

    if parent_prepare is None:
        logger.warning("Could not locate parent prepare_inputs_for_generation; skipping patch")
        return

    def prepare_inputs_for_generation_fixed(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        image_grid_thw=None,
        pixel_values_videos=None,
        video_grid_thw=None,
        attention_mask=None,
        cache_position=None,
        logits_to_keep=None,
        **kwargs,
    ):
        model_inputs = parent_prepare(
            self,
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            **kwargs,
        )
        embeds = model_inputs.get("inputs_embeds")
        ids = model_inputs.get("input_ids")
        # Parent (GenerationMixin) rebuilds cache_position for remote-code models;
        # the positional/kw arg from generate() is often None on current transformers.
        cp = model_inputs.get("cache_position", cache_position)
        is_decoding_step = (embeds is not None and embeds.shape[1] == 1) or (
            ids is not None and ids.shape[1] == 1
        )
        if (
            cp is not None
            and len(cp) > 0
            and int(cp[0]) != 0
            and is_decoding_step
        ):
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None
        return model_inputs

    model.prepare_inputs_for_generation = types.MethodType(
        prepare_inputs_for_generation_fixed, model
    )
    model._lecturebuddy_prepare_inputs_patch = True
    logger.info("Patched prepare_inputs_for_generation for transformers compatibility")


def _ensure_lm_head_tied(model: Any) -> None:
    """
    Checkpoint omits lm_head.weight (tied to embed_tokens). Under device_map=auto /
    bitsandbytes, HF's automatic tie often fails and leaves a randomly initialized
    head → fluent token salad. Force-share the embedding matrix.
    """
    import torch.nn as nn

    embed = model.get_input_embeddings()
    if embed is None or not hasattr(model, "lm_head"):
        return
    if model.lm_head.weight.data_ptr() == embed.weight.data_ptr():
        return

    head = nn.Linear(
        embed.embedding_dim,
        embed.num_embeddings,
        bias=False,
        device=embed.weight.device,
        dtype=embed.weight.dtype,
    )
    head.weight = embed.weight
    model.lm_head = head
    logger.info("Tied lm_head.weight to input embeddings (checkpoint omits lm_head)")


def _patch_processor_against_hub_strict_validation(processor: Any) -> None:
    """
    transformers kwargs-merge injects fresh object() sentinels for unset fields.
    huggingface_hub only skips its *shared* DEFAULT sentinel (identity), so
    validate_typed_dict crashes on size / do_convert_rgb / etc.

    Disable that hub check here (transformers still validates real values later
    in _validate_preprocess_kwargs) and strip sentinels from preprocess kwargs.
    """
    import transformers.processing_utils as processing_utils
    import transformers.video_processing_utils as video_processing_utils

    def _patch_module(module: Any, flag: str) -> None:
        if getattr(module, flag, False):
            return

        def validate_typed_dict(schema: Any, data: Any) -> None:
            # Intentionally no-op: hub's TypedDict→dataclass path type-checks
            # unrelated object() placeholders and aborts real video loads.
            return None

        module.validate_typed_dict = validate_typed_dict
        setattr(module, flag, True)

    _patch_module(video_processing_utils, "_lecturebuddy_validate_patch")
    _patch_module(processing_utils, "_lecturebuddy_validate_patch")

    for attr in ("video_processor", "image_processor"):
        sub = getattr(processor, attr, None)
        if sub is None or getattr(sub, "_lecturebuddy_sentinel_patch", False):
            continue

        bound = sub.preprocess
        raw = getattr(bound, "__func__", bound)

        def _make_wrapped(orig_func):
            def preprocess(self, videos, **kwargs):
                return orig_func(self, videos, **_strip_unset_sentinels(kwargs))

            return preprocess

        sub.preprocess = types.MethodType(_make_wrapped(raw), sub)
        sub._lecturebuddy_sentinel_patch = True


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
        self._input_dtype = None  # torch.dtype for floating multimodal tensors
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
        from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
        from transformers.utils.import_utils import is_flash_attn_2_available

        if not torch.cuda.is_available():
            logger.warning(
                "PyTorch reports no CUDA (likely a CPU-only torch wheel). "
                "VideoChat3 will run on CPU and be very slow. "
                "Install a CUDA build, e.g. "
                "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124"
            )

        logger.info(
            "Loading %s (load_mode=%s). First run downloads weights from Hugging Face.",
            settings.model_id,
            settings.load_mode,
        )

        config = AutoConfig.from_pretrained(settings.model_id, trust_remote_code=True)
        # VideoChat3 vision defaults to flash_attention_2; flash-attn is often missing on Windows.
        vision_cfg = getattr(config, "vision_config", None)
        use_flash = is_flash_attn_2_available()
        vision_attn = "flash_attention_2" if use_flash else "sdpa"
        if vision_cfg is not None and getattr(vision_cfg, "attn_impl", None) != vision_attn:
            vision_cfg.attn_impl = vision_attn
            logger.info("Using vision attn_impl=%s (flash_attn available=%s)", vision_attn, use_flash)

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "device_map": "auto",
            "config": config,
            "attn_implementation": "flash_attention_2" if use_flash else "sdpa",
        }

        # Keep vision + output head (+ tied embeddings) in bf16. Quantizing those
        # on VideoChat3 produces fluent-looking garbage token salad.
        _skip_4bit = [
            "lm_head",
            "embed_tokens",
            "vision_tower",
            "multi_modal_projector",
        ]

        if settings.load_mode in {"4bit", "8bit"} and not torch.cuda.is_available():
            logger.warning(
                "load_mode=%s needs CUDA; falling back to full-precision CPU load.",
                settings.load_mode,
            )
            model_kwargs["torch_dtype"] = torch.float32
            self._input_dtype = torch.float32
        elif settings.load_mode == "4bit":
            from transformers import BitsAndBytesConfig

            # Native Qwen3/VideoChat3 weights are bf16; match compute dtype.
            compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            model_kwargs["torch_dtype"] = compute_dtype
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                llm_int8_skip_modules=_skip_4bit,
            )
            self._input_dtype = compute_dtype
            logger.info(
                "4bit load: compute_dtype=%s skip_modules=%s",
                compute_dtype,
                _skip_4bit,
            )
        elif settings.load_mode == "8bit":
            from transformers import BitsAndBytesConfig

            compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            model_kwargs["torch_dtype"] = compute_dtype
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_skip_modules=_skip_4bit,
            )
            self._input_dtype = compute_dtype
        else:
            model_kwargs["torch_dtype"] = "auto"
            self._input_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        self._model = AutoModelForCausalLM.from_pretrained(settings.model_id, **model_kwargs)
        self._model.eval()
        _ensure_lm_head_tied(self._model)

        # Belt-and-suspenders: rewrite any vision layers still set to flash.
        if not use_flash:
            for module in self._model.modules():
                if getattr(module, "attn_impl", None) == "flash_attention_2":
                    module.attn_impl = "sdpa"

        _patch_prepare_inputs_for_generation(self._model)

        self._processor = AutoProcessor.from_pretrained(
            settings.model_id,
            trust_remote_code=True,
        )
        _patch_processor_against_hub_strict_validation(self._processor)

        try:
            self._device = str(next(self._model.parameters()).device)
        except StopIteration:
            self._device = "unknown"

        # Checkpoint omits lm_head (tied embeddings); we retie above if needed.
        logger.info("VideoChat3 ready on device=%s", self._device)
        self._loaded = True
        self._load_error = None

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

        # Match official demo_vc3.py; sentinel strip happens in preprocess patch.
        inputs = self._processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=video_metadatas,
            do_resize=False,
            return_tensors="pt",
            **(video_kwargs or {}),
        )
        video_token_id = self._processor.video_token_id
        n_video_tokens = int((inputs["input_ids"] == video_token_id).sum().item())
        if n_video_tokens == 0:
            raise ValueError(
                "Processor produced 0 video placeholder tokens; check video_metadata/frames_indices."
            )

        inputs = inputs.to(self._model.device)
        # Cast only floating multimodal tensors; never use model.dtype under 4bit
        # (can be a quantized/storage dtype and corrupts vision features).
        if self._input_dtype is not None:
            inputs = inputs.to(self._input_dtype)

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
