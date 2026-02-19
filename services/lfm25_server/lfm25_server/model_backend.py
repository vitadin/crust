"""
Backend abstraction for model load/generate/unload operations.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any, Optional, Protocol


@dataclass
class LoadedModel:
    model: Any
    tokenizer: Any


@dataclass
class BackendGenerateResult:
    raw_text: str
    prompt_tokens: int
    generation_tokens: int
    generation_tps: float
    peak_memory_gb: float
    finish_reason: Optional[str]


@dataclass(frozen=True)
class GenerationParams:
    max_tokens: int
    temperature: float
    top_k: int
    top_p: float
    repetition_penalty: float
    repetition_context_size: int


class ModelBackend(Protocol):
    name: str

    def load(self, model_path: str) -> LoadedModel:
        ...

    def generate(
        self,
        loaded: LoadedModel,
        messages: list[dict],
        params: GenerationParams,
    ) -> BackendGenerateResult:
        ...

    def unload(self, loaded: LoadedModel) -> None:
        ...


class MLXBackend:
    name = "mlx"

    def load(self, model_path: str) -> LoadedModel:
        from mlx_lm import load as mlx_load

        model, tokenizer = mlx_load(model_path)
        return LoadedModel(model=model, tokenizer=tokenizer)

    def generate(
        self,
        loaded: LoadedModel,
        messages: list[dict],
        params: GenerationParams,
    ) -> BackendGenerateResult:
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        sampler = make_sampler(temp=params.temperature, top_k=params.top_k, top_p=params.top_p)
        logits_processors = make_logits_processors(
            repetition_penalty=params.repetition_penalty,
            repetition_context_size=params.repetition_context_size,
        )

        prompt = loaded.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        raw_parts: list[str] = []
        last_response = None
        for response in stream_generate(
            loaded.model,
            loaded.tokenizer,
            prompt=prompt,
            max_tokens=params.max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
        ):
            raw_parts.append(response.text)
            last_response = response

        return BackendGenerateResult(
            raw_text="".join(raw_parts),
            prompt_tokens=last_response.prompt_tokens if last_response else 0,
            generation_tokens=last_response.generation_tokens if last_response else 0,
            generation_tps=last_response.generation_tps if last_response else 0.0,
            peak_memory_gb=last_response.peak_memory if last_response else 0.0,
            finish_reason=last_response.finish_reason if last_response else None,
        )

    def unload(self, loaded: LoadedModel) -> None:
        del loaded.model
        del loaded.tokenizer
        gc.collect()
        try:
            import mlx.core as mx

            mx.metal.clear_cache()
        except Exception:
            # Best-effort cache cleanup.
            pass


_BACKENDS: dict[str, ModelBackend] = {
    "mlx": MLXBackend(),
}


def get_backend(name: str) -> ModelBackend:
    backend = _BACKENDS.get(name.lower())
    if backend is None:
        raise ValueError(f"Unsupported backend: {name!r}")
    return backend


def supported_backends() -> frozenset[str]:
    return frozenset(_BACKENDS.keys())
