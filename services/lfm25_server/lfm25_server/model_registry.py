"""
Model registry with startup-time capability validation.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import ModelSpec
from .model_backend import supported_backends
from .prompt_store import PromptStore


class ModelRegistryError(ValueError):
    """Base class for model registry errors."""


class UnknownModelError(ModelRegistryError):
    """Raised when a requested model id is not configured."""


class ModelDisabledError(ModelRegistryError):
    """Raised when a requested model is configured but disabled."""


class CapabilityError(ModelRegistryError):
    """Raised when model capability contracts are not satisfied."""


@dataclass(frozen=True)
class ModelCapabilityInfo:
    model_id: str
    prompt_model_id: str
    ready: bool
    missing_templates: tuple[str, ...]


class ModelRegistry:
    def __init__(self, models: tuple[ModelSpec, ...], *, prompt_store: PromptStore | None = None) -> None:
        if not models:
            raise ModelRegistryError("No models configured")
        self._models = models
        self._by_id = {m.id: m for m in models}
        if len(self._by_id) != len(self._models):
            raise ModelRegistryError("Duplicate model ids in registry")

        self._prompt_store = prompt_store or PromptStore()
        self._capability_info = self._validate_capabilities()

    def get(self, model_id: str) -> ModelSpec:
        spec = self._by_id.get(model_id)
        if spec is None:
            raise UnknownModelError(f"Unknown model: {model_id}")
        return spec

    def resolve_enabled(self, model_id: str) -> ModelSpec:
        spec = self.get(model_id)
        if not spec.enabled:
            raise ModelDisabledError(f"Model is disabled: {model_id}")
        return spec

    def list_models(self) -> tuple[ModelSpec, ...]:
        return self._models

    def list_enabled_models(self) -> tuple[ModelSpec, ...]:
        return tuple(spec for spec in self._models if spec.enabled)

    def capability_info(self, model_id: str) -> ModelCapabilityInfo:
        info = self._capability_info.get(model_id)
        if info is None:
            raise UnknownModelError(f"Unknown model: {model_id}")
        return info

    def _validate_capabilities(self) -> dict[str, ModelCapabilityInfo]:
        infos: dict[str, ModelCapabilityInfo] = {}
        backend_names = supported_backends()

        failures: list[str] = []
        for spec in self._models:
            if spec.backend not in backend_names:
                failures.append(
                    f"model={spec.id} uses unsupported backend={spec.backend!r} "
                    f"(supported={sorted(backend_names)})"
                )
                continue

            missing = ()
            if spec.enabled:
                missing = tuple(self._prompt_store.missing_required_templates(spec.prompt_model_id))
                if missing:
                    failures.append(
                        f"model={spec.id} prompt_model_id={spec.prompt_model_id!r} "
                        f"missing templates: {', '.join(missing)}"
                    )

            infos[spec.id] = ModelCapabilityInfo(
                model_id=spec.id,
                prompt_model_id=spec.prompt_model_id,
                ready=(len(missing) == 0),
                missing_templates=missing,
            )

        if failures:
            raise CapabilityError("Capability validation failed: " + " | ".join(failures))
        return infos
