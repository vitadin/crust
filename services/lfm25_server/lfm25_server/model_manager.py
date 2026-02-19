"""
Multi-model manager with single-active-model memory policy.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .config import GenerationDefaults, ModelSpec, ServerConfig
from .model_backend import GenerationParams, LoadedModel, get_backend
from .model_registry import (
    CapabilityError,
    ModelDisabledError,
    ModelRegistry,
    UnknownModelError,
)
from .prompt_store import PromptStore

logger = logging.getLogger(__name__)


class BusyError(RuntimeError):
    """Raised when generation capacity is saturated."""


class ModelExecutionError(RuntimeError):
    """Raised when model load or generation fails."""


class ModelNotReadyError(RuntimeError):
    """Raised when a model is configured but cannot serve requests."""


class ModelState(str, Enum):
    UNLOADED = "UNLOADED"
    LOADING = "LOADING"
    LOADED = "LOADED"
    ERROR = "ERROR"


@dataclass
class GenerateResult:
    answer: str
    thinking: Optional[str]
    raw: str
    prompt_tokens: int
    generation_tokens: int
    generation_tps: float
    peak_memory_gb: float
    finish_reason: Optional[str]
    cold_start: bool
    load_time_s: float
    model_id: str
    model_switch: bool


@dataclass
class StatusInfo:
    model_loaded: bool
    model_state: str
    idle_s: Optional[float]
    idle_timeout_s: float
    load_count: int
    queue_depth: int
    inference_in_progress: bool
    last_error: Optional[str]
    uptime_s: float
    default_model_id: str
    active_model_id: Optional[str]
    registered_models_count: int
    model_switch_count: int


@dataclass
class ModelStatusInfo:
    model_id: str
    enabled: bool
    backend: str
    prompt_model_id: str
    model_path: str
    state: str
    loaded: bool
    load_count: int
    last_error: Optional[str]
    idle_s: Optional[float]
    ready: bool
    missing_templates: tuple[str, ...]


@dataclass
class _PerModelRuntime:
    state: ModelState = ModelState.UNLOADED
    load_count: int = 0
    last_error: Optional[str] = None
    last_used: Optional[float] = None


class MultiModelManager:
    def __init__(
        self,
        config: ServerConfig,
        *,
        registry: Optional[ModelRegistry] = None,
    ) -> None:
        self._config = config
        self._registry = registry or ModelRegistry(config.models, prompt_store=PromptStore())

        # Runtime model routing default is process-local.
        self._runtime_default_model_id = config.default_model_id
        self._registry.resolve_enabled(self._runtime_default_model_id)

        self._active_model_id: Optional[str] = None
        self._active_backend_name: Optional[str] = None
        self._active_loaded: Optional[LoadedModel] = None
        self._model_switch_count = 0

        self._runtime: dict[str, _PerModelRuntime] = {
            spec.id: _PerModelRuntime()
            for spec in self._registry.list_models()
        }

        self._last_error: Optional[str] = None
        self._inference_in_progress = False

        self._gen_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._queue_lock = threading.Lock()
        self._queue_waiters = 0

        self._stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None
        self._started_at = time.monotonic()

    # ---- Lifecycle -----------------------------------------------------------------

    def start_idle_monitor(self) -> None:
        if self._monitor_thread and self._monitor_thread.is_alive():
            logger.debug("Idle monitor already running; skipping duplicate start.")
            return
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._idle_monitor_loop,
            name="idle-monitor",
            daemon=True,
        )
        self._monitor_thread.start()
        logger.info(
            "Idle monitor started (timeout=%.0f s, check_interval=%.1f s)",
            self._config.idle_timeout_s,
            self._config.idle_check_interval_s,
        )

    def shutdown(self) -> None:
        logger.info("Shutdown: stopping idle monitor and unloading active model.")
        self._stop_event.set()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=self._config.shutdown_wait_timeout_s)

        acquired = self._gen_lock.acquire(timeout=self._config.shutdown_wait_timeout_s)
        if not acquired:
            logger.warning("Could not acquire generation lock during shutdown; skipping unload.")
            return
        try:
            self._unload_active_model(next_state=ModelState.UNLOADED)
        finally:
            self._gen_lock.release()

    # ---- Routing and control -------------------------------------------------------

    def resolve_model_id(self, requested_model: Optional[str]) -> str:
        if requested_model is None or str(requested_model).strip() == "":
            with self._state_lock:
                model_id = self._runtime_default_model_id
            self._registry.resolve_enabled(model_id)
            return model_id

        model_id = str(requested_model).strip()
        self._registry.resolve_enabled(model_id)
        return model_id

    def get_prompt_model_id(self, model_id: str) -> str:
        spec = self._registry.resolve_enabled(model_id)
        return spec.prompt_model_id

    def get_default_model_id(self) -> str:
        with self._state_lock:
            return self._runtime_default_model_id

    def set_default_model(self, model_id: str) -> str:
        spec = self._registry.resolve_enabled(model_id)
        with self._state_lock:
            self._runtime_default_model_id = spec.id
        return spec.id

    def force_load(self, model_id: Optional[str] = None) -> tuple[str, bool, bool, float]:
        target_model_id = self.resolve_model_id(model_id)
        acquired = self._gen_lock.acquire(timeout=self._config.generation_wait_timeout_s)
        if not acquired:
            raise BusyError("server busy: could not acquire generation lock")
        try:
            cold_start, load_time, model_switch = self._ensure_model_loaded(target_model_id)
            return target_model_id, cold_start, model_switch, load_time
        finally:
            self._gen_lock.release()

    def force_unload(self, model_id: Optional[str] = None) -> tuple[str, bool]:
        target_model_id = self.resolve_model_id(model_id)
        acquired = self._gen_lock.acquire(timeout=self._config.generation_wait_timeout_s)
        if not acquired:
            raise BusyError("server busy: could not acquire generation lock")
        try:
            with self._state_lock:
                active_id = self._active_model_id
            if active_id != target_model_id:
                return target_model_id, False
            return target_model_id, self._unload_active_model(next_state=ModelState.UNLOADED)
        finally:
            self._gen_lock.release()

    # ---- Generation ----------------------------------------------------------------

    def generate(
        self,
        messages: list[dict],
        *,
        model_id: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        repetition_context_size: Optional[int] = None,
    ) -> GenerateResult:
        from .think_parser import parse_think

        resolved_model_id = self.resolve_model_id(model_id)
        spec = self._registry.resolve_enabled(resolved_model_id)

        lock_acquired = False
        self._acquire_generation_slot()
        lock_acquired = True
        self._set_inference_in_progress(True)

        try:
            cold_start, load_time, model_switch = self._ensure_model_loaded(resolved_model_id)
            params = self._resolve_generation_params(
                spec,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                repetition_context_size=repetition_context_size,
            )

            with self._state_lock:
                loaded = self._active_loaded
            if loaded is None:
                raise ModelNotReadyError(f"model {resolved_model_id!r} is not loaded")

            backend = get_backend(spec.backend)
            backend_result = backend.generate(loaded, messages, params)
            parsed = parse_think(backend_result.raw_text)

            with self._state_lock:
                runtime = self._runtime[resolved_model_id]
                runtime.last_used = time.monotonic()
                runtime.last_error = None
                runtime.state = ModelState.LOADED
                self._last_error = None

            return GenerateResult(
                answer=parsed.answer,
                thinking=parsed.thinking,
                raw=backend_result.raw_text,
                prompt_tokens=backend_result.prompt_tokens,
                generation_tokens=backend_result.generation_tokens,
                generation_tps=backend_result.generation_tps,
                peak_memory_gb=backend_result.peak_memory_gb,
                finish_reason=backend_result.finish_reason,
                cold_start=cold_start,
                load_time_s=load_time,
                model_id=resolved_model_id,
                model_switch=model_switch,
            )
        except BusyError:
            raise
        except (UnknownModelError, ModelDisabledError):
            raise
        except Exception as exc:
            logger.exception("Generation failed for model=%s", resolved_model_id)
            self._record_error(resolved_model_id, exc)
            if lock_acquired:
                self._unload_active_model(next_state=ModelState.ERROR)
            raise ModelExecutionError(str(exc)) from exc
        finally:
            self._set_inference_in_progress(False)
            if lock_acquired:
                self._gen_lock.release()

    # ---- Status --------------------------------------------------------------------

    def status(self) -> StatusInfo:
        with self._state_lock:
            active_model_id = self._active_model_id
            model_loaded = self._active_loaded is not None
            runtime_default = self._runtime_default_model_id
            in_progress = self._inference_in_progress
            last_error = self._last_error
            model_switch_count = self._model_switch_count

            if active_model_id is not None:
                active_runtime = self._runtime[active_model_id]
                model_state = active_runtime.state.value
                idle_s = (
                    (time.monotonic() - active_runtime.last_used)
                    if active_runtime.last_used is not None
                    else None
                )
            else:
                model_state = ModelState.UNLOADED.value
                idle_s = None

            total_load_count = sum(r.load_count for r in self._runtime.values())

        with self._queue_lock:
            queue_depth = self._queue_waiters

        return StatusInfo(
            model_loaded=model_loaded,
            model_state=model_state,
            idle_s=idle_s,
            idle_timeout_s=self._config.idle_timeout_s,
            load_count=total_load_count,
            queue_depth=queue_depth,
            inference_in_progress=in_progress,
            last_error=last_error,
            uptime_s=time.monotonic() - self._started_at,
            default_model_id=runtime_default,
            active_model_id=active_model_id,
            registered_models_count=len(self._registry.list_models()),
            model_switch_count=model_switch_count,
        )

    def list_models_status(self) -> list[ModelStatusInfo]:
        with self._state_lock:
            active_id = self._active_model_id
            now = time.monotonic()
            runtime_snapshot = {
                model_id: _PerModelRuntime(
                    state=rt.state,
                    load_count=rt.load_count,
                    last_error=rt.last_error,
                    last_used=rt.last_used,
                )
                for model_id, rt in self._runtime.items()
            }

        models: list[ModelStatusInfo] = []
        for spec in self._registry.list_models():
            rt = runtime_snapshot[spec.id]
            cap = self._registry.capability_info(spec.id)
            idle_s = (now - rt.last_used) if rt.last_used is not None else None
            models.append(
                ModelStatusInfo(
                    model_id=spec.id,
                    enabled=spec.enabled,
                    backend=spec.backend,
                    prompt_model_id=spec.prompt_model_id,
                    model_path=spec.model_path,
                    state=rt.state.value,
                    loaded=(spec.id == active_id and rt.state == ModelState.LOADED),
                    load_count=rt.load_count,
                    last_error=rt.last_error,
                    idle_s=idle_s,
                    ready=cap.ready,
                    missing_templates=cap.missing_templates,
                )
            )
        return models

    def model_status(self, model_id: str) -> ModelStatusInfo:
        self._registry.get(model_id)
        for info in self.list_models_status():
            if info.model_id == model_id:
                return info
        raise UnknownModelError(f"Unknown model: {model_id}")

    # ---- Internal ------------------------------------------------------------------

    def _resolve_generation_params(
        self,
        spec: ModelSpec,
        *,
        max_tokens: Optional[int],
        temperature: Optional[float],
        top_k: Optional[int],
        top_p: Optional[float],
        repetition_penalty: Optional[float],
        repetition_context_size: Optional[int],
    ) -> GenerationParams:
        model_defaults: GenerationDefaults = spec.generation_defaults
        return GenerationParams(
            max_tokens=max_tokens
            if max_tokens is not None
            else (
                model_defaults.max_tokens
                if model_defaults.max_tokens is not None
                else self._config.max_tokens
            ),
            temperature=temperature
            if temperature is not None
            else (
                model_defaults.temperature
                if model_defaults.temperature is not None
                else self._config.temperature
            ),
            top_k=top_k
            if top_k is not None
            else (
                model_defaults.top_k
                if model_defaults.top_k is not None
                else self._config.top_k
            ),
            top_p=top_p
            if top_p is not None
            else (
                model_defaults.top_p
                if model_defaults.top_p is not None
                else self._config.top_p
            ),
            repetition_penalty=repetition_penalty
            if repetition_penalty is not None
            else (
                model_defaults.repetition_penalty
                if model_defaults.repetition_penalty is not None
                else self._config.repetition_penalty
            ),
            repetition_context_size=repetition_context_size
            if repetition_context_size is not None
            else (
                model_defaults.repetition_context_size
                if model_defaults.repetition_context_size is not None
                else self._config.repetition_context_size
            ),
        )

    def _acquire_generation_slot(self) -> None:
        if self._gen_lock.acquire(blocking=False):
            return

        with self._queue_lock:
            if self._queue_waiters >= self._config.max_queue_waiters:
                raise BusyError("server busy: wait queue is full")
            self._queue_waiters += 1

        acquired = False
        try:
            acquired = self._gen_lock.acquire(timeout=self._config.generation_wait_timeout_s)
        finally:
            with self._queue_lock:
                self._queue_waiters -= 1

        if not acquired:
            raise BusyError("server busy: timed out waiting for generation slot")

    def _ensure_model_loaded(self, target_model_id: str) -> tuple[bool, float, bool]:
        spec = self._registry.resolve_enabled(target_model_id)
        backend = get_backend(spec.backend)

        model_switch = False
        with self._state_lock:
            active_id = self._active_model_id
            active_loaded = self._active_loaded
            if active_id == target_model_id and active_loaded is not None:
                self._runtime[target_model_id].state = ModelState.LOADED
                return False, 0.0, False
            model_switch = active_id is not None and active_id != target_model_id and active_loaded is not None
            self._runtime[target_model_id].state = ModelState.LOADING

        if model_switch:
            self._unload_active_model(next_state=ModelState.UNLOADED)

        t0 = time.perf_counter()
        try:
            loaded = backend.load(spec.model_path)
        except Exception as exc:
            self._record_error(target_model_id, exc)
            raise
        load_time = time.perf_counter() - t0

        with self._state_lock:
            self._active_model_id = target_model_id
            self._active_backend_name = backend.name
            self._active_loaded = loaded
            rt = self._runtime[target_model_id]
            rt.state = ModelState.LOADED
            rt.load_count += 1
            rt.last_error = None
            rt.last_used = time.monotonic()
            if model_switch:
                self._model_switch_count += 1

        return True, load_time, model_switch

    def _unload_active_model(self, *, next_state: ModelState) -> bool:
        with self._state_lock:
            active_id = self._active_model_id
            active_backend_name = self._active_backend_name
            active_loaded = self._active_loaded

            if active_id is None or active_loaded is None or active_backend_name is None:
                return False

            self._active_model_id = None
            self._active_backend_name = None
            self._active_loaded = None
            self._runtime[active_id].state = next_state

        backend = get_backend(active_backend_name)
        try:
            backend.unload(active_loaded)
        except Exception as exc:  # pragma: no cover - best effort cleanup
            logger.warning("Failed to unload backend resources for model=%s: %s", active_id, exc)
        return True

    def _record_error(self, model_id: str, exc: Exception) -> None:
        err = f"{type(exc).__name__}: {exc}"
        with self._state_lock:
            if model_id in self._runtime:
                self._runtime[model_id].last_error = err
                self._runtime[model_id].state = ModelState.ERROR
            self._last_error = err

    def _set_inference_in_progress(self, value: bool) -> None:
        with self._state_lock:
            self._inference_in_progress = value

    def _idle_monitor_loop(self) -> None:
        cfg = self._config
        if cfg.idle_timeout_s <= 0:
            logger.info("Auto-unload disabled (idle_timeout_s=0).")
            return

        while not self._stop_event.wait(timeout=cfg.idle_check_interval_s):
            with self._state_lock:
                active_id = self._active_model_id
                active_loaded = self._active_loaded
                last_used = (
                    self._runtime[active_id].last_used
                    if active_id is not None and active_id in self._runtime
                    else None
                )

            if active_id is None or active_loaded is None or last_used is None:
                continue

            idle_s = time.monotonic() - last_used
            if idle_s < cfg.idle_timeout_s:
                continue

            acquired = self._gen_lock.acquire(timeout=5.0)
            if not acquired:
                continue
            try:
                with self._state_lock:
                    current_active = self._active_model_id
                    current_last_used = (
                        self._runtime[current_active].last_used
                        if current_active is not None and current_active in self._runtime
                        else None
                    )
                if current_active is None or current_last_used is None:
                    continue
                if (time.monotonic() - current_last_used) >= cfg.idle_timeout_s:
                    logger.info(
                        "Idle timeout reached for model=%s (%.1f s). Unloading model.",
                        current_active,
                        idle_s,
                    )
                    self._unload_active_model(next_state=ModelState.UNLOADED)
            finally:
                self._gen_lock.release()


# Backward-compatible alias.
ModelManager = MultiModelManager

__all__ = [
    "BusyError",
    "CapabilityError",
    "GenerateResult",
    "ModelDisabledError",
    "ModelExecutionError",
    "ModelManager",
    "ModelNotReadyError",
    "ModelState",
    "MultiModelManager",
    "StatusInfo",
    "UnknownModelError",
]
