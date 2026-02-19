"""
Configuration management with strict validation and multi-model support.

Load order:
  1. Dataclass defaults.
  2. YAML overrides.
  3. CLI overrides.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import re
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)


class ConfigError(ValueError):
    """Raised when configuration values are invalid."""


@dataclasses.dataclass(frozen=True)
class GenerationDefaults:
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    repetition_context_size: Optional[int] = None


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    id: str
    backend: str
    model_path: str
    prompt_model_id: str
    enabled: bool = True
    generation_defaults: GenerationDefaults = dataclasses.field(default_factory=GenerationDefaults)


@dataclasses.dataclass(frozen=True)
class ServerConfig:
    # Server
    host: str = "127.0.0.1"
    port: int = 8765
    log_level: str = "INFO"
    json_logs: bool = False
    socket_timeout_s: float = 30.0
    listen_backlog: int = 64
    request_max_body_bytes: int = 1_048_576

    # Models
    default_model_id: str = "lfm2.5"
    models: tuple[ModelSpec, ...] = dataclasses.field(default_factory=tuple)

    # Legacy single-model fields (auto-upgraded into `models` when needed)
    model_path: str = "./models/LFM2.5-1.2B-Thinking-MLX-8bit"
    prompt_model_id: str = "lfm2.5"

    # Lifecycle and queueing
    idle_timeout_s: float = 300.0
    idle_check_interval_s: float = 30.0
    generation_wait_timeout_s: float = 60.0
    max_queue_waiters: int = 16
    shutdown_wait_timeout_s: float = 5.0

    # Global generation defaults
    max_tokens: int = 1024
    temperature: float = 0.6
    top_k: int = 50
    top_p: float = 0.95
    repetition_penalty: float = 1.05
    repetition_context_size: int = 20

    # Response behavior
    include_thinking: bool = False


_VALID_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR"})
_SUPPORTED_BACKENDS = frozenset({"mlx"})
_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def load_config(
    yaml_path: str | Path = "config.yaml",
    cli_args: Optional[argparse.Namespace] = None,
) -> ServerConfig:
    """Build and validate ServerConfig by merging defaults -> YAML -> CLI."""
    base = _default_base_dict()
    known_keys = set(base.keys())

    yaml_file = Path(yaml_path)
    if yaml_file.exists():
        with yaml_file.open(encoding="utf-8") as fh:
            yaml_data: dict[str, Any] = yaml.safe_load(fh) or {}
        for key, val in yaml_data.items():
            if key in known_keys:
                base[key] = val
            else:
                logger.warning("Unknown config key %r in %s; ignored", key, yaml_file)
    elif str(yaml_path) != "config.yaml":
        logger.warning("Config file %s not found, using defaults", yaml_path)

    if cli_args is not None:
        for key in known_keys:
            if key == "models":
                continue
            val = getattr(cli_args, key, None)
            if val is not None:
                base[key] = val

    scalars = _coerce_scalar_fields(base)
    models = _parse_models(base.get("models"), scalars)
    _validate_and_normalize_scalars(scalars)
    _validate_models(models, default_model_id=str(scalars["default_model_id"]))

    default_model = next(m for m in models if m.id == scalars["default_model_id"])
    return ServerConfig(
        host=scalars["host"],
        port=scalars["port"],
        log_level=scalars["log_level"],
        json_logs=scalars["json_logs"],
        socket_timeout_s=scalars["socket_timeout_s"],
        listen_backlog=scalars["listen_backlog"],
        request_max_body_bytes=scalars["request_max_body_bytes"],
        default_model_id=scalars["default_model_id"],
        models=tuple(models),
        # Keep legacy fields aligned with active default model for backward compatibility.
        model_path=default_model.model_path,
        prompt_model_id=default_model.prompt_model_id,
        idle_timeout_s=scalars["idle_timeout_s"],
        idle_check_interval_s=scalars["idle_check_interval_s"],
        generation_wait_timeout_s=scalars["generation_wait_timeout_s"],
        max_queue_waiters=scalars["max_queue_waiters"],
        shutdown_wait_timeout_s=scalars["shutdown_wait_timeout_s"],
        max_tokens=scalars["max_tokens"],
        temperature=scalars["temperature"],
        top_k=scalars["top_k"],
        top_p=scalars["top_p"],
        repetition_penalty=scalars["repetition_penalty"],
        repetition_context_size=scalars["repetition_context_size"],
        include_thinking=scalars["include_thinking"],
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Local multi-model inference server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", default="config.yaml", metavar="PATH")
    p.add_argument("--host", metavar="HOST")
    p.add_argument("--port", type=int, metavar="PORT")

    # Legacy single-model convenience flags (auto-upgraded into manifest model)
    p.add_argument("--model-path", dest="model_path", metavar="PATH")
    p.add_argument("--prompt-model-id", dest="prompt_model_id", metavar="MODEL_ID")

    p.add_argument("--default-model-id", dest="default_model_id", metavar="MODEL_ID")
    p.add_argument("--idle-timeout", dest="idle_timeout_s", type=float, metavar="SECONDS")
    p.add_argument("--idle-check-interval", dest="idle_check_interval_s", type=float, metavar="SECONDS")
    p.add_argument("--queue-wait-timeout", dest="generation_wait_timeout_s", type=float, metavar="SECONDS")
    p.add_argument("--max-queue-waiters", dest="max_queue_waiters", type=int, metavar="N")
    p.add_argument("--shutdown-wait-timeout", dest="shutdown_wait_timeout_s", type=float, metavar="SECONDS")
    p.add_argument("--max-body-bytes", dest="request_max_body_bytes", type=int, metavar="BYTES")
    p.add_argument("--socket-timeout", dest="socket_timeout_s", type=float, metavar="SECONDS")
    p.add_argument("--listen-backlog", dest="listen_backlog", type=int, metavar="N")
    p.add_argument("--max-tokens", dest="max_tokens", type=int, metavar="N")
    p.add_argument("--log-level", dest="log_level", metavar="LEVEL", choices=sorted(_VALID_LOG_LEVELS))
    p.add_argument("--json-logs", dest="json_logs", action="store_true", default=None)

    think_group = p.add_mutually_exclusive_group()
    think_group.add_argument("--include-thinking", dest="include_thinking", action="store_true", default=None)
    think_group.add_argument("--no-include-thinking", dest="include_thinking", action="store_false", default=None)
    return p


def _default_base_dict() -> dict[str, Any]:
    defaults = ServerConfig()
    base = {field.name: getattr(defaults, field.name) for field in dataclasses.fields(ServerConfig)}
    # Keep `models` mutable for YAML overrides.
    base["models"] = []
    return base


def _coerce_scalar_fields(base: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    out["host"] = str(base["host"]).strip()
    out["port"] = _coerce_int("port", base["port"])
    out["log_level"] = str(base["log_level"]).strip().upper()
    out["json_logs"] = _coerce_bool("json_logs", base["json_logs"])
    out["socket_timeout_s"] = _coerce_float("socket_timeout_s", base["socket_timeout_s"])
    out["listen_backlog"] = _coerce_int("listen_backlog", base["listen_backlog"])
    out["request_max_body_bytes"] = _coerce_int("request_max_body_bytes", base["request_max_body_bytes"])

    out["default_model_id"] = str(base["default_model_id"]).strip()
    out["model_path"] = str(base["model_path"]).strip()
    out["prompt_model_id"] = str(base["prompt_model_id"]).strip()

    out["idle_timeout_s"] = _coerce_float("idle_timeout_s", base["idle_timeout_s"])
    out["idle_check_interval_s"] = _coerce_float("idle_check_interval_s", base["idle_check_interval_s"])
    out["generation_wait_timeout_s"] = _coerce_float("generation_wait_timeout_s", base["generation_wait_timeout_s"])
    out["max_queue_waiters"] = _coerce_int("max_queue_waiters", base["max_queue_waiters"])
    out["shutdown_wait_timeout_s"] = _coerce_float("shutdown_wait_timeout_s", base["shutdown_wait_timeout_s"])

    out["max_tokens"] = _coerce_int("max_tokens", base["max_tokens"])
    out["temperature"] = _coerce_float("temperature", base["temperature"])
    out["top_k"] = _coerce_int("top_k", base["top_k"])
    out["top_p"] = _coerce_float("top_p", base["top_p"])
    out["repetition_penalty"] = _coerce_float("repetition_penalty", base["repetition_penalty"])
    out["repetition_context_size"] = _coerce_int("repetition_context_size", base["repetition_context_size"])
    out["include_thinking"] = _coerce_bool("include_thinking", base["include_thinking"])
    return out


def _parse_models(raw_models: Any, scalars: dict[str, Any]) -> list[ModelSpec]:
    if raw_models in (None, [], (), ""):
        # Legacy auto-upgrade.
        model_id = scalars["default_model_id"] or "lfm2.5"
        return [
            ModelSpec(
                id=model_id,
                backend="mlx",
                model_path=str(Path(scalars["model_path"]).expanduser()),
                prompt_model_id=scalars["prompt_model_id"] or model_id,
                enabled=True,
                generation_defaults=GenerationDefaults(),
            )
        ]

    if not isinstance(raw_models, list):
        raise ConfigError("'models' must be a list")

    models: list[ModelSpec] = []
    for idx, raw in enumerate(raw_models):
        ctx = f"models[{idx}]"
        if not isinstance(raw, dict):
            raise ConfigError(f"{ctx} must be an object")

        model_id = str(raw.get("id", "")).strip()
        if not model_id:
            raise ConfigError(f"{ctx}.id must be a non-empty string")
        if not _MODEL_ID_RE.fullmatch(model_id):
            raise ConfigError(f"{ctx}.id has invalid characters: {model_id!r}")

        backend = str(raw.get("backend", "mlx")).strip().lower()
        if backend not in _SUPPORTED_BACKENDS:
            raise ConfigError(f"{ctx}.backend must be one of {sorted(_SUPPORTED_BACKENDS)}")

        model_path = str(raw.get("model_path", "")).strip()
        if not model_path:
            raise ConfigError(f"{ctx}.model_path must be provided")

        prompt_model_id = str(raw.get("prompt_model_id", model_id)).strip()
        if not prompt_model_id:
            raise ConfigError(f"{ctx}.prompt_model_id must be a non-empty string")

        enabled = _coerce_bool(f"{ctx}.enabled", raw.get("enabled", True))
        generation_defaults = _parse_generation_defaults(
            raw.get("generation_defaults"),
            context=f"{ctx}.generation_defaults",
        )

        models.append(
            ModelSpec(
                id=model_id,
                backend=backend,
                model_path=str(Path(model_path).expanduser()),
                prompt_model_id=prompt_model_id,
                enabled=enabled,
                generation_defaults=generation_defaults,
            )
        )
    return models


def _parse_generation_defaults(raw: Any, *, context: str) -> GenerationDefaults:
    if raw in (None, ""):
        return GenerationDefaults()
    if not isinstance(raw, dict):
        raise ConfigError(f"{context} must be an object")

    allowed = {
        "max_tokens",
        "temperature",
        "top_k",
        "top_p",
        "repetition_penalty",
        "repetition_context_size",
    }
    for key in raw:
        if key not in allowed:
            raise ConfigError(f"Unknown key {context}.{key}")

    kwargs: dict[str, Any] = {}
    if "max_tokens" in raw:
        kwargs["max_tokens"] = _coerce_int(f"{context}.max_tokens", raw["max_tokens"])
    if "temperature" in raw:
        kwargs["temperature"] = _coerce_float(f"{context}.temperature", raw["temperature"])
    if "top_k" in raw:
        kwargs["top_k"] = _coerce_int(f"{context}.top_k", raw["top_k"])
    if "top_p" in raw:
        kwargs["top_p"] = _coerce_float(f"{context}.top_p", raw["top_p"])
    if "repetition_penalty" in raw:
        kwargs["repetition_penalty"] = _coerce_float(f"{context}.repetition_penalty", raw["repetition_penalty"])
    if "repetition_context_size" in raw:
        kwargs["repetition_context_size"] = _coerce_int(
            f"{context}.repetition_context_size",
            raw["repetition_context_size"],
        )

    defaults = GenerationDefaults(**kwargs)
    _validate_generation_defaults(defaults, context=context)
    return defaults


def _validate_and_normalize_scalars(cfg: dict[str, Any]) -> None:
    host = cfg["host"]
    if host == "localhost":
        host = "127.0.0.1"
    if host != "127.0.0.1":
        raise ConfigError("host must be loopback-only (127.0.0.1)")
    cfg["host"] = host

    if cfg["log_level"] not in _VALID_LOG_LEVELS:
        raise ConfigError(f"log_level must be one of {sorted(_VALID_LOG_LEVELS)}")
    if not cfg["default_model_id"]:
        raise ConfigError("default_model_id must be a non-empty string")

    _require_range("port", cfg["port"], min_value=1, max_value=65535)
    _require_range("idle_timeout_s", cfg["idle_timeout_s"], min_value=0.0)
    _require_range("idle_check_interval_s", cfg["idle_check_interval_s"], min_value=0.1)
    _require_range("generation_wait_timeout_s", cfg["generation_wait_timeout_s"], min_value=0.1)
    _require_range("shutdown_wait_timeout_s", cfg["shutdown_wait_timeout_s"], min_value=0.1)
    _require_range("max_queue_waiters", cfg["max_queue_waiters"], min_value=0)
    _require_range("socket_timeout_s", cfg["socket_timeout_s"], min_value=1.0)
    _require_range("listen_backlog", cfg["listen_backlog"], min_value=1)
    _require_range("request_max_body_bytes", cfg["request_max_body_bytes"], min_value=1024)

    _validate_generation_defaults(
        GenerationDefaults(
            max_tokens=cfg["max_tokens"],
            temperature=cfg["temperature"],
            top_k=cfg["top_k"],
            top_p=cfg["top_p"],
            repetition_penalty=cfg["repetition_penalty"],
            repetition_context_size=cfg["repetition_context_size"],
        ),
        context="global generation defaults",
    )


def _validate_models(models: list[ModelSpec], *, default_model_id: str) -> None:
    if not models:
        raise ConfigError("At least one model must be configured")

    by_id: dict[str, ModelSpec] = {}
    enabled_count = 0
    for spec in models:
        if spec.id in by_id:
            raise ConfigError(f"Duplicate model id: {spec.id!r}")
        by_id[spec.id] = spec

        if spec.backend not in _SUPPORTED_BACKENDS:
            raise ConfigError(f"Unsupported backend for model {spec.id!r}: {spec.backend!r}")
        if spec.enabled:
            enabled_count += 1
            path = Path(spec.model_path)
            if not path.exists():
                raise ConfigError(f"model {spec.id!r} path does not exist: {path}")
        _validate_generation_defaults(spec.generation_defaults, context=f"models[{spec.id}] generation_defaults")

    if enabled_count == 0:
        raise ConfigError("At least one model must be enabled")
    if default_model_id not in by_id:
        raise ConfigError(f"default_model_id {default_model_id!r} is not in models")
    if not by_id[default_model_id].enabled:
        raise ConfigError(f"default_model_id {default_model_id!r} refers to a disabled model")


def _validate_generation_defaults(defaults: GenerationDefaults, *, context: str) -> None:
    if defaults.max_tokens is not None:
        _require_range(f"{context}.max_tokens", defaults.max_tokens, min_value=1)
    if defaults.temperature is not None:
        _require_range(f"{context}.temperature", defaults.temperature, min_value=0.0, max_value=2.0)
    if defaults.top_k is not None:
        _require_range(f"{context}.top_k", defaults.top_k, min_value=1)
    if defaults.top_p is not None:
        _require_range(f"{context}.top_p", defaults.top_p, min_value=0.0, max_value=1.0, min_open=True)
    if defaults.repetition_penalty is not None:
        _require_range(f"{context}.repetition_penalty", defaults.repetition_penalty, min_value=1.0)
    if defaults.repetition_context_size is not None:
        _require_range(f"{context}.repetition_context_size", defaults.repetition_context_size, min_value=1)


def _coerce_bool(name: str, raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    val = str(raw).strip().lower()
    if val in {"1", "true", "yes", "on"}:
        return True
    if val in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"Invalid boolean for {name!r}: {raw!r}")


def _coerce_int(name: str, raw: Any) -> int:
    if isinstance(raw, bool):
        raise ConfigError(f"Invalid boolean for integer field {name!r}: {raw!r}")
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Invalid integer for {name!r}: {raw!r}") from exc


def _coerce_float(name: str, raw: Any) -> float:
    if isinstance(raw, bool):
        raise ConfigError(f"Invalid boolean for float field {name!r}: {raw!r}")
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Invalid float for {name!r}: {raw!r}") from exc


def _require_range(
    name: str,
    value: Any,
    *,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    min_open: bool = False,
) -> None:
    if min_value is not None:
        if min_open and not (value > min_value):
            raise ConfigError(f"{name} must be > {min_value}; got {value}")
        if not min_open and not (value >= min_value):
            raise ConfigError(f"{name} must be >= {min_value}; got {value}")
    if max_value is not None and not (value <= max_value):
        raise ConfigError(f"{name} must be <= {max_value}; got {value}")
