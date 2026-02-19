"""
HTTP request handler for the local multi-model inference server.
"""

from __future__ import annotations

import json
import logging
import time
from http.server import BaseHTTPRequestHandler
from typing import TYPE_CHECKING, Any, Optional

from .model_manager import (
    BusyError,
    ModelDisabledError,
    ModelExecutionError,
    ModelNotReadyError,
    UnknownModelError,
)
from .model_router import ModelRouter
from .prompt_store import PromptError, PromptStore

if TYPE_CHECKING:
    from .config import ServerConfig
    from .model_manager import MultiModelManager

logger = logging.getLogger(__name__)

_VALID_ROLES = {"system", "user", "assistant", "tool"}
_VALID_CORRECT_MODES = {"grammar", "rewrite", "style"}


def make_handler(manager: "MultiModelManager", config: "ServerConfig") -> type:
    prompt_store = PromptStore()
    model_router = ModelRouter(manager)

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            logger.debug("HTTP %s %s -> %s", self.command, self.path, fmt % args)

        def log_error(self, fmt: str, *args: Any) -> None:
            logger.warning("HTTP error: " + fmt, *args)

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path == "/health":
                self._handle_health()
            elif path == "/status":
                self._handle_status()
            elif path == "/models":
                self._handle_models()
            elif path.startswith("/models/"):
                self._handle_model_details(path)
            else:
                self._send_error_payload(404, "validation_error", f"No route for GET {self.path}")

        def do_POST(self) -> None:
            path = self.path.split("?", 1)[0]
            body = self._read_json_body()
            if body is None:
                return

            if path == "/generate":
                self._handle_generate(body)
            elif path in ("/v1/chat/completions", "/chat/completions"):
                self._handle_chat_completions(body)
            elif path == "/correct":
                self._handle_correct(body)
            elif path == "/security/check":
                self._handle_security_check(body)
            elif path == "/control/load":
                self._handle_control_load()
            elif path == "/control/unload":
                self._handle_control_unload()
            elif path == "/control/default-model":
                self._handle_control_default_model(body)
            elif path.startswith("/control/models/") and path.endswith("/load"):
                self._handle_control_model_load(path)
            elif path.startswith("/control/models/") and path.endswith("/unload"):
                self._handle_control_model_unload(path)
            else:
                self._send_error_payload(404, "validation_error", f"No route for POST {self.path}")

        # ---- GET handlers ----------------------------------------------------

        def _handle_health(self) -> None:
            info = manager.status()
            self._send_json(200, {
                "status": "ok",
                "model_loaded": info.model_loaded,
                "model_state": info.model_state,
                "default_model_id": info.default_model_id,
                "active_model_id": info.active_model_id,
            })

        def _handle_status(self) -> None:
            info = manager.status()
            self._send_json(200, {
                "model_loaded": info.model_loaded,
                "model_state": info.model_state,
                "idle_s": round(info.idle_s, 1) if info.idle_s is not None else None,
                "idle_timeout_s": info.idle_timeout_s,
                "load_count": info.load_count,
                "queue_depth": info.queue_depth,
                "inference_in_progress": info.inference_in_progress,
                "last_error": info.last_error,
                "uptime_s": round(info.uptime_s, 1),
                "default_model_id": info.default_model_id,
                "active_model_id": info.active_model_id,
                "registered_models_count": info.registered_models_count,
                "model_switch_count": info.model_switch_count,
            })

        def _handle_models(self) -> None:
            info = manager.status()
            model_infos = manager.list_models_status()
            self._send_json(200, {
                "default_model_id": info.default_model_id,
                "active_model_id": info.active_model_id,
                "models": [
                    {
                        "id": m.model_id,
                        "enabled": m.enabled,
                        "backend": m.backend,
                        "prompt_model_id": m.prompt_model_id,
                        "model_path": m.model_path,
                        "state": m.state,
                        "loaded": m.loaded,
                        "load_count": m.load_count,
                        "last_error": m.last_error,
                        "idle_s": round(m.idle_s, 1) if m.idle_s is not None else None,
                        "ready": m.ready,
                        "missing_templates": list(m.missing_templates),
                    }
                    for m in model_infos
                ],
            })

        def _handle_model_details(self, path: str) -> None:
            model_id = path.removeprefix("/models/").strip("/")
            if not model_id:
                self._send_error_payload(400, "validation_error", "Model id is required")
                return
            try:
                m = manager.model_status(model_id)
                current = manager.status()
            except UnknownModelError as exc:
                self._send_error_payload(400, "unknown_model", str(exc))
                return
            self._send_json(200, {
                "id": m.model_id,
                "enabled": m.enabled,
                "backend": m.backend,
                "prompt_model_id": m.prompt_model_id,
                "model_path": m.model_path,
                "state": m.state,
                "loaded": m.loaded,
                "load_count": m.load_count,
                "last_error": m.last_error,
                "idle_s": round(m.idle_s, 1) if m.idle_s is not None else None,
                "ready": m.ready,
                "missing_templates": list(m.missing_templates),
                "default_model_id": current.default_model_id,
                "active_model_id": current.active_model_id,
            })

        # ---- POST inference handlers ----------------------------------------

        def _handle_generate(self, body: dict[str, Any]) -> None:
            messages = self._extract_messages(body)
            if messages is None:
                return

            model_id = self._resolve_request_model(body)
            if model_id is None:
                return

            include_thinking = self._read_optional_bool(
                body.get("include_thinking", config.include_thinking),
                field_name="include_thinking",
            )
            if include_thinking is None:
                return

            gen_kwargs = _extract_gen_kwargs(body)
            if gen_kwargs is None:
                self._send_error_payload(400, "validation_error", "Invalid generation parameters")
                return

            result = self._run_generate(model_id, messages, gen_kwargs)
            if result is None:
                return

            self._send_json(200, {
                "model": result.model_id,
                "answer": result.answer,
                "thinking": result.thinking if include_thinking else None,
                "meta": _result_meta(result),
                "cold_start": result.cold_start,
                "model_switch": result.model_switch,
                "load_time_s": round(result.load_time_s, 3),
            })

        def _handle_chat_completions(self, body: dict[str, Any]) -> None:
            messages = body.get("messages")
            if not isinstance(messages, list) or not messages:
                self._send_error_payload(400, "validation_error", "'messages' must be a non-empty list")
                return
            valid_messages = _validate_messages(messages)
            if valid_messages is None:
                self._send_error_payload(400, "validation_error", "Invalid message shape")
                return

            model_id = self._resolve_request_model(body)
            if model_id is None:
                return

            include_thinking = self._read_optional_bool(
                body.get("include_thinking", config.include_thinking),
                field_name="include_thinking",
            )
            if include_thinking is None:
                return

            gen_kwargs = _extract_gen_kwargs(body)
            if gen_kwargs is None:
                self._send_error_payload(400, "validation_error", "Invalid generation parameters")
                return

            result = self._run_generate(model_id, valid_messages, gen_kwargs)
            if result is None:
                return

            content = result.raw if include_thinking else result.answer
            self._send_json(200, {
                "id": f"chatcmpl-{int(time.time() * 1000)}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": result.model_id,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": result.finish_reason,
                }],
                "usage": {
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.generation_tokens,
                    "total_tokens": result.prompt_tokens + result.generation_tokens,
                },
                "x_thinking": result.thinking if include_thinking else None,
                "x_peak_memory_gb": round(result.peak_memory_gb, 3),
                "x_cold_start": result.cold_start,
                "x_model_switch": result.model_switch,
            })

        def _handle_correct(self, body: dict[str, Any]) -> None:
            text = body.get("text")
            if not isinstance(text, str) or not text.strip():
                self._send_error_payload(400, "validation_error", "'text' must be a non-empty string")
                return

            mode = body.get("mode", "grammar")
            if not isinstance(mode, str) or mode not in _VALID_CORRECT_MODES:
                self._send_error_payload(400, "validation_error", f"'mode' must be one of {sorted(_VALID_CORRECT_MODES)}")
                return

            language = body.get("language", "en")
            if not isinstance(language, str) or not language.strip():
                self._send_error_payload(400, "validation_error", "'language' must be a non-empty string")
                return

            model_id = self._resolve_request_model(body)
            if model_id is None:
                return

            include_thinking = self._read_optional_bool(
                body.get("include_thinking", config.include_thinking),
                field_name="include_thinking",
            )
            if include_thinking is None:
                return

            gen_kwargs = _extract_gen_kwargs(body)
            if gen_kwargs is None:
                self._send_error_payload(400, "validation_error", "Invalid generation parameters")
                return

            try:
                prompt_model_id = manager.get_prompt_model_id(model_id)
                system_prompt = prompt_store.render_correction_prompt(
                    prompt_model_id=prompt_model_id,
                    mode=mode,
                    language=language,
                )
            except PromptError as exc:
                self._send_error_payload(500, "capability_error", str(exc))
                return
            except (UnknownModelError, ModelDisabledError) as exc:
                self._send_error_payload(400, "unknown_model", str(exc))
                return

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ]
            result = self._run_generate(model_id, messages, gen_kwargs)
            if result is None:
                return

            payload: dict[str, Any] = {
                "model": result.model_id,
                "corrected_text": result.answer,
                "meta": _result_meta(result),
                "cold_start": result.cold_start,
                "model_switch": result.model_switch,
                "load_time_s": round(result.load_time_s, 3),
            }
            if include_thinking:
                payload["thinking"] = result.thinking
            self._send_json(200, payload)

        def _handle_security_check(self, body: dict[str, Any]) -> None:
            tool_name = body.get("tool_name")
            if not isinstance(tool_name, str) or not tool_name.strip():
                self._send_error_payload(400, "validation_error", "'tool_name' must be a non-empty string")
                return

            arguments = body.get("arguments")
            if arguments is None:
                self._send_error_payload(400, "validation_error", "'arguments' is required")
                return
            if not isinstance(arguments, (str, dict)):
                self._send_error_payload(400, "validation_error", "'arguments' must be a string or object")
                return

            model_id = self._resolve_request_model(body)
            if model_id is None:
                return

            include_thinking = self._read_optional_bool(
                body.get("include_thinking", config.include_thinking),
                field_name="include_thinking",
            )
            if include_thinking is None:
                return

            try:
                prompt_model_id = manager.get_prompt_model_id(model_id)
                system_prompt = prompt_store.render_security_check_prompt(
                    prompt_model_id=prompt_model_id,
                )
            except PromptError as exc:
                self._send_error_payload(500, "capability_error", str(exc))
                return
            except (UnknownModelError, ModelDisabledError) as exc:
                self._send_error_payload(400, "unknown_model", str(exc))
                return

            # Format the user message as tool_name + arguments JSON
            if isinstance(arguments, dict):
                args_str = json.dumps(arguments, ensure_ascii=False)
            else:
                args_str = arguments
            user_content = json.dumps({"tool_name": tool_name, "arguments": args_str}, ensure_ascii=False)

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]

            gen_kwargs: dict[str, Any] = {"max_tokens": 256, "temperature": 0.1}
            result = self._run_generate(model_id, messages, gen_kwargs)
            if result is None:
                return

            # Parse verdict from first line of answer
            answer = result.answer.strip() if result.answer else ""
            lines = answer.splitlines()
            first_line = lines[0].strip().upper() if lines else ""
            reason = " ".join(lines[1:]).strip() if len(lines) > 1 else ""

            if first_line == "BLOCK":
                verdict = "block"
                risk_level = "high"
            elif first_line == "ALLOW":
                verdict = "allow"
                risk_level = "low"
            else:
                # Model didn't output a clear verdict — fail open at LFM layer
                logger.warning("LFM security check: unexpected verdict line %r — defaulting to allow", first_line)
                verdict = "allow"
                risk_level = "unknown"
                reason = f"(parse failure: {answer[:100]!r})"

            payload: dict[str, Any] = {
                "verdict": verdict,
                "risk_level": risk_level,
                "reason": reason,
                "model": result.model_id,
                "meta": _result_meta(result),
                "cold_start": result.cold_start,
                "load_time_s": round(result.load_time_s, 3),
            }
            if include_thinking and result.thinking:
                payload["thinking"] = result.thinking
            self._send_json(200, payload)

        # ---- POST control handlers ------------------------------------------

        def _handle_control_load(self) -> None:
            try:
                model_id, cold_start, model_switch, load_time = manager.force_load()
            except BusyError as exc:
                self._send_error_payload(429, "busy", str(exc))
                return
            except (UnknownModelError, ModelDisabledError) as exc:
                self._send_error_payload(400, "unknown_model", str(exc))
                return
            except Exception as exc:
                logger.exception("Force-load failed")
                self._send_error_payload(500, "internal_error", str(exc))
                return
            self._send_json(200, {
                "status": "loaded",
                "model": model_id,
                "cold_start": cold_start,
                "model_switch": model_switch,
                "load_time_s": round(load_time, 3),
            })

        def _handle_control_unload(self) -> None:
            try:
                model_id, unloaded = manager.force_unload()
            except BusyError as exc:
                self._send_error_payload(429, "busy", str(exc))
                return
            except (UnknownModelError, ModelDisabledError) as exc:
                self._send_error_payload(400, "unknown_model", str(exc))
                return
            except Exception as exc:
                logger.exception("Force-unload failed")
                self._send_error_payload(500, "internal_error", str(exc))
                return
            self._send_json(200, {"status": "unloaded", "model": model_id, "unloaded": unloaded})

        def _handle_control_default_model(self, body: dict[str, Any]) -> None:
            requested = body.get("model")
            if not isinstance(requested, str) or not requested.strip():
                self._send_error_payload(400, "validation_error", "'model' must be a non-empty string")
                return
            try:
                model_id = manager.set_default_model(requested.strip())
            except UnknownModelError as exc:
                self._send_error_payload(400, "unknown_model", str(exc))
                return
            except ModelDisabledError as exc:
                self._send_error_payload(409, "model_disabled", str(exc))
                return
            self._send_json(200, {"status": "ok", "default_model_id": model_id})

        def _handle_control_model_load(self, path: str) -> None:
            model_id = path.removeprefix("/control/models/").removesuffix("/load").strip("/")
            if not model_id:
                self._send_error_payload(400, "validation_error", "Model id is required")
                return
            try:
                model_id, cold_start, model_switch, load_time = manager.force_load(model_id)
            except BusyError as exc:
                self._send_error_payload(429, "busy", str(exc))
                return
            except UnknownModelError as exc:
                self._send_error_payload(400, "unknown_model", str(exc))
                return
            except ModelDisabledError as exc:
                self._send_error_payload(400, "model_disabled", str(exc))
                return
            except Exception as exc:
                logger.exception("Model load failed")
                self._send_error_payload(500, "internal_error", str(exc))
                return
            self._send_json(200, {
                "status": "loaded",
                "model": model_id,
                "cold_start": cold_start,
                "model_switch": model_switch,
                "load_time_s": round(load_time, 3),
            })

        def _handle_control_model_unload(self, path: str) -> None:
            model_id = path.removeprefix("/control/models/").removesuffix("/unload").strip("/")
            if not model_id:
                self._send_error_payload(400, "validation_error", "Model id is required")
                return
            try:
                model_id, unloaded = manager.force_unload(model_id)
            except BusyError as exc:
                self._send_error_payload(429, "busy", str(exc))
                return
            except UnknownModelError as exc:
                self._send_error_payload(400, "unknown_model", str(exc))
                return
            except ModelDisabledError as exc:
                self._send_error_payload(400, "model_disabled", str(exc))
                return
            except Exception as exc:
                logger.exception("Model unload failed")
                self._send_error_payload(500, "internal_error", str(exc))
                return
            self._send_json(200, {
                "status": "unloaded",
                "model": model_id,
                "unloaded": unloaded,
            })

        # ---- helpers ---------------------------------------------------------

        def _run_generate(self, model_id: str, messages: list[dict[str, str]], gen_kwargs: dict[str, Any]):
            try:
                return manager.generate(messages, model_id=model_id, **gen_kwargs)
            except BusyError as exc:
                self._send_error_payload(429, "busy", str(exc))
            except UnknownModelError as exc:
                self._send_error_payload(400, "unknown_model", str(exc))
            except ModelDisabledError as exc:
                self._send_error_payload(400, "model_disabled", str(exc))
            except ModelNotReadyError as exc:
                self._send_error_payload(500, "model_not_ready", str(exc))
            except ModelExecutionError as exc:
                self._send_error_payload(500, "model_error", str(exc))
            except Exception as exc:
                logger.exception("Unhandled generation failure")
                self._send_error_payload(500, "internal_error", str(exc))
            return None

        def _resolve_request_model(self, body: dict[str, Any]) -> Optional[str]:
            requested = body.get("model")
            if requested is not None and not isinstance(requested, str):
                self._send_error_payload(400, "validation_error", "'model' must be a string")
                return None
            try:
                return model_router.resolve(requested)
            except UnknownModelError as exc:
                self._send_error_payload(400, "unknown_model", str(exc))
            except ModelDisabledError as exc:
                self._send_error_payload(400, "model_disabled", str(exc))
            return None

        def _extract_messages(self, body: dict[str, Any]) -> Optional[list[dict[str, str]]]:
            if "messages" in body:
                msgs = body["messages"]
                if not isinstance(msgs, list) or not msgs:
                    self._send_error_payload(400, "validation_error", "'messages' must be a non-empty list")
                    return None
                valid = _validate_messages(msgs)
                if valid is None:
                    self._send_error_payload(400, "validation_error", "Invalid message shape")
                    return None
                return valid

            if "prompt" in body:
                prompt = body["prompt"]
                if not isinstance(prompt, str) or not prompt.strip():
                    self._send_error_payload(400, "validation_error", "'prompt' must be a non-empty string")
                    return None
                messages = []
                system = body.get("system")
                if system is not None:
                    if not isinstance(system, str):
                        self._send_error_payload(400, "validation_error", "'system' must be a string")
                        return None
                    if system.strip():
                        messages.append({"role": "system", "content": system})
                messages.append({"role": "user", "content": prompt})
                return messages

            self._send_error_payload(400, "validation_error", "Request must include 'messages' or 'prompt'")
            return None

        def _read_json_body(self) -> Optional[dict[str, Any]]:
            content_length = self.headers.get("Content-Length")
            if content_length is None:
                length = 0
            else:
                try:
                    length = int(content_length)
                except ValueError:
                    self._send_error_payload(400, "validation_error", "Invalid Content-Length header")
                    return None

            if length < 0:
                self._send_error_payload(400, "validation_error", "Invalid Content-Length header")
                return None
            if length > config.request_max_body_bytes:
                self._send_error_payload(413, "validation_error", "Request body too large")
                return None
            if length == 0:
                return {}

            content_type = self.headers.get("Content-Type", "")
            if "application/json" not in content_type.lower():
                self._send_error_payload(415, "validation_error", "Content-Type must be application/json")
                return None

            try:
                raw = self.rfile.read(length)
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                self._send_error_payload(400, "validation_error", f"Invalid JSON body: {exc}")
                return None
            except Exception as exc:
                self._send_error_payload(400, "validation_error", f"Failed to read request body: {exc}")
                return None

            if not isinstance(obj, dict):
                self._send_error_payload(400, "validation_error", "JSON body must be an object")
                return None
            return obj

        def _read_optional_bool(self, value: Any, *, field_name: str) -> Optional[bool]:
            if isinstance(value, bool):
                return value
            self._send_error_payload(400, "validation_error", f"'{field_name}' must be a boolean")
            return None

        def _send_error_payload(self, status: int, code: str, message: str) -> None:
            self._send_json(status, {"error": {"code": code, "message": message}})

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                self.wfile.write(body)
            except BrokenPipeError:
                logger.debug("Client disconnected before response could be written.")

    return _Handler


def _validate_messages(messages: list[Any]) -> Optional[list[dict[str, str]]]:
    validated: list[dict[str, str]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            return None
        role = msg.get("role")
        content = msg.get("content")
        if not isinstance(role, str) or role not in _VALID_ROLES:
            return None
        if not isinstance(content, str):
            return None
        validated.append({"role": role, "content": content})
    return validated


def _extract_gen_kwargs(body: dict[str, Any]) -> Optional[dict[str, Any]]:
    kwargs: dict[str, Any] = {}
    specs = {
        "max_tokens": (int, 1, None),
        "top_k": (int, 1, None),
        "repetition_context_size": (int, 1, None),
        "temperature": (float, 0.0, 2.0),
        "top_p": (float, 0.0, 1.0),
        "repetition_penalty": (float, 1.0, None),
    }

    for key, (typ, min_v, max_v) in specs.items():
        if key not in body:
            continue
        val = body[key]
        if isinstance(val, bool):
            return None
        if typ is int:
            if not isinstance(val, int):
                return None
        else:
            if not isinstance(val, (int, float)):
                return None
            val = float(val)

        if min_v is not None:
            if key == "top_p":
                if not (val > min_v):
                    return None
            elif val < min_v:
                return None
        if max_v is not None and val > max_v:
            return None
        kwargs[key] = val
    return kwargs


def _result_meta(result: Any) -> dict[str, Any]:
    return {
        "prompt_tokens": result.prompt_tokens,
        "generation_tokens": result.generation_tokens,
        "generation_tps": round(result.generation_tps, 2),
        "peak_memory_gb": round(result.peak_memory_gb, 3),
        "finish_reason": result.finish_reason,
        "model_switch": result.model_switch,
    }
