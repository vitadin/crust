import http.client
import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any

from lfm25_server.config import ModelSpec, ServerConfig
from lfm25_server.handler import make_handler
from lfm25_server.model_manager import (
    BusyError,
    GenerateResult,
    ModelDisabledError,
    ModelExecutionError,
    UnknownModelError,
)


class _FakeManager:
    def __init__(self) -> None:
        self.raise_busy = False
        self.raise_model_error = False
        self.last_messages = None
        self.last_kwargs = None
        self.default_model_id = "lfm2.5"
        self.active_model_id = None
        self.model_switch_count = 0
        self.known_models = {"lfm2.5"}

    def resolve_model_id(self, requested_model: str | None) -> str:
        if requested_model is None or requested_model.strip() == "":
            return self.default_model_id
        if requested_model == "disabled":
            raise ModelDisabledError("Model is disabled: disabled")
        if requested_model not in self.known_models:
            raise UnknownModelError(f"Unknown model: {requested_model}")
        return requested_model

    def get_prompt_model_id(self, model_id: str) -> str:
        if model_id not in self.known_models:
            raise UnknownModelError(f"Unknown model: {model_id}")
        return "lfm2.5"

    def set_default_model(self, model_id: str) -> str:
        if model_id == "disabled":
            raise ModelDisabledError("Model is disabled: disabled")
        if model_id not in self.known_models:
            raise UnknownModelError(f"Unknown model: {model_id}")
        self.default_model_id = model_id
        return model_id

    def status(self):
        return SimpleNamespace(
            model_loaded=self.active_model_id is not None,
            model_state="LOADED" if self.active_model_id else "UNLOADED",
            idle_s=1.2 if self.active_model_id else None,
            idle_timeout_s=300.0,
            load_count=1,
            queue_depth=0,
            inference_in_progress=False,
            last_error=None,
            uptime_s=12.3,
            default_model_id=self.default_model_id,
            active_model_id=self.active_model_id,
            registered_models_count=1,
            model_switch_count=self.model_switch_count,
        )

    def list_models_status(self):
        return [
            SimpleNamespace(
                model_id="lfm2.5",
                enabled=True,
                backend="mlx",
                prompt_model_id="lfm2.5",
                model_path=".",
                state="LOADED" if self.active_model_id == "lfm2.5" else "UNLOADED",
                loaded=self.active_model_id == "lfm2.5",
                load_count=1,
                last_error=None,
                idle_s=1.1 if self.active_model_id == "lfm2.5" else None,
                ready=True,
                missing_templates=(),
            ),
        ]

    def model_status(self, model_id: str):
        if model_id != "lfm2.5":
            raise UnknownModelError(f"Unknown model: {model_id}")
        return self.list_models_status()[0]

    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> GenerateResult:
        self.last_messages = messages
        self.last_kwargs = kwargs
        if self.raise_busy:
            raise BusyError("server busy: wait queue is full")
        if self.raise_model_error:
            raise ModelExecutionError("mock model failure")

        model_id = kwargs.get("model_id", self.default_model_id)
        switched = self.active_model_id is not None and self.active_model_id != model_id
        if switched:
            self.model_switch_count += 1
        self.active_model_id = model_id

        return GenerateResult(
            answer="fixed text",
            thinking="hidden reasoning",
            raw="<think>hidden reasoning</think>fixed text",
            prompt_tokens=10,
            generation_tokens=4,
            generation_tps=12.5,
            peak_memory_gb=1.2,
            finish_reason="stop",
            cold_start=False,
            load_time_s=0.0,
            model_id=model_id,
            model_switch=switched,
        )

    def force_load(self, model_id: str | None = None) -> tuple[str, bool, bool, float]:
        mid = self.resolve_model_id(model_id)
        switched = self.active_model_id is not None and self.active_model_id != mid
        if switched:
            self.model_switch_count += 1
        self.active_model_id = mid
        return mid, False, switched, 0.0

    def force_unload(self, model_id: str | None = None) -> tuple[str, bool]:
        mid = self.resolve_model_id(model_id)
        if self.active_model_id != mid:
            return mid, False
        self.active_model_id = None
        return mid, True


class HandlerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = _FakeManager()
        self.config = ServerConfig(
            default_model_id="lfm2.5",
            models=(
                ModelSpec(
                    id="lfm2.5",
                    backend="mlx",
                    model_path=".",
                    prompt_model_id="lfm2.5",
                    enabled=True,
                ),
            ),
            model_path=".",
            prompt_model_id="lfm2.5",
            request_max_body_bytes=64,
        )
        handler = make_handler(self.manager, self.config)
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.host, self.port = self.httpd.server_address

    def tearDown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=1.0)

    def _request(self, method: str, path: str, body: Any = None, headers: dict[str, str] | None = None):
        conn = http.client.HTTPConnection(self.host, self.port, timeout=2.0)
        payload = None
        hdrs = headers or {}
        if body is not None:
            payload = json.dumps(body).encode("utf-8")
            hdrs.setdefault("Content-Type", "application/json")
        conn.request(method, path, body=payload, headers=hdrs)
        resp = conn.getresponse()
        raw = resp.read()
        conn.close()
        data = json.loads(raw.decode("utf-8")) if raw else {}
        return resp.status, data

    def test_status_shape(self) -> None:
        status, data = self._request("GET", "/status")
        self.assertEqual(status, 200)
        self.assertIn("queue_depth", data)
        self.assertIn("model_state", data)
        self.assertIn("uptime_s", data)
        self.assertIn("default_model_id", data)
        self.assertIn("active_model_id", data)

    def test_models_endpoint(self) -> None:
        status, data = self._request("GET", "/models")
        self.assertEqual(status, 200)
        self.assertEqual(data["default_model_id"], "lfm2.5")
        self.assertEqual(len(data["models"]), 1)
        self.assertEqual(data["models"][0]["id"], "lfm2.5")

    def test_generate_prompt_success(self) -> None:
        status, data = self._request("POST", "/generate", {"prompt": "fix this"})
        self.assertEqual(status, 200)
        self.assertEqual(data["answer"], "fixed text")
        self.assertIsNone(data["thinking"])
        self.assertEqual(data["model"], "lfm2.5")

    def test_generate_unknown_model(self) -> None:
        status, data = self._request("POST", "/generate", {"prompt": "fix this", "model": "unknown"})
        self.assertEqual(status, 400)
        self.assertEqual(data["error"]["code"], "unknown_model")

    def test_generate_invalid_messages(self) -> None:
        status, data = self._request("POST", "/generate", {"messages": "bad"})
        self.assertEqual(status, 400)
        self.assertEqual(data["error"]["code"], "validation_error")

    def test_body_size_limit(self) -> None:
        oversized = {"prompt": "x" * 512}
        status, data = self._request("POST", "/generate", oversized)
        self.assertEqual(status, 413)
        self.assertEqual(data["error"]["code"], "validation_error")

    def test_busy_maps_to_429(self) -> None:
        self.manager.raise_busy = True
        status, data = self._request("POST", "/generate", {"prompt": "fix this"})
        self.assertEqual(status, 429)
        self.assertEqual(data["error"]["code"], "busy")

    def test_correct_route(self) -> None:
        status, data = self._request(
            "POST",
            "/correct",
            {"text": "i has a pen", "mode": "grammar", "language": "en"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(data["corrected_text"], "fixed text")
        self.assertEqual(data["model"], "lfm2.5")
        self.assertIsNotNone(self.manager.last_messages)
        self.assertEqual(self.manager.last_messages[0]["role"], "system")
        self.assertIn("for en", self.manager.last_messages[0]["content"])
        self.assertIn("Correct grammar", self.manager.last_messages[0]["content"])

    def test_control_default_model(self) -> None:
        status, data = self._request("POST", "/control/default-model", {"model": "lfm2.5"})
        self.assertEqual(status, 200)
        self.assertEqual(data["default_model_id"], "lfm2.5")

    def test_control_model_load_route(self) -> None:
        status, data = self._request("POST", "/control/models/lfm2.5/load", {})
        self.assertEqual(status, 200)
        self.assertEqual(data["status"], "loaded")
        self.assertEqual(data["model"], "lfm2.5")


if __name__ == "__main__":
    unittest.main()
