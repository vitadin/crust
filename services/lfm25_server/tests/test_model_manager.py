import threading
import time
import unittest
from unittest.mock import patch

from lfm25_server.config import ModelSpec, ServerConfig
from lfm25_server.model_backend import BackendGenerateResult, LoadedModel
from lfm25_server.model_manager import BusyError, ModelManager


class _FakeBackend:
    name = "mlx"

    def __init__(self) -> None:
        self.loaded_paths: list[str] = []
        self.unload_count = 0

    def load(self, model_path: str) -> LoadedModel:
        self.loaded_paths.append(model_path)
        return LoadedModel(model={"path": model_path}, tokenizer={"fake": True})

    def generate(self, loaded: LoadedModel, messages: list[dict], params) -> BackendGenerateResult:
        _ = loaded
        _ = messages
        _ = params
        return BackendGenerateResult(
            raw_text="final answer",
            prompt_tokens=8,
            generation_tokens=4,
            generation_tps=10.0,
            peak_memory_gb=1.1,
            finish_reason="stop",
        )

    def unload(self, loaded: LoadedModel) -> None:
        _ = loaded
        self.unload_count += 1


def _make_config(*, max_queue_waiters: int = 0, generation_wait_timeout_s: float = 0.2) -> ServerConfig:
    models = (
        ModelSpec(
            id="m1",
            backend="mlx",
            model_path=".",
            prompt_model_id="lfm2.5",
            enabled=True,
        ),
        ModelSpec(
            id="m2",
            backend="mlx",
            model_path=".",
            prompt_model_id="lfm2.5",
            enabled=True,
        ),
    )
    return ServerConfig(
        default_model_id="m1",
        models=models,
        model_path=".",
        prompt_model_id="lfm2.5",
        generation_wait_timeout_s=generation_wait_timeout_s,
        max_queue_waiters=max_queue_waiters,
    )


class ModelManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = _FakeBackend()
        self.patch = patch("lfm25_server.model_manager.get_backend", return_value=self.backend)
        self.patch.start()

    def tearDown(self) -> None:
        self.patch.stop()

    def test_queue_full_raises_busy(self) -> None:
        manager = ModelManager(_make_config(max_queue_waiters=0))
        manager._gen_lock.acquire()
        try:
            with self.assertRaises(BusyError):
                manager._acquire_generation_slot()
        finally:
            manager._gen_lock.release()

    def test_status_reports_queue_depth(self) -> None:
        manager = ModelManager(_make_config(max_queue_waiters=2, generation_wait_timeout_s=0.5))
        manager._gen_lock.acquire()
        thread_done = threading.Event()

        def _waiter() -> None:
            try:
                manager._acquire_generation_slot()
                manager._gen_lock.release()
            finally:
                thread_done.set()

        t = threading.Thread(target=_waiter, daemon=True)
        t.start()
        time.sleep(0.05)

        info = manager.status()
        self.assertEqual(info.queue_depth, 1)

        manager._gen_lock.release()
        thread_done.wait(timeout=1.0)
        t.join(timeout=1.0)

    def test_model_switch_updates_status(self) -> None:
        manager = ModelManager(_make_config(max_queue_waiters=1, generation_wait_timeout_s=0.5))
        first = manager.generate([{"role": "user", "content": "hello"}], model_id="m1")
        second = manager.generate([{"role": "user", "content": "world"}], model_id="m2")

        self.assertFalse(first.model_switch)
        self.assertTrue(second.model_switch)
        info = manager.status()
        self.assertEqual(info.active_model_id, "m2")
        self.assertEqual(info.model_switch_count, 1)

    def test_force_load_and_unload_specific_model(self) -> None:
        manager = ModelManager(_make_config(max_queue_waiters=1, generation_wait_timeout_s=0.5))
        model_id, cold_start, model_switch, _ = manager.force_load("m1")
        self.assertEqual(model_id, "m1")
        self.assertTrue(cold_start)
        self.assertFalse(model_switch)

        unloaded_model, unloaded = manager.force_unload("m1")
        self.assertEqual(unloaded_model, "m1")
        self.assertTrue(unloaded)


if __name__ == "__main__":
    unittest.main()
