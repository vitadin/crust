import tempfile
import unittest
from pathlib import Path

from lfm25_server.config import ModelSpec
from lfm25_server.model_registry import CapabilityError, ModelDisabledError, ModelRegistry, UnknownModelError
from lfm25_server.prompt_store import PromptStore


class ModelRegistryTests(unittest.TestCase):
    def test_missing_required_prompt_templates_fail_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            (base / "lfm2.5" / "correction").mkdir(parents=True, exist_ok=True)
            # Only one template exists; two required templates are missing.
            (base / "lfm2.5" / "correction" / "grammar.txt").write_text("x", encoding="utf-8")

            models = (
                ModelSpec(
                    id="m1",
                    backend="mlx",
                    model_path=".",
                    prompt_model_id="lfm2.5",
                    enabled=True,
                ),
            )
            with self.assertRaises(CapabilityError):
                ModelRegistry(models, prompt_store=PromptStore(base_dir=base))

    def test_resolve_enabled_checks_unknown_and_disabled(self) -> None:
        models = (
            ModelSpec(
                id="enabled",
                backend="mlx",
                model_path=".",
                prompt_model_id="lfm2.5",
                enabled=True,
            ),
            ModelSpec(
                id="disabled",
                backend="mlx",
                model_path=".",
                prompt_model_id="lfm2.5",
                enabled=False,
            ),
        )
        registry = ModelRegistry(models)

        self.assertEqual(registry.resolve_enabled("enabled").id, "enabled")
        with self.assertRaises(UnknownModelError):
            registry.resolve_enabled("missing")
        with self.assertRaises(ModelDisabledError):
            registry.resolve_enabled("disabled")


if __name__ == "__main__":
    unittest.main()
