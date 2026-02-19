import unittest
from pathlib import Path

from lfm25_server.prompt_store import PromptError, PromptStore


class PromptStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        base = Path(__file__).resolve().parent.parent / "lfm25_server" / "prompts"
        self.store = PromptStore(base_dir=base)

    def test_render_correction_prompt_uses_model_specific_template(self) -> None:
        text = self.store.render_correction_prompt(
            prompt_model_id="LFM2.5",
            mode="grammar",
            language="en",
        )
        self.assertIn("for en", text)
        self.assertIn("Correct grammar", text)

    def test_missing_prompt_raises(self) -> None:
        with self.assertRaises(PromptError):
            self.store.render_correction_prompt(
                prompt_model_id="unknown-model",
                mode="grammar",
                language="en",
            )

    def test_required_template_check(self) -> None:
        missing = self.store.missing_required_templates("lfm2.5")
        self.assertEqual(missing, ())


if __name__ == "__main__":
    unittest.main()
