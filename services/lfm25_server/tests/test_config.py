import tempfile
import textwrap
import unittest
from pathlib import Path

from lfm25_server.config import ConfigError, load_config


class ConfigTests(unittest.TestCase):
    def _write_yaml(self, data: str) -> str:
        tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        tmp.write(textwrap.dedent(data))
        tmp.flush()
        tmp.close()
        return tmp.name

    def test_invalid_host_rejected(self) -> None:
        path = self._write_yaml(
            """
            host: "0.0.0.0"
            model_path: "."
            """
        )
        with self.assertRaises(ConfigError):
            load_config(path)

    def test_invalid_top_p_rejected(self) -> None:
        path = self._write_yaml(
            """
            model_path: "."
            top_p: 0
            """
        )
        with self.assertRaises(ConfigError):
            load_config(path)

    def test_boolean_for_numeric_field_rejected(self) -> None:
        path = self._write_yaml(
            """
            model_path: "."
            port: true
            """
        )
        with self.assertRaises(ConfigError):
            load_config(path)

    def test_localhost_is_normalized(self) -> None:
        path = self._write_yaml(
            """
            host: "localhost"
            model_path: "."
            """
        )
        cfg = load_config(path)
        self.assertEqual(cfg.host, "127.0.0.1")
        self.assertEqual(len(cfg.models), 1)
        self.assertEqual(cfg.default_model_id, "lfm2.5")
        self.assertTrue(Path(cfg.models[0].model_path).exists())

    def test_duplicate_model_id_rejected(self) -> None:
        path = self._write_yaml(
            """
            default_model_id: "m1"
            models:
              - id: "m1"
                backend: "mlx"
                model_path: "."
                prompt_model_id: "lfm2.5"
              - id: "m1"
                backend: "mlx"
                model_path: "."
                prompt_model_id: "lfm2.5"
            """
        )
        with self.assertRaises(ConfigError):
            load_config(path)

    def test_default_model_must_exist_and_be_enabled(self) -> None:
        path = self._write_yaml(
            """
            default_model_id: "missing"
            models:
              - id: "m1"
                backend: "mlx"
                model_path: "."
                prompt_model_id: "lfm2.5"
            """
        )
        with self.assertRaises(ConfigError):
            load_config(path)

    def test_unsupported_backend_rejected(self) -> None:
        path = self._write_yaml(
            """
            default_model_id: "m1"
            models:
              - id: "m1"
                backend: "unknown"
                model_path: "."
                prompt_model_id: "lfm2.5"
            """
        )
        with self.assertRaises(ConfigError):
            load_config(path)

    def test_empty_prompt_model_id_in_manifest_rejected(self) -> None:
        path = self._write_yaml(
            """
            default_model_id: "m1"
            models:
              - id: "m1"
                backend: "mlx"
                model_path: "."
                prompt_model_id: "   "
            """
        )
        with self.assertRaises(ConfigError):
            load_config(path)

    def test_legacy_single_model_auto_upgrade(self) -> None:
        path = self._write_yaml(
            """
            model_path: "."
            prompt_model_id: "lfm2.5"
            """
        )
        cfg = load_config(path)
        self.assertEqual(len(cfg.models), 1)
        self.assertEqual(cfg.models[0].id, cfg.default_model_id)
        self.assertEqual(cfg.models[0].backend, "mlx")


if __name__ == "__main__":
    unittest.main()
