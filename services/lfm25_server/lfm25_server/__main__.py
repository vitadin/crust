"""
Entry point: `python -m lfm25_server`

Startup sequence:
  1. Parse CLI arguments.
  2. Load configuration (YAML file + CLI overrides).
  3. Configure logging.
  4. Create ModelManager (model is NOT loaded yet — lazy loading).
  5. Start idle-monitor daemon thread.
  6. Start HTTP server (blocks until SIGTERM/SIGINT).
"""

from __future__ import annotations

import logging

from .config import ConfigError, build_arg_parser, load_config
from .logging_setup import configure_logging
from .model_manager import CapabilityError, ModelManager
from .model_registry import ModelRegistryError
from .server import run_server


def main() -> None:
    parser = build_arg_parser()
    cli_args = parser.parse_args()

    try:
        config = load_config(yaml_path=cli_args.config, cli_args=cli_args)
    except (ConfigError, CapabilityError, ModelRegistryError) as exc:
        raise SystemExit(f"Configuration error: {exc}") from exc

    configure_logging(config.log_level, json_logs=config.json_logs)
    logger = logging.getLogger(__name__)

    enabled_models = [m.id for m in config.models if m.enabled]
    logger.info("─" * 60)
    logger.info("Local LLM Inference Server  v%s", _version())
    logger.info("  models  : %s", ", ".join(enabled_models))
    logger.info("  default : %s", config.default_model_id)
    logger.info("  address : http://%s:%d", config.host, config.port)
    logger.info("  timeout : %.0f s (idle auto-unload)", config.idle_timeout_s)
    logger.info("─" * 60)

    try:
        manager = ModelManager(config)
    except (CapabilityError, ModelRegistryError) as exc:
        raise SystemExit(f"Startup capability error: {exc}") from exc
    manager.start_idle_monitor()

    run_server(config, manager)


def _version() -> str:
    try:
        from . import __version__
        return __version__
    except ImportError:
        return "unknown"


if __name__ == "__main__":
    main()
