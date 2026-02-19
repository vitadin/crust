"""
HTTP server lifecycle.
"""

from __future__ import annotations

import logging
import signal
import threading
from http.server import ThreadingHTTPServer
from socket import socket
from typing import Tuple

from .config import ServerConfig
from .handler import make_handler
from .model_manager import MultiModelManager

logger = logging.getLogger(__name__)


class _LocalThreadingHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address, RequestHandlerClass, *, socket_timeout_s: float, listen_backlog: int):
        self.socket_timeout_s = socket_timeout_s
        self.request_queue_size = listen_backlog
        super().__init__(server_address, RequestHandlerClass)

    def get_request(self) -> Tuple[socket, tuple]:
        request, client_address = super().get_request()
        request.settimeout(self.socket_timeout_s)
        return request, client_address


def run_server(config: ServerConfig, manager: MultiModelManager) -> None:
    handler_cls = make_handler(manager, config)

    with _LocalThreadingHTTPServer(
        (config.host, config.port),
        handler_cls,
        socket_timeout_s=config.socket_timeout_s,
        listen_backlog=config.listen_backlog,
    ) as httpd:
        httpd.daemon_threads = True

        logger.info("Local LLM server listening on http://%s:%d", config.host, config.port)
        logger.info(
            "idle_timeout=%.0fs queue_waiters=%d body_max=%dB",
            config.idle_timeout_s,
            config.max_queue_waiters,
            config.request_max_body_bytes,
        )

        def _signal_shutdown(signum: int, frame: object) -> None:
            logger.info("Signal %d received; initiating graceful shutdown.", signum)
            threading.Thread(target=httpd.shutdown, daemon=True).start()

        signal.signal(signal.SIGTERM, _signal_shutdown)
        signal.signal(signal.SIGINT, _signal_shutdown)

        try:
            httpd.serve_forever()
        finally:
            logger.info("HTTP server stopped. Cleaning up.")
            manager.shutdown()
            logger.info("Shutdown complete.")
