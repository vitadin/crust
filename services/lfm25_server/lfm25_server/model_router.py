"""
Request-time model routing helper.
"""

from __future__ import annotations


class ModelRouter:
    def __init__(self, manager) -> None:
        self._manager = manager

    def resolve(self, requested_model: str | None) -> str:
        """Resolve optional request model to an enabled model id."""
        return self._manager.resolve_model_id(requested_model)
