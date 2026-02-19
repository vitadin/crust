"""
Prompt storage and rendering for model-specific prompt templates.
"""

from __future__ import annotations

import re
from pathlib import Path

_REQUIRED_TEMPLATES = (
    "correction/grammar.txt",
    "correction/rewrite.txt",
    "correction/style.txt",
)


class PromptError(RuntimeError):
    """Raised when a prompt template cannot be loaded or rendered."""


class PromptStore:
    def __init__(self, base_dir: Path | None = None) -> None:
        self._base_dir = base_dir or (Path(__file__).resolve().parent / "prompts")

    def render_correction_prompt(self, *, prompt_model_id: str, mode: str, language: str) -> str:
        template = self._read_template(
            prompt_model_id=prompt_model_id,
            task="correction",
            name=f"{mode}.txt",
        )
        try:
            rendered = template.format(language=language)
        except KeyError as exc:
            raise PromptError(f"Prompt template is missing placeholder: {exc}") from exc
        return rendered.strip()

    def render_security_check_prompt(self, *, prompt_model_id: str) -> str:
        template = self._read_template(
            prompt_model_id=prompt_model_id,
            task="security",
            name="check.txt",
        )
        return template.strip()

    def missing_required_templates(self, prompt_model_id: str) -> tuple[str, ...]:
        safe_id = _normalize_prompt_model_id(prompt_model_id)
        missing: list[str] = []
        for rel in _REQUIRED_TEMPLATES:
            path = self._base_dir / safe_id / rel
            if not path.exists():
                missing.append(rel)
        return tuple(missing)

    def _read_template(self, *, prompt_model_id: str, task: str, name: str) -> str:
        safe_id = _normalize_prompt_model_id(prompt_model_id)
        path = self._base_dir / safe_id / task / name
        try:
            content = path.read_text(encoding="utf-8").strip()
        except FileNotFoundError as exc:
            raise PromptError(f"Prompt template not found: {path}") from exc
        except OSError as exc:
            raise PromptError(f"Failed to read prompt template {path}: {exc}") from exc

        if not content:
            raise PromptError(f"Prompt template is empty: {path}")
        return content


def _normalize_prompt_model_id(prompt_model_id: str) -> str:
    raw = prompt_model_id.strip().lower()
    if not raw:
        raise PromptError("prompt model id is empty")

    raw = re.sub(r"[\s/\\]+", "-", raw)
    raw = re.sub(r"[^a-z0-9._-]", "-", raw)
    raw = re.sub(r"-{2,}", "-", raw).strip("-")
    if not raw:
        raise PromptError(f"Invalid prompt model id: {prompt_model_id!r}")
    return raw
