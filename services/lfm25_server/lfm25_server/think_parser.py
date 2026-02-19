"""
Parser for <think>...</think> sections.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

_OPEN = "<think>"
_CLOSE = "</think>"


class ThinkResult(NamedTuple):
    thinking: Optional[str]
    answer: str


def parse_think(raw: str) -> ThinkResult:
    """
    Parse one or more leading <think>...</think> blocks.

    Malformed tags are ignored and the full text is treated as answer.
    """
    text = raw.strip()
    if not text.startswith(_OPEN):
        return ThinkResult(thinking=None, answer=text)

    thoughts: list[str] = []
    rest = text
    while rest.startswith(_OPEN):
        end = rest.find(_CLOSE, len(_OPEN))
        if end < 0:
            # Malformed first block -> do not strip anything.
            return ThinkResult(thinking=None, answer=text)
        thought = rest[len(_OPEN):end].strip()
        thoughts.append(thought)
        rest = rest[end + len(_CLOSE):].lstrip()

    thinking = "\n\n".join(thoughts).strip()
    return ThinkResult(thinking=thinking, answer=rest.strip())
