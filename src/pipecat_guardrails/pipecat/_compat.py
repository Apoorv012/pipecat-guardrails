"""Small shims for differences between Pipecat 1.12 and later versions."""

from __future__ import annotations

from pipecat.frames.frames import AggregatedTextFrame

try:  # Pipecat 1.13+
    from pipecat.utils.text.base_text_aggregator import TextType
except ImportError:  # pragma: no cover - Pipecat 1.12 named it AggregationType
    from pipecat.utils.text.base_text_aggregator import (  # type: ignore[no-redef]
        AggregationType as TextType,  # pyright: ignore[reportAttributeAccessIssue]
    )

SENTENCE = TextType.SENTENCE


def text_type_of(frame: AggregatedTextFrame) -> str:
    """The frame's text type as a string (``text_type`` since 1.13, ``aggregated_by`` before)."""
    value = getattr(frame, "text_type", None)
    if value is None:  # pragma: no cover - Pipecat 1.12
        value = getattr(frame, "aggregated_by", "")
    return str(value)
