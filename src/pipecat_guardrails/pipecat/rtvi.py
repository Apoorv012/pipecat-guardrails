"""Keep unchecked LLM text away from RTVI clients."""

from __future__ import annotations

from typing import Any

from pipecat.processors.frameworks.rtvi.observer import RTVIObserverParams


def guarded_rtvi_observer_params(**kwargs: Any) -> RTVIObserverParams:
    """RTVI observer settings that don't send unchecked LLM text to the client.

    By default, Pipecat's RTVI observer sends every LLM token to the client app as a
    ``bot-llm-text`` message (and, when enabled, ``llm-raw-text``) as soon as the LLM
    produces it, before :class:`OutputGuardProcessor` has checked it. A client that shows
    or logs those messages would display blocked text.

    These settings turn both off. Clients still get ``bot-output`` and TTS text messages,
    which come from the output transport, after the guard. ``bot-llm-started`` and
    ``bot-llm-stopped`` are also turned off, because Pipecat controls them with the same
    switch.

    Example::

        worker = PipelineWorker(
            pipeline,
            rtvi_observer_params=guarded_rtvi_observer_params(),
        )

    Args:
        **kwargs: Any other ``RTVIObserverParams`` fields.
    """
    params = RTVIObserverParams(**kwargs)
    params.bot_llm_enabled = False
    if hasattr(params, "llm_raw_text_enabled"):  # Pipecat 1.13+
        params.llm_raw_text_enabled = False
    return params
