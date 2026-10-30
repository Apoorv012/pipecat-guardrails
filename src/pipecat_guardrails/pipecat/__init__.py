"""Pipecat integration: the processor that checks LLM output before the TTS speaks it."""

from pipecat_guardrails.pipecat.processor import OutputGuardProcessor, Strategy
from pipecat_guardrails.pipecat.rtvi import guarded_rtvi_observer_params

__all__ = ["OutputGuardProcessor", "Strategy", "guarded_rtvi_observer_params"]
