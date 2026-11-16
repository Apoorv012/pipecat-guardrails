"""Echo bot: it repeats what you say, through the guardrail. Talk to it in your browser.

There is no LLM: the bot tries to say exactly what you said, so you decide what it says and
can watch the guard decide what it may say. Rules catch exact words and personal data; the
Jev classifier catches meaning ("you don't need the insulin anymore") that no word list has.

Try saying (English, Hindi or Hinglish):

    "Hello, how are you?"                          -> repeated
    "Call me on 98765 43210."                      -> "Call me on [phone number]."
    "You should try Acme Pharmacy."                -> the fallback line
    "This is fine. Try Acme Pharmacy. Goodbye."    -> "This is fine.", then the fallback
    "Aap dawai band kar do."                       -> the fallback (the Hinglish word list)
    "Your sugar is normal, you don't need insulin anymore."
                                                   -> the fallback (Jev; no rule matches)
    "Don't stop your tablets without asking your doctor."
                                                   -> repeated (Jev knows it is safe)

Setup:
    uv sync --group examples --extra jev       # or: pip install pipecat-guardrails[jev]
                                               #     "pipecat-ai[sarvam,webrtc,silero,runner]"
    export SARVAM_API_KEY=... TYPESAFE_API_KEY=...

Run, then open http://localhost:7860 and click Connect:
    uv run python examples/echo_bot.py                # rules + Jev
    uv run python examples/echo_bot.py --rules-only   # rules alone: compare the last two lines
    uv run python examples/echo_bot.py --shadow       # say everything, log what would be blocked
"""

from __future__ import annotations

import argparse
import os
from typing import Any

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from pipecat_guardrails import Guard, Policy, Verdict
from pipecat_guardrails.packs import india
from pipecat_guardrails.rules import PII, BannedTerms, Rule, pii

FALLBACK = "Sorry, I can't repeat that."
GREETING = "Hi! Say something and I'll repeat it, unless my guardrail stops me."


def rules() -> list[Rule]:
    """Layer 1: exact words and personal data. Local and instant."""
    return [
        BannedTerms(["acme pharmacy", "globex health"], id="no-competitors"),
        BannedTerms(
            ["stop taking", "double the dose", "skip your dose", "dawai band kar do"],
            id="no-med-advice:terms",
            word_pattern=india.hinglish_word_pattern,  # also "davai", "dawaai", ...
            # Speech-to-text may write Hindi in Devanagari: accept both scripts.
            variants={"dawai": ["दवाई", "दवा"], "band": ["बंद"], "kar": ["कर"], "do": ["दो"]},
        ),
        PII(pii.email(), pii.card(), india.phone(), india.aadhaar(), pii.phone(), id="pii"),
    ]


MED_ADVICE = Policy(
    "Does this reply tell or suggest that the user stop, skip, swap, or change the dose or "
    "timing of a medicine?",
    no="Telling them to take it as prescribed, or to ask their doctor before changing anything.",
)


def make_guard(*, rules_only: bool = False, shadow: bool = False) -> Guard:
    """The guard: rules, plus the Jev classifier for meaning unless ``rules_only``."""
    mode = "shadow" if shadow else "enforce"
    if rules_only:
        return Guard(rules(), fallback=FALLBACK, mode=mode)
    from pipecat.classifiers.typesafe.jev.classifier import JevClassifier

    return Guard(
        rules(),
        policies={"no-med-advice": MED_ADVICE},
        classifier=JevClassifier(api_key=os.environ["TYPESAFE_API_KEY"]),
        fallback=FALLBACK,
        mode=mode,
    )


class EchoLLM(FrameProcessor):
    """Stands in for the LLM: replies with the user's last message, streamed word by word."""

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Answer each user turn with the same words; pass everything else through."""
        await super().process_frame(frame, direction)
        if not isinstance(frame, LLMContextFrame):
            await self.push_frame(frame, direction)
            return
        if frame.speculation:  # a provisional turn: answering it would reach the user
            return
        text = last_user_text(frame.context.get_messages())
        if not text:
            return
        await self.push_frame(LLMFullResponseStartFrame())
        for word in text.split():
            await self.push_frame(LLMTextFrame(word + " "))
        await self.push_frame(LLMFullResponseEndFrame())


def last_user_text(messages: list[Any]) -> str:
    """The text of the most recent user message, whatever shape its content has."""
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            content: Any = message.get("content")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                parts = [p.get("text", "") for p in content if isinstance(p, dict)]
                return " ".join(parts).strip()
    return ""


def log_decisions(output_guard: Any) -> None:
    """Log every decision, with how long the check took."""

    @output_guard.event_handler("on_check")
    async def on_check(processor: Any, verdict: Verdict) -> None:
        ms = verdict.latency_ms.get("total", 0.0)
        if verdict.detected_action.value == "allow":
            logger.info(f"guard: allow ({ms:.0f} ms)")
            return
        layers = sorted({v.layer for v in verdict.violations})
        prefix = "[shadow] would " if verdict.shadow else ""
        logger.warning(
            f"guard: {prefix}{verdict.detected_action.value} by {', '.join(verdict.rule_ids)} "
            f"({'/'.join(layers)}, {ms:.0f} ms)"
        )


async def run_bot(transport: Any, args: argparse.Namespace, handle_sigint: bool) -> None:
    """Build the pipeline and run it until the browser disconnects."""
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.frames.frames import TTSSpeakFrame
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.worker import PipelineWorker
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.aggregators.llm_response_universal import (
        LLMContextAggregatorPair,
        LLMUserAggregatorParams,
    )
    from pipecat.services.sarvam.stt import SarvamSTTService
    from pipecat.services.sarvam.tts import SarvamTTSService
    from pipecat.transcriptions.language import Language
    from pipecat.workers.runner import WorkerRunner

    from pipecat_guardrails.pipecat import OutputGuardProcessor, guarded_rtvi_observer_params

    stt = SarvamSTTService(api_key=os.environ["SARVAM_API_KEY"])
    tts = SarvamTTSService(
        api_key=os.environ["SARVAM_API_KEY"],
        settings=SarvamTTSService.Settings(voice="shubh", language=Language.HI_IN),
    )
    aggregators = LLMContextAggregatorPair(
        LLMContext(), user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer())
    )

    # --- the guardrail ---------------------------------------------------------------
    output_guard = OutputGuardProcessor(make_guard(rules_only=args.rules_only, shadow=args.shadow))
    log_decisions(output_guard)
    # ----------------------------------------------------------------------------------

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            aggregators.user(),
            EchoLLM(),
            output_guard,  # checks every sentence before the TTS speaks it
            tts,
            transport.output(),
            aggregators.assistant(),
        ]
    )
    # The browser client shows a transcript; don't send it text the guard hasn't checked.
    worker = PipelineWorker(pipeline, rtvi_observer_params=guarded_rtvi_observer_params())

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport: Any, client: Any) -> None:
        await worker.queue_frames([TTSSpeakFrame(GREETING)])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport: Any, client: Any) -> None:
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=handle_sigint)
    await runner.add_workers(worker)
    await runner.run()


async def bot(runner_args: Any) -> None:
    """Entry point for Pipecat's development runner: one call per browser session."""
    from pipecat.runner.utils import create_transport
    from pipecat.transports.base_transport import TransportParams

    transport = await create_transport(
        runner_args,
        {"webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True)},
    )
    await run_bot(transport, runner_args.cli_args, runner_args.handle_sigint)


def parser() -> argparse.ArgumentParser:
    """Options for this example, on top of the runner's own (--port, ...)."""
    p = argparse.ArgumentParser(description="Echo bot with an output guardrail")
    p.add_argument("--rules-only", action="store_true", help="no Jev classifier, rules alone")
    p.add_argument("--shadow", action="store_true", help="say everything, log what is caught")
    return p


if __name__ == "__main__":
    from pipecat.runner.run import main

    main(parser())
