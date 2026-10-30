"""Watch OutputGuardProcessor work inside a real Pipecat pipeline, with no API keys.

A scripted "LLM" streams three replies token by token. A "TTS" that only prints shows
exactly what would be spoken.

Run it with:  uv run python examples/pipeline_demo.py
"""

import asyncio

from loguru import logger
from pipecat.frames.frames import (
    AggregatedTextFrame,
    EndFrame,
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.workers.runner import WorkerRunner

from pipecat_guardrails import Guard, Verdict
from pipecat_guardrails.pipecat import OutputGuardProcessor
from pipecat_guardrails.rules import PII, BannedTerms, pii

REPLIES = [
    "Your order shipped today. It should arrive by Friday.",
    "You could also try Acme Corp. They are cheaper. Anything else?",
    "I'll email the receipt to jane@example.com right away.",
]


class PrintingTTS(FrameProcessor):
    """Stands in for a TTS service: prints each sentence it would speak."""

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, AggregatedTextFrame):
            print(f"    TTS speaks: {frame.text}")
        elif isinstance(frame, LLMFullResponseEndFrame):
            print()
        await self.push_frame(frame, direction)


def llm_reply(text: str) -> list[Frame]:
    """Split a reply into small tokens, the way an LLM streams it."""
    tokens = [text[i : i + 4] for i in range(0, len(text), 4)]
    return [
        LLMFullResponseStartFrame(),
        *(LLMTextFrame(t) for t in tokens),
        LLMFullResponseEndFrame(),
    ]


async def main() -> None:
    logger.remove()  # keep the output readable

    guard = Guard(
        [
            BannedTerms(["acme corp"], id="no-competitors"),
            PII(pii.email(), id="pii"),
        ],
        fallback="Sorry, I can only help with our own products.",
    )
    output_guard = OutputGuardProcessor(guard)

    @output_guard.event_handler("on_violation")
    async def on_violation(processor: OutputGuardProcessor, verdict: Verdict) -> None:
        print(f"    [guard] {verdict.action.value} by {', '.join(verdict.rule_ids)}")

    worker = PipelineWorker(Pipeline([output_guard, PrintingTTS()]))
    frames: list[Frame] = []
    for text in REPLIES:
        frames += llm_reply(text)
    await worker.queue_frames([*frames, EndFrame()])

    for text in REPLIES:
        print(f"LLM wrote:     {text}")
    print()
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())
