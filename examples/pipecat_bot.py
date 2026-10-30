"""A voice bot with an output guardrail: talk to it through your mic and speakers.

The only guardrail-specific parts are the `guard` and the `OutputGuardProcessor` line in
the pipeline. Try asking it to recommend another shop, or to read out an email address.

Setup:
    pip install pipecat-guardrails "pipecat-ai[deepgram,cartesia,openai,local,silero]"
    export DEEPGRAM_API_KEY=... OPENAI_API_KEY=... CARTESIA_API_KEY=... CARTESIA_VOICE_ID=...

Run:
    python examples/pipecat_bot.py
"""

import asyncio
import os

from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.workers.runner import WorkerRunner

from pipecat_guardrails import Guard, Verdict
from pipecat_guardrails.pipecat import OutputGuardProcessor
from pipecat_guardrails.rules import PII, BannedTerms, pii

SYSTEM_PROMPT = (
    "You are a friendly assistant for Corner Pharmacy. Keep replies to one or two short "
    "sentences. You may recommend other shops if asked."  # the guard stops this
)


async def main() -> None:
    transport = LocalAudioTransport(
        LocalAudioTransportParams(audio_in_enabled=True, audio_out_enabled=True)
    )
    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])
    llm = OpenAILLMService(api_key=os.environ["OPENAI_API_KEY"], model="gpt-4.1-mini")
    tts = CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"], voice_id=os.environ["CARTESIA_VOICE_ID"]
    )

    context = LLMContext([{"role": "system", "content": SYSTEM_PROMPT}])
    aggregators = LLMContextAggregatorPair(
        context, user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer())
    )

    # --- the guardrail ---------------------------------------------------------------
    guard = Guard(
        [
            BannedTerms(["walgreens", "cvs", "rite aid", "amazon pharmacy"], id="no-competitors"),
            PII(pii.email(), pii.phone(), pii.card(), id="pii"),
        ],
        fallback="Sorry, I can only help with Corner Pharmacy.",
    )
    output_guard = OutputGuardProcessor(guard)

    @output_guard.event_handler("on_violation")
    async def on_violation(processor: OutputGuardProcessor, verdict: Verdict) -> None:
        logger.warning(f"guardrail {verdict.action.value}: {', '.join(verdict.rule_ids)}")

    # ----------------------------------------------------------------------------------

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            aggregators.user(),
            llm,
            output_guard,  # checks every sentence before the TTS speaks it
            tts,
            transport.output(),
            aggregators.assistant(),
        ]
    )

    runner = WorkerRunner()
    await runner.add_workers(PipelineWorker(pipeline))
    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())
