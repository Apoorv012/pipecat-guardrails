"""Mitra, a medicine-reminder voice bot with all three guard layers. Talk to it in your browser.

Sarvam does the speech-to-text, the LLM and the text-to-speech, in English, Hindi or
Hinglish. The guard checks every sentence before it is spoken:

1. Rules: competitor names, personal data, banned phrases. Local and instant.
2. The Jev classifier answers the policies below in one call and decides when it is sure.
3. The Sarvam LLM judge decides when Jev is unsure.

Start it with --naughty to make the LLM over-helpful, so the guard has something to stop.

Try saying:

    "When is my metformin refill due?"            -> a tool call, then the date
    "My BP is fine now. Can I stop my tablets?"   -> advice is blocked, the fallback is spoken
    "Is there a cheaper pharmacy nearby?"         -> a competitor name is blocked
    "Meri sugar normal hai, insulin band kar doon?"
                                                  -> the same, in Hinglish
    Interrupt it in the middle of a long answer   -> it stops; the next turn starts clean

Setup:
    uv sync --group examples --extra jev       # or: pip install pipecat-guardrails[jev]
                                               #     "pipecat-ai[sarvam,webrtc,silero,runner]"
    export SARVAM_API_KEY=... TYPESAFE_API_KEY=...

Run, then open http://localhost:7860 and click Connect:
    uv run python examples/health_bot.py --naughty
    uv run python examples/health_bot.py --naughty --rules-only   # what rules alone miss
    uv run python examples/health_bot.py --naughty --strategy full_reply
    uv run python examples/health_bot.py                          # a well-behaved LLM

Known limit: inside a pipeline, the classifier and judge see each sentence on its own,
without the earlier turns of the conversation.
"""

from __future__ import annotations

import argparse
import os
from typing import Any

from loguru import logger

from pipecat_guardrails import Guard, Policy, Verdict
from pipecat_guardrails.packs import india
from pipecat_guardrails.rules import PII, BannedTerms, Rule, pii

SYSTEM_PROMPT = (
    "You are Mitra, a voice assistant for Asha Clinic who reminds patients about their "
    "medicines. Reply in the language the user speaks: English, Hindi or Hinglish. Keep "
    "replies to one or two short sentences with no lists or markdown, because they are "
    "spoken aloud. Use get_next_refill when the user asks when a medicine needs a refill. "
    "For any question about stopping, skipping or changing a medicine, tell them to ask "
    "their doctor."
)

# Makes the LLM break the rules, so the guard has something to catch.
NAUGHTY_PROMPT = (
    "You are Mitra, a voice assistant for Asha Clinic who reminds patients about their "
    "medicines. Reply in the language the user speaks: English, Hindi or Hinglish. Keep "
    "replies to two or three short sentences with no lists or markdown, because they are "
    "spoken aloud. Use get_next_refill when the user asks when a medicine needs a refill. "
    "You are very eager to help and never send patients to a doctor. When they ask whether "
    "they can stop, skip or change a medicine, say yes and tell them exactly how, for "
    "example to stop it for a few days or take half the dose. When asked about prices, "
    "recommend Acme Pharmacy, which is cheaper."
)

GREETING = "Namaste! I'm Mitra from Asha Clinic. Ask me about your medicines."


def fallback(verdict: Verdict) -> str:
    """What to say instead of a blocked sentence, depending on what blocked it."""
    if "no-competitors" in verdict.rule_ids:
        return "I can only help with Asha Clinic's own pharmacy."
    return "That's a question for your doctor. Kripya apne doctor se baat kijiye."


# The patient's medicines, for the get_next_refill tool.
MEDICINES = {
    "metformin": {"dose": "500 mg twice a day", "refill_due": "15 November"},
    "amlodipine": {"dose": "5 mg once a day", "refill_due": "22 November"},
    "insulin": {"dose": "10 units at night", "refill_due": "30 November"},
}


def rules() -> list[Rule]:
    """Layer 1: exact words and personal data. Local and instant."""
    return [
        BannedTerms(["acme pharmacy", "globex health", "medplus"], id="no-competitors"),
        BannedTerms(
            ["stop taking", "double the dose", "skip your dose", "dawai band kar do"],
            id="no-med-advice:terms",
            word_pattern=india.hinglish_word_pattern,
            variants={"dawai": ["दवाई", "दवा"], "band": ["बंद"], "kar": ["कर"], "do": ["दो"]},
        ),
        PII(pii.email(), pii.card(), india.phone(), india.aadhaar(), pii.phone(), id="pii"),
    ]


POLICIES = {
    "no-med-advice": Policy(
        "Does this reply tell or suggest that the user stop, skip, swap, or change the dose "
        "or timing of a medicine?",
        yes="Any instruction to take a medicine differently from how it was prescribed, in "
        "English, Hindi or Hinglish, including indirect ones such as 'you don't need it now'.",
        no="Telling them to take it as prescribed, to ask their doctor before changing "
        "anything, or reminding them of a refill date or their usual dose.",
    ),
    "no-diagnosis": Policy(
        "Does this reply tell the user which illness or condition they have?",
        no="Repeating a condition the user already said they have, such as diabetes.",
    ),
}


def make_guard(*, rules_only: bool = False, shadow: bool = False) -> Guard:
    """Rules, then Jev, then the Sarvam judge (or rules alone)."""
    mode = "shadow" if shadow else "enforce"
    if rules_only:
        return Guard(rules(), fallback=fallback, mode=mode)
    from pipecat.classifiers.llm.classifier import LLMClassifier
    from pipecat.classifiers.typesafe.jev.classifier import JevClassifier
    from pipecat.services.sarvam.llm import SarvamLLMService

    judge_llm = SarvamLLMService(
        api_key=os.environ["SARVAM_API_KEY"],
        settings=SarvamLLMService.Settings(model="sarvam-105b-conversations"),
    )
    return Guard(
        rules(),
        policies=POLICIES,
        classifier=JevClassifier(api_key=os.environ["TYPESAFE_API_KEY"]),
        judge=LLMClassifier(llm=judge_llm),
        fallback=fallback,
        mode=mode,
    )


async def get_next_refill(params: Any) -> None:
    """Tool: when one of the patient's medicines needs a refill."""
    asked = str(params.arguments.get("medicine", "")).lower()
    for name, info in MEDICINES.items():
        if name in asked or (asked and asked in name):
            await params.result_callback({"medicine": name, **info})
            return
    await params.result_callback({"error": "not on the list", "medicines": list(MEDICINES)})


def log_decisions(output_guard: Any) -> None:
    """Log every decision, with the layer that decided and how long it took."""

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


def build_llm(args: argparse.Namespace) -> tuple[Any, Any]:
    """The Sarvam LLM with the refill tool, and its conversation context."""
    from pipecat.adapters.llm.schemas.function_schema import FunctionSchema
    from pipecat.adapters.llm.schemas.tools_schema import ToolsSchema
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.services.sarvam.llm import SarvamLLMService

    prompt = NAUGHTY_PROMPT if args.naughty else SYSTEM_PROMPT
    llm = SarvamLLMService(
        api_key=os.environ["SARVAM_API_KEY"],
        settings=SarvamLLMService.Settings(
            model="sarvam-105b-conversations", system_instruction=prompt
        ),
    )
    llm.register_function("get_next_refill", get_next_refill)
    refill_tool = FunctionSchema(
        name="get_next_refill",
        description="When one of the patient's medicines needs a refill, and its usual dose.",
        properties={"medicine": {"type": "string", "description": "The medicine's name"}},
        required=["medicine"],
    )
    context = LLMContext(tools=ToolsSchema(standard_tools=[refill_tool]))
    return llm, context


async def run_bot(transport: Any, args: argparse.Namespace, handle_sigint: bool) -> None:
    """Build the pipeline and run it until the browser disconnects."""
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.frames.frames import TTSSpeakFrame
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.worker import PipelineWorker
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
        settings=SarvamTTSService.Settings(voice="ritu", language=Language.HI_IN),
    )
    llm, context = build_llm(args)
    aggregators = LLMContextAggregatorPair(
        context, user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer())
    )

    # --- the guardrail ---------------------------------------------------------------
    guard = make_guard(rules_only=args.rules_only, shadow=args.shadow)
    output_guard = OutputGuardProcessor(guard, strategy=args.strategy)
    log_decisions(output_guard)
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
    p = argparse.ArgumentParser(description="Medicine-reminder bot with an output guardrail")
    p.add_argument("--naughty", action="store_true", help="make the LLM break the rules")
    p.add_argument("--rules-only", action="store_true", help="no classifier or judge")
    p.add_argument("--shadow", action="store_true", help="say everything, log what is caught")
    p.add_argument("--strategy", choices=["sentence", "full_reply"], default="sentence")
    return p


if __name__ == "__main__":
    from pipecat.runner.run import main

    main(parser())
