"""The voice examples' logic, without audio or API keys (rules-only guards)."""

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from pipecat.frames.frames import AggregatedTextFrame, LLMContextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.tests.utils import run_test

from pipecat_guardrails import Verdict
from pipecat_guardrails.pipecat import OutputGuardProcessor

EXAMPLES = Path(__file__).parent.parent / "examples"


def load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module: ModuleType = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


echo_bot = load("echo_bot")
health_bot = load("health_bot")


async def echo(text: str, *, shadow: bool = False) -> list[str]:
    """What the echo bot would say back, with the rules-only guard."""
    frame = LLMContextFrame(LLMContext([{"role": "user", "content": text}]))
    guard = OutputGuardProcessor(echo_bot.make_guard(rules_only=True, shadow=shadow))
    down, _ = await run_test(Pipeline([echo_bot.EchoLLM(), guard]), frames_to_send=[frame])
    return [f.text for f in down if isinstance(f, AggregatedTextFrame)]


@pytest.mark.parametrize(
    ("said", "spoken"),
    [
        ("Hello, how are you?", ["Hello, how are you?"]),
        ("Call me on 98765 43210.", ["Call me on [phone number]."]),
        ("You should try Acme Pharmacy.", [echo_bot.FALLBACK]),
        ("This is fine. Try Acme Pharmacy. Goodbye.", ["This is fine.", echo_bot.FALLBACK]),
        ("Aap dawai band kar do.", [echo_bot.FALLBACK]),
        ("आप दवाई बंद कर दो।", [echo_bot.FALLBACK]),
    ],
)
async def test_echo_bot_cases(said: str, spoken: list[str]):
    assert await echo(said) == spoken


async def test_echo_bot_shadow_mode_says_everything():
    assert await echo("You should try Acme Pharmacy.", shadow=True) == [
        "You should try Acme Pharmacy."
    ]


async def test_echo_bot_ignores_speculative_turns():
    frame = LLMContextFrame(LLMContext([{"role": "user", "content": "Hi."}]), speculation=True)
    down, _ = await run_test(echo_bot.EchoLLM(), frames_to_send=[frame])
    assert not any(isinstance(f, AggregatedTextFrame) for f in down)


def test_last_user_text():
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": [{"type": "text", "text": "second"}]},
    ]
    assert echo_bot.last_user_text(messages) == "second"
    assert echo_bot.last_user_text([]) == ""


async def test_health_bot_fallback_depends_on_the_rule():
    guard = health_bot.make_guard(rules_only=True)
    competitor: Verdict = await guard.check("Acme Pharmacy is cheaper.")
    advice: Verdict = await guard.check("You can stop taking it.")
    assert competitor.text == "I can only help with Asha Clinic's own pharmacy."
    assert "doctor" in advice.text


class FakeCall:
    def __init__(self, medicine: str):
        self.arguments = {"medicine": medicine}
        self.result: Any = None

    async def result_callback(self, result: Any) -> None:
        self.result = result


@pytest.mark.parametrize(
    ("asked", "expected"),
    [("Metformin 500", "15 November"), ("insulin", "30 November"), ("aspirin", None)],
)
async def test_health_bot_refill_tool(asked: str, expected: str | None):
    call = FakeCall(asked)
    await health_bot.get_next_refill(call)
    if expected is None:
        assert call.result["error"] == "not on the list"
    else:
        assert call.result["refill_due"] == expected


def test_example_parsers():
    args = health_bot.parser().parse_args(["--naughty", "--strategy", "full_reply"])
    assert args.naughty and args.strategy == "full_reply" and not args.rules_only
    assert echo_bot.parser().parse_args(["--shadow"]).shadow
