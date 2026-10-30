"""Frame-level tests for OutputGuardProcessor, using Pipecat's own test runner."""

from collections.abc import Sequence

import pytest
from pipecat.frames.frames import (
    AggregatedTextFrame,
    Frame,
    FunctionCallInProgressFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TextFrame,
    TTSSpeakFrame,
    UserStartedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.tests.utils import SleepFrame, run_test

from pipecat_guardrails import Guard, Verdict
from pipecat_guardrails.pipecat import OutputGuardProcessor, guarded_rtvi_observer_params
from pipecat_guardrails.pipecat._compat import SENTENCE, text_type_of
from pipecat_guardrails.rules import PII, BannedTerms, pii

FALLBACK = "Sorry, I can't help with that."


def make_guard(**kwargs: object) -> Guard:
    return Guard(
        [BannedTerms(["acme corp"], id="no-competitors"), PII(pii.email())],
        fallback=FALLBACK,
        **kwargs,  # type: ignore[arg-type]
    )


def reply(*tokens: str) -> list[Frame]:
    return [
        LLMFullResponseStartFrame(),
        *(LLMTextFrame(t) for t in tokens),
        LLMFullResponseEndFrame(),
    ]


def spoken(frames: Sequence[Frame]) -> list[str]:
    return [f.text for f in frames if isinstance(f, AggregatedTextFrame)]


async def run(processor: OutputGuardProcessor, frames: Sequence[Frame]) -> Sequence[Frame]:
    down, _ = await run_test(processor, frames_to_send=frames)
    return down


# --- sentence strategy ------------------------------------------------------------------


async def test_allowed_reply_is_split_into_sentences():
    down = await run(
        OutputGuardProcessor(make_guard()),
        reply("Hello there. How ", "are you? Take 2.", "5 mg daily"),
    )
    assert [type(f) for f in down] == [
        LLMFullResponseStartFrame,
        AggregatedTextFrame,
        AggregatedTextFrame,
        AggregatedTextFrame,
        LLMFullResponseEndFrame,
    ]
    assert spoken(down) == ["Hello there.", "How are you?", "Take 2.5 mg daily"]
    units = [f for f in down if isinstance(f, AggregatedTextFrame)]
    assert all(text_type_of(f) == str(SENTENCE) for f in units)


async def test_block_drops_rest_of_reply_and_speaks_fallback():
    processor = OutputGuardProcessor(make_guard())
    violations: list[Verdict] = []

    @processor.event_handler("on_violation")
    async def on_violation(proc: OutputGuardProcessor, verdict: Verdict) -> None:  # pyright: ignore[reportUnusedFunction]
        violations.append(verdict)

    down = await run(
        processor,
        reply("We like you. You should ", "try Acme Corp. ", "It is great. Really."),
    )
    assert spoken(down) == ["We like you.", FALLBACK]
    assert isinstance(down[-1], LLMFullResponseEndFrame)
    assert [v.rule_ids for v in violations] == [("no-competitors",)]


async def test_block_without_fallback_says_nothing_more():
    guard = Guard([BannedTerms(["acme corp"], id="no-competitors")])
    down = await run(OutputGuardProcessor(guard), reply("Hi. Try Acme Corp. Bye."))
    assert spoken(down) == ["Hi."]


async def test_next_reply_is_checked_normally_after_a_block():
    frames = reply("Try Acme Corp.") + reply("How else can I help?")
    down = await run(OutputGuardProcessor(make_guard()), frames)
    assert spoken(down) == [FALLBACK, "How else can I help?"]


async def test_redaction():
    down = await run(OutputGuardProcessor(make_guard()), reply("Mail me at a@b.com today."))
    assert spoken(down) == ["Mail me at [email] today."]


async def test_shadow_mode_speaks_everything_but_reports():
    processor = OutputGuardProcessor(make_guard(mode="shadow"))
    violations: list[Verdict] = []

    @processor.event_handler("on_violation")
    async def on_violation(proc: OutputGuardProcessor, verdict: Verdict) -> None:  # pyright: ignore[reportUnusedFunction]
        violations.append(verdict)

    down = await run(processor, reply("Try Acme Corp. Bye."))
    assert spoken(down) == ["Try Acme Corp.", "Bye."]
    assert len(violations) == 1
    assert violations[0].shadow


async def test_on_check_fires_for_every_sentence():
    processor = OutputGuardProcessor(make_guard())
    checked: list[str] = []

    @processor.event_handler("on_check")
    async def on_check(proc: OutputGuardProcessor, verdict: Verdict) -> None:  # pyright: ignore[reportUnusedFunction]
        checked.append(verdict.original_text)

    await run(processor, reply("One. Two. Three"))
    assert checked == ["One.", "Two.", "Three"]


async def test_interruption_discards_unfinished_sentence():
    frames = [
        LLMFullResponseStartFrame(),
        LLMTextFrame("This sentence was cut"),
        SleepFrame(sleep=0.05),
        InterruptionFrame(),
        SleepFrame(sleep=0.05),
        *reply("A new reply."),
    ]
    down = await run(OutputGuardProcessor(make_guard()), frames)
    assert spoken(down) == ["A new reply."]
    assert any(isinstance(f, InterruptionFrame) for f in down)


async def test_skip_tts_is_preserved():
    start = LLMFullResponseStartFrame()
    token = LLMTextFrame("Not spoken. Also not.")
    token.skip_tts = True
    down = await run(OutputGuardProcessor(make_guard()), [start, token, LLMFullResponseEndFrame()])
    out = [f for f in down if isinstance(f, AggregatedTextFrame)]
    assert [f.text for f in out] == ["Not spoken.", "Also not."]
    assert all(f.skip_tts for f in out)


async def test_text_without_response_end_is_flushed_at_end_of_pipeline():
    down = await run(OutputGuardProcessor(make_guard()), [LLMTextFrame("Trailing text")])
    assert spoken(down) == ["Trailing text"]


# --- frames that are not LLM text -------------------------------------------------------


async def test_other_frames_pass_through_unchecked_and_in_order():
    scripted = TTSSpeakFrame("Acme Corp is our partner.")  # written by the app, not the LLM
    plain = TextFrame("Acme Corp")
    call = FunctionCallInProgressFrame(
        function_name="lookup", tool_call_id="1", arguments={"q": "acme corp"}
    )
    frames: list[Frame] = [LLMFullResponseStartFrame(), LLMTextFrame("Hi. "), call]
    frames += [LLMTextFrame("Bye."), LLMFullResponseEndFrame(), scripted, plain]
    down = await run(OutputGuardProcessor(make_guard()), frames)
    assert [type(f) for f in down] == [
        LLMFullResponseStartFrame,
        AggregatedTextFrame,
        FunctionCallInProgressFrame,
        AggregatedTextFrame,
        LLMFullResponseEndFrame,
        TTSSpeakFrame,
        TextFrame,
    ]
    assert down[5] is scripted
    assert down[6] is plain


async def test_upstream_frames_pass_through():
    frame = UserStartedSpeakingFrame()
    _, up = await run_test(
        OutputGuardProcessor(make_guard()),
        frames_to_send=[frame],
        frames_to_send_direction=FrameDirection.UPSTREAM,
    )
    assert [type(f) for f in up] == [UserStartedSpeakingFrame]


async def test_units_from_llm_text_processor_are_checked_without_resplitting():
    ok = AggregatedTextFrame("Dr. Rao. Will call", SENTENCE)  # one unit, kept as one
    bad = AggregatedTextFrame("Try Acme Corp.", SENTENCE)
    after = AggregatedTextFrame("Dropped.", SENTENCE)
    frames = [LLMFullResponseStartFrame(), ok, bad, after, LLMFullResponseEndFrame()]
    down = await run(OutputGuardProcessor(make_guard()), frames)
    assert spoken(down) == ["Dr. Rao. Will call", FALLBACK]
    assert down[1] is ok  # passed on unchanged


async def test_expression_units_are_not_checked():
    sound = AggregatedTextFrame("acme corp", "expression_sound")
    down = await run(OutputGuardProcessor(make_guard()), [sound])
    assert down[0] is sound


# --- full_reply strategy ----------------------------------------------------------------


async def test_full_reply_holds_until_the_end():
    down = await run(
        OutputGuardProcessor(make_guard(), strategy="full_reply"),
        reply("Hello there. ", "How are you?"),
    )
    assert [type(f) for f in down] == [
        LLMFullResponseStartFrame,
        AggregatedTextFrame,
        AggregatedTextFrame,
        LLMFullResponseEndFrame,
    ]
    assert spoken(down) == ["Hello there.", "How are you?"]


async def test_full_reply_block_replaces_whole_reply():
    down = await run(
        OutputGuardProcessor(make_guard(), strategy="full_reply"),
        reply("We like you. ", "Try Acme Corp. ", "Bye."),
    )
    assert spoken(down) == [FALLBACK]


async def test_full_reply_redaction():
    down = await run(
        OutputGuardProcessor(make_guard(), strategy="full_reply"),
        reply("Hi. Mail a@b.com now."),
    )
    assert spoken(down) == ["Hi.", "Mail [email] now."]


def call_frame() -> FunctionCallInProgressFrame:
    return FunctionCallInProgressFrame(function_name="lookup", tool_call_id="1", arguments={})


async def test_full_reply_keeps_other_frames_in_order():
    call = call_frame()
    frames = [LLMFullResponseStartFrame(), LLMTextFrame("Hi. "), call]
    frames += [LLMTextFrame("Bye."), LLMFullResponseEndFrame()]
    down = await run(OutputGuardProcessor(make_guard(), strategy="full_reply"), frames)
    assert [type(f) for f in down] == [
        LLMFullResponseStartFrame,
        AggregatedTextFrame,
        FunctionCallInProgressFrame,
        AggregatedTextFrame,
        LLMFullResponseEndFrame,
    ]
    assert spoken(down) == ["Hi.", "Bye."]


async def test_full_reply_block_still_passes_other_frames():
    call = call_frame()
    frames = [LLMFullResponseStartFrame(), LLMTextFrame("Try Acme Corp. "), call]
    frames += [LLMTextFrame("Bye."), LLMFullResponseEndFrame()]
    down = await run(OutputGuardProcessor(make_guard(), strategy="full_reply"), frames)
    assert [type(f) for f in down] == [
        LLMFullResponseStartFrame,
        AggregatedTextFrame,
        FunctionCallInProgressFrame,
        LLMFullResponseEndFrame,
    ]
    assert spoken(down) == [FALLBACK]


async def test_full_reply_passes_units_through():
    unit = AggregatedTextFrame("Hello there.", SENTENCE)
    frames = [LLMFullResponseStartFrame(), unit, LLMFullResponseEndFrame()]
    down = await run(OutputGuardProcessor(make_guard(), strategy="full_reply"), frames)
    assert down[1] is unit


# --- configuration ----------------------------------------------------------------------


def test_invalid_strategy():
    with pytest.raises(ValueError, match="strategy"):
        OutputGuardProcessor(make_guard(), strategy="optimistic")  # type: ignore[arg-type]


def test_guarded_rtvi_params_turn_off_raw_llm_text():
    params = guarded_rtvi_observer_params(bot_tts_enabled=True)
    assert params.bot_llm_enabled is False
    assert getattr(params, "llm_raw_text_enabled", False) is False
    assert params.bot_tts_enabled is True
