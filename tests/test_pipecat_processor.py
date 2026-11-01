"""Frame-level tests for OutputGuardProcessor, using Pipecat's own test runner."""

import time
from collections.abc import Sequence
from typing import Any

import pytest
from fakes import FakeClassifier
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


def unspoken(text: str) -> TextFrame:
    frame = TextFrame(text)
    frame.skip_tts = True
    return frame


async def test_unspoken_text_frame_is_dropped_if_blocked():
    down = await run(OutputGuardProcessor(make_guard()), [unspoken("Try Acme Corp.")])
    assert not any(isinstance(f, TextFrame) for f in down)


async def test_unspoken_text_frame_is_redacted():
    frame = unspoken("Mail a@b.com")
    frame.append_to_context = False
    down = await run(OutputGuardProcessor(make_guard()), [frame])
    out = [f for f in down if type(f) is TextFrame]
    assert [f.text for f in out] == ["Mail [email]"]
    assert out[0].skip_tts is True
    assert out[0].append_to_context is False


async def test_unspoken_text_frame_passes_unchanged_if_allowed():
    frame = unspoken("All good.")
    down = await run(OutputGuardProcessor(make_guard()), [frame])
    assert down[0] is frame


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


async def test_full_reply_interruption_keeps_function_call_frames():
    call = call_frame()  # Pipecat marks it uninterruptible
    frames = [LLMFullResponseStartFrame(), LLMTextFrame("Let me check. "), call]
    frames += [SleepFrame(sleep=0.05), InterruptionFrame(), SleepFrame(sleep=0.05)]
    down = await run(OutputGuardProcessor(make_guard(), strategy="full_reply"), frames)
    assert call in down
    assert spoken(down) == []  # the held text was interrupted, so it is not spoken


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


# --- concurrent checks with a slow judge (sentence strategy) -----------------------------


def judged(judge: FakeClassifier, **kwargs: Any) -> Guard:
    return Guard(policies={"bad": "Is it bad?"}, judge=judge, fallback=FALLBACK, **kwargs)


def says_bad(state: Any, name: str) -> float:
    return 0.9 if "bad" in str(state).lower() else 0.1


async def test_sentences_are_checked_concurrently_and_released_in_order():
    # The first sentence is the slowest to check, but still comes out first.
    delays = {"One.": 0.15, "Two.": 0.01, "Three.": 0.05}
    judge = FakeClassifier(delay=lambda state: delays[state])
    start = time.perf_counter()
    down = await run(OutputGuardProcessor(judged(judge)), reply("One. Two. Three."))
    elapsed = time.perf_counter() - start
    assert spoken(down) == ["One.", "Two.", "Three."]
    assert judge.max_in_flight == 3
    assert elapsed < sum(delays.values()) + 0.1


async def test_events_fire_in_sentence_order():
    judge = FakeClassifier(delay=lambda state: 0.1 if state == "One." else 0.0)
    processor = OutputGuardProcessor(judged(judge))
    checked: list[str] = []

    @processor.event_handler("on_check")
    async def on_check(proc: OutputGuardProcessor, verdict: Verdict) -> None:  # pyright: ignore[reportUnusedFunction]
        checked.append(verdict.original_text)

    await run(processor, reply("One. Two."))
    assert checked == ["One.", "Two."]


async def test_block_cancels_checks_for_the_rest_of_the_reply():
    judge = FakeClassifier(judge=says_bad, delay=lambda state: 0.1 if state == "Later." else 0.02)
    processor = OutputGuardProcessor(judged(judge))
    checked: list[str] = []

    @processor.event_handler("on_check")
    async def on_check(proc: OutputGuardProcessor, verdict: Verdict) -> None:  # pyright: ignore[reportUnusedFunction]
        checked.append(verdict.original_text)

    down = await run(processor, reply("Fine. Bad idea. Later."))
    assert spoken(down) == ["Fine.", FALLBACK]
    assert checked == ["Fine.", "Bad idea."]
    assert isinstance(down[-1], LLMFullResponseEndFrame)


async def test_reply_after_a_block_is_checked_normally_with_a_judge():
    judge = FakeClassifier(judge=says_bad, delay=0.02)
    frames = reply("Bad. Dropped.") + reply("Hello again.")
    down = await run(OutputGuardProcessor(judged(judge)), frames)
    assert spoken(down) == [FALLBACK, "Hello again."]


async def test_frames_wait_for_earlier_checks():
    judge = FakeClassifier(delay=0.05)
    call = call_frame()
    frames = [LLMFullResponseStartFrame(), LLMTextFrame("Let me check. "), call]
    frames += [LLMTextFrame("Done."), LLMFullResponseEndFrame()]
    down = await run(OutputGuardProcessor(judged(judge)), frames)
    assert [type(f) for f in down] == [
        LLMFullResponseStartFrame,
        AggregatedTextFrame,
        FunctionCallInProgressFrame,
        AggregatedTextFrame,
        LLMFullResponseEndFrame,
    ]


async def test_interruption_cancels_pending_checks():
    judge = FakeClassifier(delay=0.2)
    call = call_frame()
    frames = [LLMFullResponseStartFrame(), LLMTextFrame("Slow one. "), call]
    frames += [SleepFrame(sleep=0.05), InterruptionFrame(), SleepFrame(sleep=0.05)]
    frames += reply("New reply.")
    processor = OutputGuardProcessor(judged(judge))
    start = time.perf_counter()
    down = await run(processor, frames)
    assert spoken(down) == ["New reply."]
    assert call in down  # uninterruptible, so it survives
    assert time.perf_counter() - start < 1.0


async def test_unspoken_text_is_checked_in_order_with_a_judge():
    judge = FakeClassifier(judge=says_bad, delay=0.02)
    frames = [*reply("Hello."), unspoken("Bad note."), unspoken("Good note.")]
    down = await run(OutputGuardProcessor(judged(judge)), frames)
    texts = [f.text for f in down if isinstance(f, TextFrame)]
    assert texts == ["Hello.", "Good note."]


async def test_judge_failure_blocks_with_fallback():
    judge = FakeClassifier(error=RuntimeError("down"))
    down = await run(OutputGuardProcessor(judged(judge)), reply("Hello. There."))
    assert spoken(down) == [FALLBACK]


async def test_text_after_a_block_is_not_checked():
    judge = FakeClassifier(judge=says_bad)
    # "Bad one." is complete as soon as "Mo" arrives, so it is blocked during the pause.
    frames = [LLMFullResponseStartFrame(), LLMTextFrame("Bad one. Mo"), SleepFrame(sleep=0.05)]
    frames += [LLMTextFrame("re text. "), LLMTextFrame("And more."), LLMFullResponseEndFrame()]
    down = await run(OutputGuardProcessor(judged(judge)), frames)
    assert spoken(down) == [FALLBACK]
    assert [state for state, _ in judge.calls] == ["Bad one."]


async def test_full_reply_checks_unspoken_text():
    frames = [unspoken("Try Acme Corp."), unspoken("Mail a@b.com"), unspoken("Fine.")]
    down = await run(OutputGuardProcessor(make_guard(), strategy="full_reply"), frames)
    assert [f.text for f in down if type(f) is TextFrame] == ["Mail [email]", "Fine."]


def broken_fallback(verdict: Verdict) -> str:
    raise RuntimeError("bug in the fallback function")


@pytest.mark.parametrize("strategy", ["sentence", "full_reply"])
async def test_crashing_check_fails_closed(strategy: Any):
    guard = Guard([BannedTerms(["acme corp"], id="no-competitors")], fallback=broken_fallback)
    frames = reply("Try Acme Corp.") + reply("Hello.")
    down = await run(OutputGuardProcessor(guard, strategy=strategy), frames)
    assert spoken(down) == ["Hello."]


async def test_crashing_check_can_fail_open():
    guard = Guard(
        [BannedTerms(["acme corp"], id="no-competitors")],
        fallback=broken_fallback,
        on_error="allow",
    )
    down = await run(OutputGuardProcessor(guard), reply("Try Acme Corp."))
    assert spoken(down) == ["Try Acme Corp."]


async def test_parallel_checks_are_capped_as_a_sliding_window():
    judge = FakeClassifier(delay=0.05)
    sentences = [f"Sentence {i}." for i in range(8)]
    processor = OutputGuardProcessor(judged(judge), max_parallel_checks=3)
    down = await run(processor, reply(" ".join(sentences)))
    assert spoken(down) == sentences
    assert judge.max_in_flight == 3
    assert [state for state, _ in judge.calls] == sentences  # started in order


async def test_default_cap_is_five():
    judge = FakeClassifier(delay=0.05)
    sentences = [f"Sentence {i}." for i in range(8)]
    await run(OutputGuardProcessor(judged(judge)), reply(" ".join(sentences)))
    assert judge.max_in_flight == 5


async def test_one_check_at_a_time():
    judge = FakeClassifier(delay=0.01)
    await run(OutputGuardProcessor(judged(judge), max_parallel_checks=1), reply("A. B. C."))
    assert judge.max_in_flight == 1


async def test_block_cancels_waiting_checks_before_they_call_the_judge():
    judge = FakeClassifier(judge=says_bad, delay=0.05)
    sentences = ["Bad start."] + [f"Sentence {i}." for i in range(6)]
    processor = OutputGuardProcessor(judged(judge), max_parallel_checks=2)
    down = await run(processor, reply(" ".join(sentences)))
    assert spoken(down) == [FALLBACK]
    assert len(judge.calls) == 2  # only the first window was ever sent


def test_invalid_max_parallel_checks():
    with pytest.raises(ValueError, match="max_parallel_checks"):
        OutputGuardProcessor(make_guard(), max_parallel_checks=0)
