"""OutputGuardProcessor: checks what the LLM says before the TTS speaks it."""

# Parts of Pipecat's FrameProcessor API are untyped (**kwargs, event handler args).
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal, cast

from loguru import logger
from pipecat.frames.frames import (
    AggregatedTextFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    SystemFrame,
    TextFrame,
    TTSTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from pipecat_guardrails.guard import Guard
from pipecat_guardrails.pipecat._compat import SENTENCE, text_type_of
from pipecat_guardrails.text.chunker import (
    DEFAULT_ABBREVIATIONS,
    SentenceChunker,
    split_sentences,
)
from pipecat_guardrails.types import Action, CheckContext, CheckError, Verdict, Violation

Strategy = Literal["sentence", "full_reply"]

# Expressive sound and pause units carry a sound name or a duration, not words.
_UNCHECKED_TEXT_TYPES = frozenset({"expression_sound", "expression_pause"})


@dataclass(frozen=True, slots=True)
class _PassThrough:
    """A frame held only to keep its place in the order (full_reply strategy)."""

    frame: Frame


_Held = str | AggregatedTextFrame | _PassThrough


@dataclass(slots=True)
class _Pending:
    """One item waiting to be released, in order (sentence strategy).

    Either a frame that only keeps its place (``frame``), or text being checked
    (``check``): a sentence of the reply (with ``source`` when it arrived as a unit), or an
    unspoken ``skip_tts`` frame (``unspoken``), which is checked on its own.
    """

    reply: int
    text: str = ""
    source: TextFrame | None = None
    frame: Frame | None = None
    skip_tts: bool | None = None
    check: asyncio.Task[Verdict] | None = None
    unspoken: bool = False
    holds_slot: bool = False  # its check took one of the max_parallel_checks slots


class OutputGuardProcessor(FrameProcessor):
    """Checks LLM output with a :class:`Guard` before it reaches the TTS.

    Place it between the LLM and the TTS::

        Pipeline([
            transport.input(), stt, context_aggregator.user(), llm,
            OutputGuardProcessor(guard),
            tts, transport.output(), context_aggregator.assistant(),
        ])

    With the default ``"sentence"`` strategy, the LLM's text is split into sentences and
    each sentence is checked before it is passed on, so nothing unchecked is spoken:

    - **allowed** (or flagged) sentences go to the TTS unchanged;
    - **redacted** sentences go to the TTS with the matches replaced;
    - on a **block**, the rest of the reply is dropped and the guard's fallback line is
      spoken instead (or nothing, if the guard has no fallback). Sentences approved
      earlier in the reply have already been passed on and still play.

    Checks run concurrently: the LLM usually writes faster than the TTS speaks, so while
    one sentence is being checked or spoken, the sentences after it are already being
    checked, up to ``max_parallel_checks`` at a time (a sliding window: when one finishes,
    the next waiting sentence starts). Sentences are always released in order. Usually only
    the first sentence of a reply waits for a full classifier or judge round trip.

    Blocked text never reaches the TTS, and Pipecat builds the assistant's history from
    what the TTS speaks, so blocked text never enters the conversation history.

    The ``"full_reply"`` strategy holds the whole reply until the LLM finishes, checks
    it once, then passes it on sentence by sentence. A block then replaces the entire
    reply. It is safer for rules that need the whole reply, at the cost of latency.

    Frames other than LLM text pass through unchanged and keep their order relative to
    the text: function calls, ``TTSSpeakFrame``s written by your own code, and so on. Text
    held when such a frame arrives is checked and passed on first. Upstream frames and
    system frames (interruptions, metrics) pass straight through.

    If an ``LLMTextProcessor`` runs before this processor, its ``AggregatedTextFrame``s
    are checked as units instead of being split again.

    A plain ``TextFrame`` with ``skip_tts=True`` is not spoken, but it can still reach the
    conversation history and the client app, so it is checked too and dropped if blocked.

    Events:
        on_violation(processor, verdict): a rule or policy fired. Also fires in shadow mode.
        on_check(processor, verdict): every check, including allowed ones.

    Note:
        With RTVI enabled, Pipecat sends raw LLM text to the client app before this
        processor sees it. Use :func:`guarded_rtvi_observer_params` to stop that.
    """

    def __init__(
        self,
        guard: Guard,
        *,
        strategy: Strategy = "sentence",
        context: CheckContext | None = None,
        abbreviations: Iterable[str] = DEFAULT_ABBREVIATIONS,
        max_parallel_checks: int = 5,
        **kwargs: Any,
    ):
        """Create the processor.

        Args:
            guard: The guard that checks each sentence (or the whole reply).
            strategy: ``"sentence"`` (default) or ``"full_reply"``. See the class docs.
            context: Extra information passed to every check.
            abbreviations: Words after which a dot does not end a sentence. See
                :class:`~pipecat_guardrails.text.SentenceChunker`.
            max_parallel_checks: With the ``"sentence"`` strategy, how many sentences may
                be checked at once. More keeps long replies flowing; fewer limits calls to
                the classifier and judge (and the calls wasted when a reply is blocked,
                since checks still waiting are cancelled before they start). ``1`` checks
                one sentence at a time.
            **kwargs: Passed to ``FrameProcessor``.
        """
        super().__init__(**kwargs)
        if strategy not in ("sentence", "full_reply"):
            raise ValueError(f"strategy must be 'sentence' or 'full_reply', not {strategy!r}")
        self._guard = guard
        self._strategy: Strategy = strategy
        self._context = context
        if max_parallel_checks < 1:
            raise ValueError("max_parallel_checks must be at least 1")
        self._abbreviations = frozenset(abbreviations)
        self._check_slots = asyncio.Semaphore(max_parallel_checks)
        self._chunker = SentenceChunker(abbreviations=self._abbreviations)
        self._reply = 0  # id of the reply being received; bumped at each reply start
        self._blocked_reply: int | None = None  # the reply that was blocked; drop its text
        self._in_reply = False  # between LLMFullResponseStartFrame and its end frame
        self._skip_tts: bool | None = None
        # full_reply: the reply so far, in order.
        self._held: list[_Held] = []
        # sentence: checks and frames waiting to be released, in order.
        self._pending: deque[_Pending] = deque()
        self._wakeup = asyncio.Event()
        self._idle = asyncio.Event()
        self._idle.set()
        self._releaser: asyncio.Task[None] | None = None
        self._register_event_handler("on_violation")
        self._register_event_handler("on_check")

    @property
    def guard(self) -> Guard:
        """The guard used for checks."""
        return self._guard

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Check LLM text before passing it on; pass every other frame through."""
        await super().process_frame(frame, direction)

        if isinstance(frame, InterruptionFrame):
            await self._interrupt(frame, direction)
        elif direction is FrameDirection.UPSTREAM:
            await self.push_frame(frame, direction)
        elif isinstance(frame, LLMFullResponseStartFrame):
            await self._flush()
            self._new_reply()
            await self._pass(frame)
            self._in_reply = True
        elif isinstance(frame, LLMTextFrame):
            await self._handle_llm_text(frame)
        elif isinstance(frame, AggregatedTextFrame) and not isinstance(frame, TTSTextFrame):
            await self._handle_unit(frame)
        elif type(frame) is TextFrame and frame.skip_tts:
            await self._handle_unspoken_text(frame)
        elif isinstance(frame, LLMFullResponseEndFrame):
            await self._flush()
            self._in_reply = False
            await self._pass(frame)
            self._new_reply()
        elif isinstance(frame, EndFrame):
            await self._flush()
            await self._idle.wait()
            await self._stop_releasing()
            await self.push_frame(frame, direction)
        elif isinstance(frame, CancelFrame):
            await self._stop_releasing()
            await self.push_frame(frame, direction)
        elif isinstance(frame, SystemFrame):
            await self.push_frame(frame, direction)
        else:
            await self._pass(frame)

    async def cleanup(self) -> None:
        """Cancel checks that are still running."""
        await super().cleanup()
        await self._stop_releasing()

    # --- incoming text -------------------------------------------------------------------

    async def _handle_llm_text(self, frame: LLMTextFrame) -> None:
        if self._is_blocked():
            return
        self._skip_tts = frame.skip_tts
        if self._strategy == "full_reply":
            self._held.append(frame.text)
            return
        for sentence in self._chunker.push(frame.text):
            self._queue_check(sentence)

    async def _handle_unit(self, frame: AggregatedTextFrame) -> None:
        if self._is_blocked():
            return
        if text_type_of(frame) in _UNCHECKED_TEXT_TYPES:
            await self._pass(frame)
            return
        self._skip_tts = frame.skip_tts
        if self._strategy == "full_reply":
            self._held.append(frame)
        else:
            self._queue_check(frame.text, frame)

    async def _handle_unspoken_text(self, frame: TextFrame) -> None:
        """Check a ``skip_tts`` text frame on its own: drop it if blocked."""
        if self._strategy == "sentence":
            self._flush_chunker()
            if text := frame.text.strip():
                self._enqueue_check(_Pending(self._reply, text, frame, unspoken=True))
            else:
                self._enqueue(_Pending(self._reply, frame=frame))
            return
        out = frame
        if frame.text.strip():
            verdict = await self._run_check(frame.text.strip())
            await self._report(verdict)
            out = _unspoken_output(frame, verdict)
        if out is not None:
            await self._pass(out)

    async def _pass(self, frame: Frame) -> None:
        """Pass on a frame that is not checked, without letting it overtake held text."""
        if self._strategy == "full_reply":
            if self._in_reply or self._held:
                self._held.append(_PassThrough(frame))
            else:
                await self.push_frame(frame)
            return
        self._flush_chunker()
        self._enqueue(_Pending(self._reply, frame=frame))

    async def _flush(self) -> None:
        """Check whatever is still held at the end of a reply."""
        if self._strategy == "sentence":
            self._flush_chunker()
        elif self._held:
            held, self._held = self._held, []
            if not self._is_blocked():
                await self._check_full_reply(held)

    def _flush_chunker(self) -> None:
        if (rest := self._chunker.flush()) and not self._is_blocked():
            self._queue_check(rest)

    # --- sentence strategy: concurrent checks, released in order --------------------------

    def _queue_check(self, text: str, source: AggregatedTextFrame | None = None) -> None:
        sentence = text.strip()
        if not sentence or self._is_blocked():
            return
        self._enqueue_check(_Pending(self._reply, sentence, source, skip_tts=self._skip_tts))

    def _enqueue_check(self, item: _Pending) -> None:
        task = self.create_task(self._limited_check(item), "guard-check")
        item.check = cast(asyncio.Task[Verdict], task)
        self._enqueue(item)

    async def _limited_check(self, item: _Pending) -> Verdict:
        # A sliding window over the queue: a sentence keeps its slot until it has been
        # released or dropped (see _free_slot), not just until its check returns. So when
        # a check blocks, the checks waiting behind it are cancelled before they start.
        # Waiters get slots in the order they asked, so earlier sentences go first.
        await self._check_slots.acquire()
        item.holds_slot = True
        return await self._run_check(item.text)

    def _free_slot(self, item: _Pending) -> None:
        if item.holds_slot:
            item.holds_slot = False
            self._check_slots.release()

    def _enqueue(self, item: _Pending) -> None:
        self._pending.append(item)
        self._idle.clear()
        self._wakeup.set()
        if self._releaser is None:
            task = self.create_task(self._release_loop(), "guard-release")
            self._releaser = cast(asyncio.Task[None], task)

    async def _release_loop(self) -> None:
        while True:
            await self._wakeup.wait()
            while self._pending:
                item = self._pending[0]
                try:
                    await self._release(item)
                finally:
                    self._free_slot(item)
                self._pending.popleft()  # only after release, so nothing can overtake it
            self._wakeup.clear()
            self._idle.set()

    async def _release(self, item: _Pending) -> None:
        if item.frame is not None:
            await self.push_frame(item.frame)
            return
        if not item.unspoken and item.reply == self._blocked_reply:
            return  # an earlier sentence of this reply was blocked
        verdict = await self._result(item)
        if verdict is None:
            return
        await self._report(verdict)
        if item.unspoken:
            assert item.source is not None
            if (out := _unspoken_output(item.source, verdict)) is not None:
                await self.push_frame(out)
        elif verdict.blocked:
            await self._block(item.reply, verdict, item.skip_tts)
        elif item.source is not None and verdict.text == item.text:
            await self.push_frame(item.source)
        else:
            await self._push_text(verdict.text, item.skip_tts)

    async def _result(self, item: _Pending) -> Verdict | None:
        """Wait for an item's check. ``None`` if it was cancelled after a block."""
        assert item.check is not None
        try:
            return await item.check
        except asyncio.CancelledError:
            if item.check.cancelled():
                return None
            raise

    async def _block(self, reply: int, verdict: Verdict, skip_tts: bool | None) -> None:
        self._blocked_reply = reply
        for later in self._pending:  # cancel checks for the rest of this reply
            if later.reply == reply and later.check is not None and not later.unspoken:
                later.check.cancel()
        if reply == self._reply:
            self._chunker.reset()
        if verdict.fallback_text:
            await self._push_text(verdict.fallback_text, skip_tts)

    async def _stop_releasing(self) -> None:
        """Cancel the release loop and every check still running, dropping what is queued."""
        for item in self._pending:
            if item.check is not None:
                item.check.cancel()
            self._free_slot(item)
        self._pending.clear()
        if self._releaser is not None:
            releaser, self._releaser = self._releaser, None
            await self.cancel_task(releaser)
        self._wakeup.clear()
        self._idle.set()

    async def _interrupt(self, frame: InterruptionFrame, direction: FrameDirection) -> None:
        # Frames Pipecat marks uninterruptible (function-call progress and results)
        # must survive an interruption, or the context misses the function call.
        held = [h.frame for h in self._held if isinstance(h, _PassThrough)]
        queued = [p.frame for p in self._pending if p.frame is not None]
        kept = [f for f in held + queued if not getattr(f, "interruptible", True)]
        await self._stop_releasing()
        self._new_reply()
        await self.push_frame(frame, direction)
        for f in kept:
            await self.push_frame(f)

    # --- checking ------------------------------------------------------------------------

    async def _run_check(self, text: str) -> Verdict:
        try:
            return await self._guard.check(text, self._context)
        except Exception as e:
            # Guard.check handles failing rules and classifiers itself, so this is a bug
            # or a broken fallback function. Fail open or closed like the guard would.
            logger.opt(exception=e).error("pipecat-guardrails: check crashed")
            return _crash_verdict(self._guard, text, e)

    async def _report(self, verdict: Verdict) -> None:
        await self._call_event_handler("on_check", verdict)
        if verdict.detected_action is not Action.ALLOW:
            await self._call_event_handler("on_violation", verdict)

    async def _check_full_reply(self, held: list[_Held]) -> None:
        text = "".join(
            h if isinstance(h, str) else f" {h.text} "
            for h in held
            if not isinstance(h, _PassThrough)
        ).strip()
        verdict = None
        if text:
            verdict = await self._run_check(text)
            await self._report(verdict)

        if verdict is None or (not verdict.blocked and verdict.text == text):
            # Allowed: units go on as they came; plain text is split into sentences.
            tokens: list[str] = []
            for h in [*held, None]:
                if isinstance(h, str):
                    tokens.append(h)
                    continue
                for sentence in split_sentences("".join(tokens), abbreviations=self._abbreviations):
                    await self._push_text(sentence.strip(), self._skip_tts)
                tokens = []
                if isinstance(h, _PassThrough):
                    await self.push_frame(h.frame)
                elif h is not None:
                    await self.push_frame(h)
            return

        # Blocked or redacted: the replacement text takes the place of the first text,
        # and frames that are not text keep their positions.
        if verdict.blocked:
            self._blocked_reply = self._reply
            replacement = [verdict.fallback_text] if verdict.fallback_text else []
        else:
            replacement = split_sentences(verdict.text, abbreviations=self._abbreviations)
        for h in held:
            if isinstance(h, _PassThrough):
                await self.push_frame(h.frame)
            elif replacement:
                for sentence in replacement:
                    await self._push_text(sentence.strip(), self._skip_tts)
                replacement = []

    # --- output --------------------------------------------------------------------------

    async def _push_text(self, text: str, skip_tts: bool | None) -> None:
        if not text:
            return
        frame = AggregatedTextFrame(text, SENTENCE)
        frame.skip_tts = skip_tts
        await self.push_frame(frame)

    def _is_blocked(self) -> bool:
        return self._blocked_reply == self._reply

    def _new_reply(self) -> None:
        self._reply += 1
        self._blocked_reply = None
        self._chunker.reset()
        self._held.clear()
        self._in_reply = False
        self._skip_tts = None


def _unspoken_output(frame: TextFrame, verdict: Verdict) -> TextFrame | None:
    """What to pass on for a checked ``skip_tts`` frame: itself, a redacted copy, or nothing."""
    if verdict.blocked:
        return None
    if verdict.text == verdict.original_text:
        return frame
    redacted = TextFrame(verdict.text)
    redacted.skip_tts = frame.skip_tts
    redacted.append_to_context = frame.append_to_context
    return redacted


def _crash_verdict(guard: Guard, text: str, error: Exception) -> Verdict:
    """A verdict for a check that raised: fail open or closed like the guard does."""
    message = f"{type(error).__name__}: {error}"
    failed_open = guard.on_error == "allow"
    if failed_open:
        return Verdict(
            Action.ALLOW,
            Action.ALLOW,
            text,
            text,
            errors=(CheckError("guard", message, failed_open=True),),
        )
    fallback = guard.fallback if isinstance(guard.fallback, str) else None
    return Verdict(
        Action.BLOCK,
        Action.BLOCK,
        fallback or "",
        text,
        violations=(Violation("guard", Action.BLOCK, f"check failed: {message}", error=True),),
        errors=(CheckError("guard", message, failed_open=False),),
        fallback_text=fallback,
    )
