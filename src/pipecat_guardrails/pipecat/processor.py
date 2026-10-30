"""OutputGuardProcessor: checks what the LLM says before the TTS speaks it."""

# Parts of Pipecat's FrameProcessor API are untyped (**kwargs, event handler args).
# pyright: reportUnknownMemberType=false

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

from pipecat.frames.frames import (
    AggregatedTextFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    SystemFrame,
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
from pipecat_guardrails.types import Action, CheckContext, Verdict

Strategy = Literal["sentence", "full_reply"]

# Expressive sound and pause units carry a sound name or a duration, not words.
_UNCHECKED_TEXT_TYPES = frozenset({"expression_sound", "expression_pause"})


@dataclass(frozen=True, slots=True)
class _PassThrough:
    """A frame held only to keep its place in the order (full_reply strategy)."""

    frame: Frame


_Held = str | AggregatedTextFrame | _PassThrough


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

    Events:
        on_violation(processor, verdict): a rule fired. Also fires in shadow mode.
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
        **kwargs: Any,
    ):
        """Create the processor.

        Args:
            guard: The guard that checks each sentence (or the whole reply).
            strategy: ``"sentence"`` (default) or ``"full_reply"``. See the class docs.
            context: Extra information passed to every check.
            abbreviations: Words after which a dot does not end a sentence. See
                :class:`~pipecat_guardrails.text.SentenceChunker`.
            **kwargs: Passed to ``FrameProcessor``.
        """
        super().__init__(**kwargs)
        if strategy not in ("sentence", "full_reply"):
            raise ValueError(f"strategy must be 'sentence' or 'full_reply', not {strategy!r}")
        self._guard = guard
        self._strategy: Strategy = strategy
        self._context = context
        self._abbreviations = frozenset(abbreviations)
        self._chunker = SentenceChunker(abbreviations=self._abbreviations)
        self._held: list[_Held] = []  # full_reply: the reply so far, in order
        self._in_reply = False  # between LLMFullResponseStartFrame and its end frame
        self._blocked = False  # the current reply was blocked; drop the rest of it
        self._skip_tts: bool | None = None
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
            self._reset()
            await self.push_frame(frame, direction)
        elif direction is FrameDirection.UPSTREAM:
            await self.push_frame(frame, direction)
        elif isinstance(frame, LLMFullResponseStartFrame):
            self._reset()
            self._in_reply = True
            await self.push_frame(frame, direction)
        elif isinstance(frame, LLMTextFrame):
            await self._handle_llm_text(frame)
        elif isinstance(frame, AggregatedTextFrame) and not isinstance(frame, TTSTextFrame):
            await self._handle_unit(frame)
        elif isinstance(frame, LLMFullResponseEndFrame):
            await self._flush()
            self._reset()
            await self.push_frame(frame, direction)
        elif isinstance(frame, EndFrame):
            await self._flush()
            await self.push_frame(frame, direction)
        elif isinstance(frame, SystemFrame):
            await self.push_frame(frame, direction)
        else:
            await self._handle_other(frame)

    # --- incoming text -------------------------------------------------------------------

    async def _handle_llm_text(self, frame: LLMTextFrame) -> None:
        if self._blocked:
            return
        self._skip_tts = frame.skip_tts
        if self._strategy == "full_reply":
            self._held.append(frame.text)
            return
        for sentence in self._chunker.push(frame.text):
            if self._blocked:
                return
            await self._check_and_push(sentence)

    async def _handle_unit(self, frame: AggregatedTextFrame) -> None:
        if self._blocked:
            return
        if text_type_of(frame) in _UNCHECKED_TEXT_TYPES:
            await self._handle_other(frame)
            return
        self._skip_tts = frame.skip_tts
        if self._strategy == "full_reply":
            self._held.append(frame)
            return
        await self._check_and_push(frame.text, frame)

    async def _handle_other(self, frame: Frame) -> None:
        """Pass on a frame that is not checked, without letting it overtake held text."""
        if self._strategy == "full_reply" and (self._in_reply or self._held):
            self._held.append(_PassThrough(frame))
            return
        if not self._blocked and (rest := self._chunker.flush()):
            await self._check_and_push(rest)
        await self.push_frame(frame)

    async def _flush(self) -> None:
        """Check whatever is still held at the end of a reply."""
        if self._blocked:
            return
        if self._strategy == "full_reply":
            held, self._held = self._held, []
            await self._check_full_reply(held)
        elif rest := self._chunker.flush():
            await self._check_and_push(rest)

    # --- checking ------------------------------------------------------------------------

    async def _check(self, text: str) -> Verdict:
        verdict = await self._guard.check(text, self._context)
        await self._call_event_handler("on_check", verdict)
        if verdict.detected_action is not Action.ALLOW:
            await self._call_event_handler("on_violation", verdict)
        return verdict

    async def _check_and_push(self, text: str, source: AggregatedTextFrame | None = None) -> None:
        sentence = text.strip()
        if not sentence:
            return
        verdict = await self._check(sentence)
        if verdict.blocked:
            await self._block(verdict)
        elif source is not None and verdict.text == sentence:
            await self.push_frame(source)
        else:
            await self._push_text(verdict.text)

    async def _check_full_reply(self, held: list[_Held]) -> None:
        text = "".join(
            h if isinstance(h, str) else f" {h.text} "
            for h in held
            if not isinstance(h, _PassThrough)
        ).strip()
        verdict = await self._check(text) if text else None

        if verdict is None or (not verdict.blocked and verdict.text == text):
            # Allowed: units go on as they came; plain text is split into sentences.
            tokens: list[str] = []
            for h in [*held, None]:
                if isinstance(h, str):
                    tokens.append(h)
                    continue
                for sentence in split_sentences("".join(tokens), abbreviations=self._abbreviations):
                    await self._push_text(sentence.strip())
                tokens = []
                if isinstance(h, _PassThrough):
                    await self.push_frame(h.frame)
                elif h is not None:
                    await self.push_frame(h)
            return

        # Blocked or redacted: the replacement text takes the place of the first text,
        # and frames that are not text keep their positions.
        if verdict.blocked:
            self._blocked = True
            replacement = [verdict.fallback_text] if verdict.fallback_text else []
        else:
            replacement = split_sentences(verdict.text, abbreviations=self._abbreviations)
        for h in held:
            if isinstance(h, _PassThrough):
                await self.push_frame(h.frame)
            elif replacement:
                for sentence in replacement:
                    await self._push_text(sentence.strip())
                replacement = []

    async def _block(self, verdict: Verdict) -> None:
        self._blocked = True
        self._chunker.reset()
        self._held.clear()
        if verdict.fallback_text:
            await self._push_text(verdict.fallback_text)

    # --- output --------------------------------------------------------------------------

    async def _push_text(self, text: str) -> None:
        if not text:
            return
        frame = AggregatedTextFrame(text, SENTENCE)
        frame.skip_tts = self._skip_tts
        await self.push_frame(frame)

    def _reset(self) -> None:
        self._chunker.reset()
        self._held.clear()
        self._in_reply = False
        self._blocked = False
        self._skip_tts = None
