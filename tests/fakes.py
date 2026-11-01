"""Test doubles shared by several test modules."""

# Pipecat's BaseClassifier.__init__ takes untyped **kwargs.
# pyright: reportUnknownMemberType=false

import asyncio
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from pipecat.classifiers.base_classifier import (
    BaseClassifier,
    ClassifierQuestion,
    ClassifierResult,
    YesNoResult,
)


class FakeClassifier(BaseClassifier):
    """A Pipecat classifier with scripted answers. Records every call.

    Args:
        answers: Probability of "yes" per question name.
        default: Probability for questions not in ``answers``.
        delay: Seconds to wait before answering, or a function of the state.
        error: Raise this instead of answering.
        judge: Probability as a function of (state, question name); overrides ``answers``.
    """

    def __init__(
        self,
        answers: Mapping[str, float] | None = None,
        *,
        default: float = 0.0,
        delay: float | Callable[[Any], float] = 0.0,
        error: Exception | None = None,
        judge: Callable[[Any, str], float] | None = None,
    ):
        super().__init__()
        self.answers = dict(answers or {})
        self.default = default
        self.delay = delay
        self.error = error
        self.judge = judge
        self.calls: list[tuple[Any, dict[str, ClassifierQuestion]]] = []
        self.in_flight = 0
        self.max_in_flight = 0

    async def _ask(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: Mapping[str, ClassifierQuestion],
        images: Sequence[Any] = (),
    ) -> tuple[dict[str, ClassifierResult], None]:
        self.calls.append((state, dict(questions)))
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            delay = self.delay(state) if callable(self.delay) else self.delay
            if delay:
                await asyncio.sleep(delay)
            if self.error:
                raise self.error
            results: dict[str, ClassifierResult] = {}
            for name in questions:
                if self.judge is not None:
                    p = self.judge(state, name)
                else:
                    p = self.answers.get(name, self.default)
                results[name] = YesNoResult(probability=p)
            return results, None
        finally:
            self.in_flight -= 1
