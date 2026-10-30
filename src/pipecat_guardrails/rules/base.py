"""The rule interface (layer 1) and the ``@rule`` decorator for plain functions."""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import cast

from pipecat_guardrails.types import Action, CheckContext, OnError


@dataclass(frozen=True, slots=True)
class Match:
    """A place where a rule fired.

    Parameters:
        start: Start offset in the checked text.
        end: End offset in the checked text.
        reason: Why it fired. Must not contain the matched text (it is logged).
        score: Confidence from 0 to 1, if known.
        replacement: Text to put in place of the span when redacting. ``None`` uses the
            rule's default replacement.
    """

    start: int
    end: int
    reason: str
    score: float | None = None
    replacement: str | None = None


class Rule(ABC):
    """A local, fast check that finds problems in text.

    Subclasses implement :meth:`find`. Rules must be synchronous and quick (well under a
    millisecond per sentence), because they run on every sentence before it is spoken.
    """

    def __init__(
        self,
        *,
        id: str,
        action: Action | str = Action.BLOCK,
        replacement: str = "[redacted]",
        on_error: OnError | None = None,
        enabled: bool = True,
    ):
        """Configure the rule.

        Args:
            id: Unique id, used in verdicts, logs and metrics.
            action: What to do when the rule fires.
            replacement: Default text used in place of a match when ``action`` is REDACT.
            on_error: What to do if :meth:`find` raises. ``None`` uses the guard's setting.
            enabled: Set to False to keep the rule configured but switched off. It can be
                switched on later with :meth:`Guard.enable`.
        """
        if not id:
            raise ValueError("rule id must not be empty")
        self.id = id
        self.action = Action(action)
        self.replacement = replacement
        self.on_error: OnError | None = on_error
        self.enabled = enabled

    @abstractmethod
    def find(self, text: str, context: CheckContext) -> Iterable[Match]:
        """Return every match in ``text``. Offsets refer to ``text`` as given."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(id={self.id!r}, action={self.action.value!r})"


RuleResult = bool | str | Match | Iterable[Match] | None
"""What a function rule may return.

``None``/``False``: no problem. ``True``: the whole text is a problem. A ``str``: the whole
text is a problem, with this reason. One or more :class:`Match`: problems at these spans.
"""

RuleFunction = Callable[[str], RuleResult] | Callable[[str, CheckContext], RuleResult]


class FunctionRule(Rule):
    """A rule backed by a plain function. Usually created with :func:`rule`."""

    def __init__(
        self,
        fn: RuleFunction,
        *,
        id: str,
        action: Action | str = Action.BLOCK,
        replacement: str = "[redacted]",
        on_error: OnError | None = None,
        enabled: bool = True,
    ):
        """Wrap ``fn``, which takes ``(text)`` or ``(text, context)``."""
        super().__init__(
            id=id, action=action, replacement=replacement, on_error=on_error, enabled=enabled
        )
        if len(inspect.signature(fn).parameters) >= 2:
            self._call = cast(Callable[[str, CheckContext], RuleResult], fn)
        else:
            one_arg = cast(Callable[[str], RuleResult], fn)
            self._call: Callable[[str, CheckContext], RuleResult] = lambda text, _: one_arg(text)

    def find(self, text: str, context: CheckContext) -> list[Match]:
        """Call the function and convert its result into matches."""
        return _to_matches(self._call(text, context), text, self.id)


def _to_matches(result: RuleResult, text: str, rule_id: str) -> list[Match]:
    if result is None or result is False:
        return []
    if result is True:
        return [Match(0, len(text), f"rule {rule_id} fired")]
    if isinstance(result, str):
        return [Match(0, len(text), result)]
    if isinstance(result, Match):
        return [result]
    matches = list(result)
    for m in matches:
        if not isinstance(m, Match):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(f"rule {rule_id} returned {type(m).__name__}, expected Match")
    return matches


def rule(
    id: str | None = None,
    *,
    action: Action | str = Action.BLOCK,
    replacement: str = "[redacted]",
    on_error: OnError | None = None,
    enabled: bool = True,
) -> Callable[[RuleFunction], FunctionRule]:
    """Turn a function into a rule.

    Example::

        @rule(id="no-urls")
        def no_urls(text: str) -> bool:
            return "http://" in text or "https://" in text

    Args:
        id: Rule id. Defaults to the function name, with underscores replaced by dashes.
        action: What to do when the rule fires.
        replacement: Text used in place of a match when ``action`` is REDACT.
        on_error: What to do if the function raises. ``None`` uses the guard's setting.
        enabled: Set to False to keep the rule configured but switched off.
    """

    def wrap(fn: RuleFunction) -> FunctionRule:
        rule_id = id or getattr(fn, "__name__", "rule").replace("_", "-")
        return FunctionRule(
            fn,
            id=rule_id,
            action=action,
            replacement=replacement,
            on_error=on_error,
            enabled=enabled,
        )

    return wrap
