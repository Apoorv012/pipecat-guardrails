"""Core data types shared by every layer of the guard."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

Mode = Literal["enforce", "shadow"]
"""``enforce`` applies verdicts; ``shadow`` only reports what would have happened."""

OnError = Literal["allow", "block"]
"""What to do when a check fails: ``allow`` fails open, ``block`` fails closed."""

PolicyOnError = Literal["allow", "block", "escalate"]
"""What to do when a classifier or judge fails on a policy.

``allow`` fails open, ``block`` fails closed, and ``escalate`` passes the policy to the next
layer (from the classifier to the judge). With no next layer, ``escalate`` falls back to the
guard's ``on_error``.
"""

Layer = Literal["rules", "classifier", "judge"]
"""The layer that produced a violation or an error."""


class Action(StrEnum):
    """What to do with a piece of text, from least to most severe."""

    ALLOW = "allow"
    """Release the text unchanged."""
    FLAG = "flag"
    """Release the text unchanged, but report the violation."""
    REDACT = "redact"
    """Replace the matched spans, then release the text."""
    BLOCK = "block"
    """Do not release the text; use the fallback instead."""

    @property
    def severity(self) -> int:
        """Rank used to pick the strongest action (ALLOW=0 ... BLOCK=3)."""
        return _SEVERITY[self]


_SEVERITY = {Action.ALLOW: 0, Action.FLAG: 1, Action.REDACT: 2, Action.BLOCK: 3}


def strongest(actions: Sequence[Action]) -> Action:
    """Return the most severe action, or ALLOW for an empty sequence."""
    return max(actions, key=lambda a: a.severity, default=Action.ALLOW)


@dataclass(frozen=True, slots=True)
class CheckContext:
    """Extra information a check may use.

    Parameters:
        history: Recent conversation messages in OpenAI chat format, oldest first.
            Rules ignore it. The classifier and the judge get the last few messages
            (``Guard(history_turns=...)``) so they can understand the reply in context.
        language: Language hint such as ``"hi"`` or ``"en-IN"``, if known.
        metadata: Free-form values for custom rules (call id, user tier, ...).
    """

    history: Sequence[Mapping[str, Any]] = ()
    language: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict[str, Any])


@dataclass(frozen=True, slots=True)
class Violation:
    """One rule (or policy) that the text broke.

    Parameters:
        rule_id: Id of the rule or policy that fired.
        action: The action that rule asks for.
        reason: Human-readable explanation, safe to log (never contains the matched text).
        span: ``(start, end)`` offsets of the match in the original text, if the rule
            located one.
        score: Confidence from 0 to 1, if the rule produces one. For a policy, the
            probability of "yes" that decided it.
        error: True when this violation was produced because the check itself failed
            and its error policy is ``"block"``.
        layer: ``"rules"``, ``"classifier"`` or ``"judge"``: the layer that decided.
    """

    rule_id: str
    action: Action
    reason: str
    span: tuple[int, int] | None = None
    score: float | None = None
    error: bool = False
    layer: Layer = "rules"


@dataclass(frozen=True, slots=True)
class CheckError:
    """A check that raised an exception (or timed out) instead of returning a result.

    Parameters:
        rule_id: Id of the rule or policy that failed.
        message: Exception type and message.
        failed_open: True if the text was allowed despite the failure.
        layer: ``"rules"``, ``"classifier"`` or ``"judge"``: where it failed.
        escalated: True if the policy was passed on to the next layer instead of being
            decided here (a classifier failure with ``on_error="escalate"``).
    """

    rule_id: str
    message: str
    failed_open: bool
    layer: Layer = "rules"
    escalated: bool = False


@dataclass(frozen=True, slots=True)
class Verdict:
    """The outcome of checking one piece of text.

    Parameters:
        action: The action applied. Always ALLOW in shadow mode.
        detected_action: The strongest action any rule asked for, even in shadow mode.
        text: What to say: the original text, the redacted text, or the fallback
            (an empty string if blocked with no fallback).
        original_text: The text that was checked.
        violations: Every rule that fired, in rule order, then every policy.
        errors: Checks that failed while running.
        fallback_text: The fallback line, set only when the text was blocked.
        latency_ms: Time spent per rule id, per layer (``"classifier"``, ``"judge"``)
            when it ran, plus ``"total"``.
        shadow: True if the guard ran in shadow mode.
    """

    action: Action
    detected_action: Action
    text: str
    original_text: str
    violations: tuple[Violation, ...] = ()
    errors: tuple[CheckError, ...] = ()
    fallback_text: str | None = None
    latency_ms: Mapping[str, float] = field(default_factory=dict[str, float])
    shadow: bool = False

    @property
    def blocked(self) -> bool:
        """True if the text must not be spoken."""
        return self.action is Action.BLOCK

    @property
    def redacted(self) -> bool:
        """True if ``text`` differs from ``original_text`` because of redaction."""
        return self.action is Action.REDACT

    @property
    def rule_ids(self) -> tuple[str, ...]:
        """Ids of the rules that fired, without duplicates, in order."""
        return tuple(dict.fromkeys(v.rule_id for v in self.violations))
