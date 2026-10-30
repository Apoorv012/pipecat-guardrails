"""The guard: runs the configured checks on text and decides what may be said."""

from __future__ import annotations

import inspect
import time
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import replace

from loguru import logger

from pipecat_guardrails.rules.base import Rule
from pipecat_guardrails.types import (
    Action,
    CheckContext,
    CheckError,
    Mode,
    OnError,
    Verdict,
    Violation,
    strongest,
)

Fallback = str | Callable[[Verdict], str]
"""A fixed fallback line, or a function that picks one for a blocked verdict."""

DecisionHook = Callable[[Verdict], None] | Callable[[Verdict], Awaitable[None]]
"""Called with every verdict, e.g. to log decisions or collect metrics."""

_EMPTY_CONTEXT = CheckContext()


class Guard:
    """Checks text against the rules you configure.

    Everything is opt-in. A guard with no rules allows everything; each rule runs only if
    you add it, and can be switched off with ``enabled=False`` or :meth:`disable`.

    Example::

        guard = Guard(
            rules=[BannedTerms(["acme corp"], id="no-competitors")],
            fallback="Sorry, I can't help with that.",
        )
        verdict = await guard.check("Try Acme Corp instead.")
        verdict.blocked   # True
        verdict.text      # "Sorry, I can't help with that."
    """

    def __init__(
        self,
        rules: Iterable[Rule] = (),
        *,
        fallback: Fallback | None = None,
        mode: Mode = "enforce",
        on_error: OnError = "block",
        on_decision: DecisionHook | None = None,
        log_text: bool = False,
    ):
        """Configure the guard.

        Args:
            rules: Layer-1 rules to run, in order. Ids must be unique.
            fallback: What to say instead of blocked text. ``None`` means say nothing.
            mode: ``"enforce"`` applies verdicts. ``"shadow"`` only reports them, so new
                rules can be tried on live traffic without affecting users.
            on_error: Default for rules whose check raises: ``"block"`` (fail closed, the
                default) or ``"allow"`` (fail open). Rules can override it.
            on_decision: Called with every verdict. Errors in it are logged, not raised.
            log_text: Include the checked text in log messages. Off by default because
                the text may contain personal data.
        """
        self._rules: dict[str, Rule] = {}
        for r in rules:
            if r.id in self._rules:
                raise ValueError(f"duplicate rule id {r.id!r}")
            self._rules[r.id] = r
        if mode not in ("enforce", "shadow"):
            raise ValueError(f"mode must be 'enforce' or 'shadow', not {mode!r}")
        if on_error not in ("allow", "block"):
            raise ValueError(f"on_error must be 'allow' or 'block', not {on_error!r}")
        self.fallback = fallback
        self.mode: Mode = mode
        self.on_error: OnError = on_error
        self.on_decision = on_decision
        self.log_text = log_text

    # --- configuration -------------------------------------------------------------------

    @property
    def rules(self) -> Sequence[Rule]:
        """All configured rules, enabled or not, in order."""
        return tuple(self._rules.values())

    def rule(self, rule_id: str) -> Rule:
        """Return the rule with this id. Raises ``KeyError`` if there is none."""
        return self._rules[rule_id]

    def enable(self, *rule_ids: str) -> None:
        """Switch rules on. Raises ``KeyError`` for unknown ids."""
        for rule_id in rule_ids:
            self._rules[rule_id].enabled = True

    def disable(self, *rule_ids: str) -> None:
        """Switch rules off without removing them. Raises ``KeyError`` for unknown ids."""
        for rule_id in rule_ids:
            self._rules[rule_id].enabled = False

    # --- checking ------------------------------------------------------------------------

    async def check(self, text: str, context: CheckContext | None = None) -> Verdict:
        """Check ``text`` and return what may be said.

        Args:
            text: A sentence or a whole reply.
            context: Optional extra information for the checks.

        Returns:
            The verdict. Use ``verdict.text`` as the text to speak, or check
            ``verdict.blocked`` to decide yourself (e.g. to regenerate the reply).
        """
        verdict = self._check_rules(text, context or _EMPTY_CONTEXT)
        self._log(verdict)
        await self._notify(verdict)
        return verdict

    def _check_rules(self, text: str, context: CheckContext) -> Verdict:
        t0 = time.perf_counter()
        violations: list[Violation] = []
        errors: list[CheckError] = []
        redactions: list[tuple[int, int, str]] = []
        latency: dict[str, float] = {}

        if text.strip():
            for r in self._rules.values():
                if not r.enabled:
                    continue
                t = time.perf_counter()
                try:
                    matches = list(r.find(text, context))
                except Exception as e:
                    policy = r.on_error or self.on_error
                    message = f"{type(e).__name__}: {e}"
                    logger.opt(exception=e).warning(
                        f"pipecat-guardrails: rule {r.id!r} failed ({message}); failing "
                        + ("open" if policy == "allow" else "closed")
                    )
                    errors.append(CheckError(r.id, message, failed_open=policy == "allow"))
                    if policy == "block":
                        violations.append(
                            Violation(r.id, Action.BLOCK, f"check failed: {message}", error=True)
                        )
                    matches = []
                for m in matches:
                    violations.append(
                        Violation(r.id, r.action, m.reason, (m.start, m.end), m.score)
                    )
                    if r.action is Action.REDACT:
                        redactions.append((m.start, m.end, m.replacement or r.replacement))
                latency[r.id] = (time.perf_counter() - t) * 1000

        detected = strongest([v.action for v in violations])
        applied = Action.ALLOW if self.mode == "shadow" else detected
        if applied is Action.REDACT:
            spoken = _redact(text, redactions)
        elif applied is Action.BLOCK:
            spoken = ""
        else:
            spoken = text
        latency["total"] = (time.perf_counter() - t0) * 1000

        verdict = Verdict(
            action=applied,
            detected_action=detected,
            text=spoken,
            original_text=text,
            violations=tuple(violations),
            errors=tuple(errors),
            latency_ms=latency,
            shadow=self.mode == "shadow",
        )
        if applied is Action.BLOCK and self.fallback is not None:
            line = self.fallback if isinstance(self.fallback, str) else self.fallback(verdict)
            verdict = replace(verdict, text=line, fallback_text=line)
        return verdict

    def _log(self, verdict: Verdict) -> None:
        if verdict.detected_action is Action.ALLOW:
            return
        prefix = "[shadow] would " if verdict.shadow else ""
        message = (
            f"pipecat-guardrails: {prefix}{verdict.detected_action.value} "
            f"(rules: {', '.join(verdict.rule_ids)})"
        )
        if self.log_text:
            message += f" text={verdict.original_text!r}"
        # Blocks are rare and important; redactions and flags can be frequent.
        level = "INFO" if verdict.detected_action is Action.BLOCK else "DEBUG"
        logger.log(level, message)

    async def _notify(self, verdict: Verdict) -> None:
        if self.on_decision is None:
            return
        try:
            result = self.on_decision(verdict)
            if inspect.isawaitable(result):
                await result
        except Exception as e:
            logger.opt(exception=e).warning("pipecat-guardrails: on_decision hook failed")


def _redact(text: str, spans: list[tuple[int, int, str]]) -> str:
    """Replace spans, merging overlaps (the earliest span's replacement wins)."""
    merged: list[tuple[int, int, str]] = []
    for start, end, repl in sorted(spans, key=lambda s: (s[0], -s[1])):
        if merged and start < merged[-1][1]:
            s, e, r = merged[-1]
            merged[-1] = (s, max(e, end), r)
        else:
            merged.append((start, end, repl))
    parts: list[str] = []
    pos = 0
    for start, end, repl in merged:
        parts += [text[pos:start], repl]
        pos = end
    parts.append(text[pos:])
    return "".join(parts)
