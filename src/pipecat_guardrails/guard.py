"""The guard: runs the configured checks on text and decides what may be said."""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from loguru import logger

from pipecat_guardrails.policies import Policy
from pipecat_guardrails.rules.base import Rule
from pipecat_guardrails.types import (
    Action,
    CheckContext,
    CheckError,
    Layer,
    Mode,
    OnError,
    Verdict,
    Violation,
    strongest,
)

if TYPE_CHECKING:
    from pipecat.classifiers.base_classifier import BaseClassifier, YesNoResult

Fallback = str | Callable[[Verdict], str]
"""A fixed fallback line, or a function that picks one for a blocked verdict."""

DecisionHook = Callable[[Verdict], None] | Callable[[Verdict], Awaitable[None]]
"""Called with every verdict, e.g. to log decisions or collect metrics."""

_EMPTY_CONTEXT = CheckContext()


@dataclass
class _Findings:
    """What the layers found, before it is turned into a verdict."""

    violations: list[Violation] = field(default_factory=list[Violation])
    errors: list[CheckError] = field(default_factory=list[CheckError])
    redactions: list[tuple[int, int, str]] = field(default_factory=list[tuple[int, int, str]])
    latency: dict[str, float] = field(default_factory=dict[str, float])

    @property
    def blocked(self) -> bool:
        return any(v.action is Action.BLOCK for v in self.violations)


class Guard:
    """Checks text against the rules and policies you configure.

    Each piece of text goes through up to three layers, and stops at the first one that is
    sure:

    1. **Rules**: banned terms, regexes, personal data, custom functions. Local and instant.
    2. **Classifier** (optional): a fast model such as TypeSafe Jev answers every policy
       at once. A confident answer decides; an unsure one goes to layer 3.
    3. **Judge** (optional): an LLM gives the final answer for the policies still open.

    Everything is opt-in. A guard with no rules or policies allows everything; each one runs
    only if you add it, and can be switched off with ``enabled=False`` or :meth:`disable`.

    Example::

        guard = Guard(
            rules=[BannedTerms(["acme corp"], id="no-competitors")],
            policies={
                "no-med-advice": "Does this reply tell the user to change their medication?",
            },
            judge=LLMClassifier(llm=OpenAILLMService(...)),
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
        policies: Mapping[str, str | Policy] | None = None,
        classifier: BaseClassifier | None = None,
        judge: BaseClassifier | None = None,
        classifier_timeout: float = 2.0,
        judge_timeout: float = 5.0,
        history_turns: int = 4,
        fallback: Fallback | None = None,
        mode: Mode = "enforce",
        on_error: OnError = "block",
        on_decision: DecisionHook | None = None,
        log_text: bool = False,
    ):
        """Configure the guard.

        Args:
            rules: Layer-1 rules to run, in order.
            policies: Yes/no questions for layers 2 and 3, by id. A value is the question
                itself, or a :class:`Policy` with examples, thresholds and an action.
                Needs a ``classifier``, a ``judge``, or both. Ids must be unique across
                rules and policies.
            classifier: Layer 2: a Pipecat classifier (``pipecat.classifiers``) that
                answers all policies in one call. Probabilities between a policy's
                ``allow_below`` and ``block_above`` go to the judge.
            judge: Layer 3: a Pipecat classifier, usually ``LLMClassifier``, that gives
                the final yes/no for policies the classifier was unsure about (or for all
                policies, if there is no classifier).
            classifier_timeout: Seconds to wait for the classifier before treating the
                call as failed.
            judge_timeout: Seconds to wait for the judge before treating the call as failed.
            history_turns: How many recent messages from ``CheckContext.history`` the
                classifier and judge see. ``0`` sends the text alone.
            fallback: What to say instead of blocked text. ``None`` means say nothing.
            mode: ``"enforce"`` applies verdicts. ``"shadow"`` only reports them, so new
                rules can be tried on live traffic without affecting users.
            on_error: Default for checks that raise or time out: ``"block"`` (fail closed,
                the default) or ``"allow"`` (fail open). Rules and policies can override it.
            on_decision: Called with every verdict. Errors in it are logged, not raised.
            log_text: Include the checked text in log messages. Off by default because
                the text may contain personal data.
        """
        self._rules: dict[str, Rule] = {}
        for r in rules:
            if r.id in self._rules:
                raise ValueError(f"duplicate rule id {r.id!r}")
            self._rules[r.id] = r
        self._policies: dict[str, Policy] = {}
        for policy_id, policy in (policies or {}).items():
            if not policy_id:
                raise ValueError("policy id must not be empty")
            if policy_id in self._rules:
                raise ValueError(f"id {policy_id!r} is used by both a rule and a policy")
            self._policies[policy_id] = policy if isinstance(policy, Policy) else Policy(policy)
        if self._policies and classifier is None and judge is None:
            raise ValueError("policies need a classifier, a judge, or both to answer them")
        if not self._policies and (classifier is not None or judge is not None):
            raise ValueError("a classifier or judge needs policies to ask")
        if mode not in ("enforce", "shadow"):
            raise ValueError(f"mode must be 'enforce' or 'shadow', not {mode!r}")
        if on_error not in ("allow", "block"):
            raise ValueError(f"on_error must be 'allow' or 'block', not {on_error!r}")
        if classifier_timeout <= 0 or judge_timeout <= 0:
            raise ValueError("timeouts must be positive")
        if history_turns < 0:
            raise ValueError("history_turns must not be negative")
        self.classifier = classifier
        self.judge = judge
        self.classifier_timeout = classifier_timeout
        self.judge_timeout = judge_timeout
        self.history_turns = history_turns
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

    @property
    def policies(self) -> Mapping[str, Policy]:
        """All configured policies by id, enabled or not."""
        return dict(self._policies)

    def rule(self, rule_id: str) -> Rule:
        """Return the rule with this id. Raises ``KeyError`` if there is none."""
        return self._rules[rule_id]

    def policy(self, policy_id: str) -> Policy:
        """Return the policy with this id. Raises ``KeyError`` if there is none."""
        return self._policies[policy_id]

    def enable(self, *ids: str) -> None:
        """Switch rules or policies on. Raises ``KeyError`` for unknown ids."""
        for item in [self._item(i) for i in ids]:
            item.enabled = True

    def disable(self, *ids: str) -> None:
        """Switch rules or policies off, keeping them configured. ``KeyError`` for unknown ids."""
        for item in [self._item(i) for i in ids]:
            item.enabled = False

    def _item(self, item_id: str) -> Rule | Policy:
        if item_id in self._rules:
            return self._rules[item_id]
        if item_id in self._policies:
            return self._policies[item_id]
        raise KeyError(item_id)

    # --- checking ------------------------------------------------------------------------

    async def check(self, text: str, context: CheckContext | None = None) -> Verdict:
        """Check ``text`` and return what may be said.

        Rules run first. If they block, the classifier and judge are skipped. Otherwise the
        policies go to the classifier and then, if it is unsure, to the judge. If the rules
        redacted anything, the classifier and judge see the redacted text, so personal
        data is not sent to them.

        Args:
            text: A sentence or a whole reply.
            context: Optional extra information for the checks.

        Returns:
            The verdict. Use ``verdict.text`` as the text to speak, or check
            ``verdict.blocked`` to decide yourself (e.g. to regenerate the reply).
        """
        verdict = await self._evaluate(text, context or _EMPTY_CONTEXT)
        self._log(verdict)
        await self._notify(verdict)
        return verdict

    async def _evaluate(
        self, text: str, context: CheckContext, *, stop_on_block: bool = True
    ) -> Verdict:
        """Run every layer. ``stop_on_block=False`` asks the policies even after a rule blocked."""
        t0 = time.perf_counter()
        found = _Findings()
        if text.strip():
            self._run_rules(text, context, found)
            if self._policies and not (stop_on_block and found.blocked):
                state_text = _redact(text, found.redactions) if found.redactions else text
                await self._run_policies(state_text, context, found)
        found.latency["total"] = (time.perf_counter() - t0) * 1000
        return self._verdict(text, found)

    def _run_rules(self, text: str, context: CheckContext, found: _Findings) -> None:
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
                found.errors.append(CheckError(r.id, message, failed_open=policy == "allow"))
                if policy == "block":
                    found.violations.append(
                        Violation(r.id, Action.BLOCK, f"check failed: {message}", error=True)
                    )
                matches = []
            for m in matches:
                found.violations.append(
                    Violation(r.id, r.action, m.reason, (m.start, m.end), m.score)
                )
                if r.action is Action.REDACT:
                    found.redactions.append((m.start, m.end, m.replacement or r.replacement))
            found.latency[r.id] = (time.perf_counter() - t) * 1000

    async def _run_policies(self, text: str, context: CheckContext, found: _Findings) -> None:
        open_ = {pid: p for pid, p in self._policies.items() if p.enabled}
        if not open_:
            return
        state = self._state(text, context)

        if self.classifier is not None:
            results = await self._ask(
                "classifier", self.classifier, self.classifier_timeout, state, open_, found
            )
            if isinstance(results, str):  # the call failed: apply each policy's error policy
                can_escalate = self.judge is not None
                for pid, p in list(open_.items()):
                    choice = p.on_error or ("escalate" if can_escalate else self.on_error)
                    if choice == "escalate" and can_escalate:
                        found.errors.append(
                            CheckError(
                                pid, results, failed_open=False, layer="classifier", escalated=True
                            )
                        )
                        continue
                    if choice == "escalate":  # no judge to escalate to
                        choice = self.on_error
                    _fail(found, pid, "classifier", results, choice)
                    del open_[pid]
            else:
                for pid, p in list(open_.items()):
                    yes = results[pid].probability
                    if self.judge is None:
                        decided, fires = True, yes >= 0.5
                    else:
                        decided = yes >= p.block_above or yes <= p.allow_below
                        fires = yes >= p.block_above
                    if fires:
                        found.violations.append(_policy_violation(pid, p, "classifier", yes))
                    if decided:
                        del open_[pid]

        if self.judge is not None and open_:
            results = await self._ask("judge", self.judge, self.judge_timeout, state, open_, found)
            for pid, p in open_.items():
                if isinstance(results, str):
                    choice = p.on_error if p.on_error in ("allow", "block") else self.on_error
                    _fail(found, pid, "judge", results, "allow" if choice == "allow" else "block")
                elif results[pid].is_yes:
                    found.violations.append(
                        _policy_violation(pid, p, "judge", results[pid].probability)
                    )

    async def _ask(
        self,
        layer: Layer,
        classifier: BaseClassifier,
        timeout: float,
        state: str | dict[str, Any],
        policies: Mapping[str, Policy],
        found: _Findings,
    ) -> Mapping[str, YesNoResult] | str:
        """Ask one layer every open policy in one call. A string is the error if it failed."""
        questions = {pid: p.to_question() for pid, p in policies.items()}
        t = time.perf_counter()
        try:
            results = await asyncio.wait_for(classifier.yes_no(state, questions), timeout)
            missing = [pid for pid in policies if pid not in results]
            if missing:
                raise ValueError(f"no answer for {', '.join(missing)}")
            return results
        except Exception as e:
            message = (
                f"timed out after {timeout:g}s"
                if isinstance(e, TimeoutError)
                else f"{type(e).__name__}: {e}"
            )
            logger.warning(f"pipecat-guardrails: {layer} failed ({message})")
            return message
        finally:
            found.latency[layer] = (time.perf_counter() - t) * 1000

    def _state(self, text: str, context: CheckContext) -> str | dict[str, Any]:
        """What the classifier and judge see: the text, with recent turns if there are any."""
        if not self.history_turns or not context.history:
            return text
        turns = [
            {"role": m.get("role"), "content": m.get("content")}
            for m in list(context.history)[-self.history_turns :]
        ]
        return {"conversation": turns, "reply": text}

    def _verdict(self, text: str, found: _Findings) -> Verdict:
        detected = strongest([v.action for v in found.violations])
        applied = Action.ALLOW if self.mode == "shadow" else detected
        if applied is Action.REDACT:
            spoken = _redact(text, found.redactions)
        elif applied is Action.BLOCK:
            spoken = ""
        else:
            spoken = text
        verdict = Verdict(
            action=applied,
            detected_action=detected,
            text=spoken,
            original_text=text,
            violations=tuple(found.violations),
            errors=tuple(found.errors),
            latency_ms=found.latency,
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


def _fail(found: _Findings, pid: str, layer: Layer, message: str, choice: OnError) -> None:
    """Record a failed policy check, and block if it fails closed."""
    found.errors.append(CheckError(pid, message, failed_open=choice == "allow", layer=layer))
    if choice == "block":
        found.violations.append(
            Violation(pid, Action.BLOCK, f"check failed: {message}", error=True, layer=layer)
        )


def _policy_violation(pid: str, policy: Policy, layer: Layer, probability: float) -> Violation:
    return Violation(
        pid,
        policy.action,
        f"{layer} answered yes (p={probability:.2f})",
        score=probability,
        layer=layer,
    )


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
