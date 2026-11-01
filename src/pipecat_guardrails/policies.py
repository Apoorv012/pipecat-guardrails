"""Layers 2 and 3: policies, written once as yes/no questions.

A policy is a rule that needs understanding rather than matching, such as "does this reply
tell the user to change their medication?". The guard asks it of a fast classifier (layer 2)
and, when the classifier is unsure, of an LLM judge (layer 3). Both are Pipecat classifiers
(``pipecat.classifiers``), so any classifier Pipecat supports can answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pipecat_guardrails.types import Action, PolicyOnError

if TYPE_CHECKING:
    from pipecat.classifiers.base_classifier import YesNoQuestion


@dataclass(slots=True)
class Policy:
    """A yes/no question about the bot's reply. "Yes" means the reply breaks the policy.

    Example::

        Policy(
            "Does this reply tell the user to stop, skip or change a medication?",
            yes="Any instruction to change how they take a medicine, in any language.",
            no="Telling them to ask their doctor first, or to keep taking it as prescribed.",
        )

    Parameters:
        question: The question, phrased so that "yes" is a violation.
        yes: What counts as a yes, when the question alone leaves it open.
        no: What counts as a no. Near misses belong here: they matter most for accuracy.
        action: ``BLOCK`` (default) or ``FLAG`` (report only). A policy has no span to
            redact, so ``REDACT`` is not allowed.
        allow_below: The classifier's probability of "yes" at or below which the reply
            passes without asking the judge.
        block_above: The classifier's probability at or above which the policy fires
            without asking the judge. Between the two, the judge decides. With no judge,
            the classifier decides alone at 0.5.
        on_error: What to do if the classifier or judge fails or times out on this
            policy. ``None`` (default) escalates a classifier failure to the judge when
            there is one, and otherwise uses the guard's ``on_error``.
        enabled: Set to False to keep the policy configured but switched off.
    """

    question: str
    yes: str | None = None
    no: str | None = None
    action: Action = Action.BLOCK
    allow_below: float = 0.2
    block_above: float = 0.8
    on_error: PolicyOnError | None = None
    enabled: bool = True

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise ValueError("policy question must not be empty")
        self.action = Action(self.action)
        if self.action not in (Action.BLOCK, Action.FLAG):
            raise ValueError(f"policy action must be 'block' or 'flag', not {self.action.value!r}")
        if not 0.0 <= self.allow_below <= self.block_above <= 1.0:
            raise ValueError("need 0 <= allow_below <= block_above <= 1")
        if self.on_error not in (None, "allow", "block", "escalate"):
            raise ValueError(
                f"on_error must be 'allow', 'block' or 'escalate', not {self.on_error!r}"
            )

    def to_question(self) -> YesNoQuestion:
        """The policy as a Pipecat ``YesNoQuestion``."""
        from pipecat.classifiers.base_classifier import YesNoQuestion

        return YesNoQuestion(instructions=self.question, yes=self.yes, no=self.no)
