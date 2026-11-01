"""The classifier -> judge cascade, using fake Pipecat classifiers."""

from collections.abc import Sequence
from typing import Any

import pytest
from fakes import FakeClassifier
from pipecat.classifiers.base_classifier import ClassifierError, YesNoQuestion

from pipecat_guardrails import Action, CheckContext, Guard, Policy
from pipecat_guardrails.rules import PII, BannedTerms, pii

FALLBACK = "Please ask your doctor."
MED = "Does this reply tell the user to change their medication?"


def guard(**kwargs: Any) -> Guard:
    kwargs.setdefault("policies", {"no-med-advice": MED})
    kwargs.setdefault("fallback", FALLBACK)
    return Guard(**kwargs)


# --- configuration ----------------------------------------------------------------------


def test_policies_need_a_classifier_or_judge():
    with pytest.raises(ValueError, match="classifier, a judge"):
        Guard(policies={"p": "Is it bad?"})


def test_classifier_needs_policies():
    with pytest.raises(ValueError, match="needs policies"):
        Guard(judge=FakeClassifier())


def test_policy_and_rule_ids_must_differ():
    with pytest.raises(ValueError, match="both a rule and a policy"):
        Guard([BannedTerms(["x"], id="p")], policies={"p": "Is it bad?"}, judge=FakeClassifier())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"action": Action.REDACT},
        {"allow_below": 0.9, "block_above": 0.1},
        {"on_error": "maybe"},
    ],
)
def test_invalid_policy(kwargs: dict[str, Any]):
    with pytest.raises(ValueError):
        Policy("Is it bad?", **kwargs)


def test_policy_becomes_a_pipecat_question():
    q = Policy("Is it bad?", yes="rude words", no="jokes").to_question()
    assert q == YesNoQuestion(instructions="Is it bad?", yes="rude words", no="jokes")


# --- judge only -------------------------------------------------------------------------


async def test_judge_yes_blocks():
    judge = FakeClassifier({"no-med-advice": 0.9})
    verdict = await guard(judge=judge).check("Stop the pills.")
    assert verdict.blocked
    assert verdict.text == FALLBACK
    [v] = verdict.violations
    assert (v.rule_id, v.layer, v.score) == ("no-med-advice", "judge", 0.9)
    assert "judge" in verdict.latency_ms


async def test_judge_no_allows():
    verdict = await guard(judge=FakeClassifier({"no-med-advice": 0.1})).check("Take care.")
    assert verdict.action is Action.ALLOW


async def test_all_policies_go_in_one_call():
    judge = FakeClassifier({"b": 0.7})
    g = Guard(policies={"a": "Is it A?", "b": "Is it B?"}, judge=judge)
    verdict = await g.check("text")
    assert len(judge.calls) == 1
    assert set(judge.calls[0][1]) == {"a", "b"}
    assert verdict.rule_ids == ("b",)


async def test_flag_policy_reports_but_allows():
    judge = FakeClassifier({"tone": 0.9})
    g = Guard(policies={"tone": Policy("Is it rude?", action=Action.FLAG)}, judge=judge)
    verdict = await g.check("text")
    assert verdict.action is Action.FLAG
    assert verdict.text == "text"


# --- classifier -> judge ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("probability", "asks_judge", "blocked"),
    [(0.95, False, True), (0.05, False, False), (0.5, True, True)],
)
async def test_classifier_decides_when_sure(probability: float, asks_judge: bool, blocked: bool):
    classifier = FakeClassifier({"no-med-advice": probability})
    judge = FakeClassifier({"no-med-advice": 0.9})
    verdict = await guard(classifier=classifier, judge=judge).check("text")
    assert bool(judge.calls) is asks_judge
    assert verdict.blocked is blocked
    if blocked:
        assert verdict.violations[0].layer == ("judge" if asks_judge else "classifier")


async def test_only_unsure_policies_go_to_judge():
    classifier = FakeClassifier({"sure-yes": 0.99, "sure-no": 0.01, "unsure": 0.5})
    judge = FakeClassifier({"unsure": 0.1})
    policies = {k: f"{k}?" for k in ("sure-yes", "sure-no", "unsure")}
    verdict = await Guard(policies=policies, classifier=classifier, judge=judge).check("t")
    assert set(judge.calls[0][1]) == {"unsure"}
    assert verdict.rule_ids == ("sure-yes",)


async def test_thresholds_are_per_policy():
    classifier = FakeClassifier({"strict": 0.6})
    judge = FakeClassifier()
    policies = {"strict": Policy("strict?", block_above=0.6)}
    verdict = await Guard(policies=policies, classifier=classifier, judge=judge).check("t")
    assert verdict.blocked
    assert not judge.calls


async def test_classifier_alone_decides_at_one_half():
    for probability, blocked in [(0.5, True), (0.49, False)]:
        g = guard(classifier=FakeClassifier({"no-med-advice": probability}))
        assert (await g.check("t")).blocked is blocked


# --- rules come first -------------------------------------------------------------------


async def test_rule_block_skips_policies():
    judge = FakeClassifier({"no-med-advice": 0.9})
    g = guard(rules=[BannedTerms(["acme"], id="no-competitors")], judge=judge)
    verdict = await g.check("Try Acme.")
    assert verdict.rule_ids == ("no-competitors",)
    assert not judge.calls


async def test_policies_see_redacted_text():
    judge = FakeClassifier()
    g = guard(rules=[PII(pii.email())], judge=judge)
    verdict = await g.check("Mail a@b.com today.")
    assert judge.calls[0][0] == "Mail [email] today."
    assert verdict.text == "Mail [email] today."


async def test_evaluate_can_ask_policies_after_a_rule_block():
    judge = FakeClassifier({"no-med-advice": 0.9})
    g = guard(rules=[BannedTerms(["acme"], id="no-competitors")], judge=judge)
    verdict = await g._evaluate("Try Acme.", CheckContext(), stop_on_block=False)  # pyright: ignore[reportPrivateUsage]
    assert verdict.rule_ids == ("no-competitors", "no-med-advice")


async def test_empty_text_asks_nobody():
    judge = FakeClassifier()
    await guard(judge=judge).check("   ")
    assert not judge.calls


async def test_disabled_policy_is_not_asked():
    judge = FakeClassifier({"no-med-advice": 0.9})
    g = guard(judge=judge)
    g.disable("no-med-advice")
    assert (await g.check("t")).action is Action.ALLOW
    assert not judge.calls
    g.enable("no-med-advice")
    assert (await g.check("t")).blocked


async def test_shadow_mode_reports_policy_violations():
    verdict = await guard(judge=FakeClassifier(default=0.9), mode="shadow").check("t")
    assert verdict.action is Action.ALLOW
    assert verdict.detected_action is Action.BLOCK
    assert verdict.rule_ids == ("no-med-advice",)


# --- context ----------------------------------------------------------------------------


async def test_history_is_sent_with_the_reply():
    judge = FakeClassifier()
    history = [{"role": "user", "content": f"msg {i}", "extra": 1} for i in range(6)]
    await guard(judge=judge, history_turns=2).check("Yes, do that.", CheckContext(history=history))
    assert judge.calls[0][0] == {
        "conversation": [
            {"role": "user", "content": "msg 4"},
            {"role": "user", "content": "msg 5"},
        ],
        "reply": "Yes, do that.",
    }


async def test_history_turns_zero_sends_text_only():
    judge = FakeClassifier()
    ctx = CheckContext(history=[{"role": "user", "content": "hi"}])
    await guard(judge=judge, history_turns=0).check("t", ctx)
    assert judge.calls[0][0] == "t"


# --- failures ---------------------------------------------------------------------------


async def test_judge_error_fails_closed_by_default():
    judge = FakeClassifier(error=ClassifierError("down"))
    verdict = await guard(judge=judge).check("t")
    assert verdict.blocked
    [error] = verdict.errors
    assert (error.rule_id, error.layer, error.failed_open) == ("no-med-advice", "judge", False)
    assert verdict.violations[0].error


async def test_judge_error_can_fail_open():
    judge = FakeClassifier(error=ClassifierError("down"))
    verdict = await guard(judge=judge, on_error="allow").check("t")
    assert verdict.action is Action.ALLOW
    assert verdict.errors[0].failed_open


async def test_policy_on_error_overrides_guard():
    judge = FakeClassifier(error=ClassifierError("down"))
    policies = {"p": Policy("p?", on_error="allow")}
    verdict = await Guard(policies=policies, judge=judge).check("t")
    assert verdict.action is Action.ALLOW


async def test_judge_timeout():
    judge = FakeClassifier(default=0.9, delay=1.0)
    verdict = await guard(judge=judge, judge_timeout=0.05).check("t")
    assert verdict.blocked
    assert "timed out" in verdict.errors[0].message
    assert verdict.latency_ms["judge"] < 500


async def test_classifier_error_escalates_to_judge():
    classifier = FakeClassifier(error=ClassifierError("down"))
    judge = FakeClassifier({"no-med-advice": 0.1})
    verdict = await guard(classifier=classifier, judge=judge).check("t")
    assert verdict.action is Action.ALLOW
    assert judge.calls
    [error] = verdict.errors
    assert error.escalated
    assert error.layer == "classifier"


async def test_classifier_error_without_judge_uses_guard_on_error():
    classifier = FakeClassifier(error=ClassifierError("down"))
    assert (await guard(classifier=classifier).check("t")).blocked
    assert (await guard(classifier=classifier, on_error="allow").check("t")).action is Action.ALLOW


async def test_classifier_error_with_policy_block_does_not_escalate():
    classifier = FakeClassifier(error=ClassifierError("down"))
    judge = FakeClassifier()
    policies = {"p": Policy("p?", on_error="block")}
    verdict = await Guard(policies=policies, classifier=classifier, judge=judge).check("t")
    assert verdict.blocked
    assert not judge.calls


async def test_missing_answer_counts_as_failure():
    class Forgetful(FakeClassifier):
        async def _ask(self, state: Any, questions: Any, images: Sequence[Any] = ()) -> Any:
            return {}, None

    verdict = await guard(judge=Forgetful()).check("t")
    assert verdict.blocked
    assert "no answer" in verdict.errors[0].message


async def test_escalate_without_judge_uses_guard_on_error():
    classifier = FakeClassifier(error=ClassifierError("down"))
    policies = {"p": Policy("p?", on_error="escalate")}
    g = Guard(policies=policies, classifier=classifier, on_error="allow")
    assert (await g.check("t")).action is Action.ALLOW
