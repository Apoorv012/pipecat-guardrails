from collections.abc import Iterable

import pytest

from pipecat_guardrails import Action, CheckContext, Guard, Verdict
from pipecat_guardrails.packs import india
from pipecat_guardrails.rules import PII, BannedTerms, Match, Rule, pii, rule


def competitors(**kwargs: object) -> BannedTerms:
    return BannedTerms(["acme corp"], id="no-competitors", **kwargs)  # type: ignore[arg-type]


class Exploding(Rule):
    def find(self, text: str, context: CheckContext) -> Iterable[Match]:
        raise RuntimeError("boom")


async def test_guard_with_no_rules_allows_everything():
    verdict = await Guard().check("Try Acme Corp.")
    assert verdict.action is Action.ALLOW
    assert verdict.text == "Try Acme Corp."
    assert verdict.violations == ()


async def test_block_uses_fallback():
    guard = Guard([competitors()], fallback="Sorry, I can't help with that.")
    verdict = await guard.check("You should try ACME Corp.")
    assert verdict.blocked
    assert verdict.text == verdict.fallback_text == "Sorry, I can't help with that."
    assert verdict.original_text == "You should try ACME Corp."
    assert verdict.rule_ids == ("no-competitors",)
    [v] = verdict.violations
    assert v.span == (15, 24)
    assert "acme corp" in v.reason


async def test_block_without_fallback_says_nothing():
    verdict = await Guard([competitors()]).check("acme corp is great")
    assert verdict.blocked
    assert verdict.text == ""
    assert verdict.fallback_text is None


async def test_fallback_function_receives_the_verdict():
    def pick(v: Verdict) -> str:
        return f"blocked by {v.rule_ids[0]}"

    verdict = await Guard([competitors()], fallback=pick).check("acme corp")
    assert verdict.text == "blocked by no-competitors"


async def test_redact_replaces_spans_with_labels():
    guard = Guard([PII(pii.email(), india.phone())])
    verdict = await guard.check("Mail a@b.com or call 9876543210.")
    assert verdict.action is Action.REDACT
    assert verdict.redacted
    assert verdict.text == "Mail [email] or call [phone number]."


async def test_flag_reports_but_does_not_change_text():
    verdict = await Guard([competitors(action="flag")]).check("acme corp")
    assert verdict.action is Action.FLAG
    assert verdict.text == "acme corp"


async def test_strongest_action_wins():
    guard = Guard([PII(pii.email()), competitors()], fallback="no")
    verdict = await guard.check("acme corp: a@b.com")
    assert verdict.action is Action.BLOCK
    assert {v.action for v in verdict.violations} == {Action.BLOCK, Action.REDACT}


async def test_overlapping_redactions_merge():
    r1 = BannedTerms(["secret plan"], id="a", action="redact", replacement="[x]")
    r2 = BannedTerms(["plan b"], id="b", action="redact", replacement="[y]")
    verdict = await Guard([r1, r2]).check("the secret plan b now")
    assert verdict.text == "the [x] now"


async def test_shadow_mode_reports_without_acting():
    guard = Guard([competitors()], fallback="no", mode="shadow")
    verdict = await guard.check("acme corp")
    assert verdict.action is Action.ALLOW
    assert verdict.detected_action is Action.BLOCK
    assert verdict.shadow
    assert verdict.text == "acme corp"
    assert verdict.fallback_text is None


async def test_disabled_rules_do_not_run():
    guard = Guard([competitors(enabled=False)])
    assert not (await guard.check("acme corp")).blocked
    guard.enable("no-competitors")
    assert (await guard.check("acme corp")).blocked
    guard.disable("no-competitors")
    verdict = await guard.check("acme corp")
    assert not verdict.blocked
    assert "no-competitors" not in verdict.latency_ms


async def test_enable_unknown_rule_raises():
    with pytest.raises(KeyError):
        Guard().enable("nope")


async def test_rule_error_fails_closed_by_default():
    verdict = await Guard([Exploding(id="x")], fallback="no").check("hello")
    assert verdict.blocked
    [err] = verdict.errors
    assert err.rule_id == "x"
    assert "RuntimeError: boom" in err.message
    assert not err.failed_open
    assert verdict.violations[0].error


async def test_rule_error_can_fail_open():
    guard = Guard([Exploding(id="x", on_error="allow")])
    verdict = await guard.check("hello")
    assert verdict.action is Action.ALLOW
    assert verdict.errors[0].failed_open
    assert verdict.violations == ()


async def test_guard_level_fail_open():
    verdict = await Guard([Exploding(id="x")], on_error="allow").check("hello")
    assert verdict.action is Action.ALLOW


async def test_duplicate_rule_ids_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        Guard([competitors(), competitors()])


async def test_invalid_mode_rejected():
    with pytest.raises(ValueError, match="mode"):
        Guard(mode="off")  # type: ignore[arg-type]


async def test_empty_text_is_allowed_without_running_rules():
    verdict = await Guard([Exploding(id="x")]).check("   ")
    assert verdict.action is Action.ALLOW
    assert verdict.errors == ()


async def test_on_decision_sync_and_async():
    seen: list[Action] = []

    def sync_hook(v: Verdict) -> None:
        seen.append(v.action)

    async def async_hook(v: Verdict) -> None:
        seen.append(v.action)

    await Guard([competitors()], on_decision=sync_hook).check("acme corp")
    await Guard([competitors()], on_decision=async_hook).check("hello")
    assert seen == [Action.BLOCK, Action.ALLOW]


async def test_on_decision_errors_are_swallowed():
    def broken(v: Verdict) -> None:
        raise RuntimeError("hook failed")

    verdict = await Guard([competitors()], on_decision=broken).check("acme corp")
    assert verdict.blocked


async def test_latency_is_reported_per_rule():
    verdict = await Guard([competitors(), PII(pii.email())]).check("hello")
    assert set(verdict.latency_ms) == {"no-competitors", "pii", "total"}
    assert all(ms >= 0 for ms in verdict.latency_ms.values())


async def test_context_reaches_function_rules():
    @rule(id="hindi-only", action="flag")
    def hindi_only(text: str, context: CheckContext) -> bool:
        return context.language != "hi"

    guard = Guard([hindi_only])
    assert (await guard.check("hello", CheckContext(language="en"))).action is Action.FLAG
    assert (await guard.check("hello", CheckContext(language="hi"))).action is Action.ALLOW
