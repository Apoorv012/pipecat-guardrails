"""The same health-bot policy, in English and in Hinglish.

Policy: no dosage changes, no reassurance, and no personal data read aloud.
Each language has its own guard and its own replies; the expected outcomes mirror each other.
"""

import pytest

from pipecat_guardrails import Action, Guard
from pipecat_guardrails.packs import india
from pipecat_guardrails.rules import PII, BannedTerms, pii

FALLBACK = "Please ask your doctor about that."

english = Guard(
    [
        BannedTerms(
            ["stop taking", "double the dose", "increase the dose", "skip your dose"],
            id="no-dosage-change",
        ),
        BannedTerms(
            ["don't worry", "everything will be fine", "nothing to worry about"],
            id="no-reassurance",
        ),
        PII(pii.email(), pii.phone(), pii.card(), id="pii"),
    ],
    fallback=FALLBACK,
)

hinglish = Guard(
    [
        BannedTerms(
            ["dawai band kar do", "dose double", "dawai chhod do"],
            id="no-dosage-change",
            word_pattern=india.hinglish_word_pattern,
            variants={"dawai": ["दवाई", "दवा"]},
        ),
        BannedTerms(
            ["chinta mat", "sab theek ho jayega", "ghabraiye mat"],
            id="no-reassurance",
            word_pattern=india.hinglish_word_pattern,
        ),
        PII(pii.email(), india.phone(), india.aadhaar(), id="pii"),
    ],
    fallback=FALLBACK,
)

CASES = [
    # (scenario, English reply, Hinglish reply, expected action, expected rule)
    (
        "safe question",
        "Did you take your morning medicine today?",
        "Kya aapne aaj subah ki dawai le li?",
        Action.ALLOW,
        None,
    ),
    (
        "dosage change",
        "You can stop taking it for a few days.",
        "Aap kuch din dawai band kar do.",
        Action.BLOCK,
        "no-dosage-change",
    ),
    (
        "dosage change, unusual spelling or script",
        "Just DOUBLE   the dose tonight.",
        "Aaj raat davaai chhod do.",
        Action.BLOCK,
        "no-dosage-change",
    ),
    (
        "dosage change in Devanagari",
        "Increase the dose from tomorrow.",
        "कल से दवाई band kar do.",
        Action.BLOCK,
        "no-dosage-change",
    ),
    (
        "reassurance",
        "Don’t worry, everything will be fine.",  # noqa: RUF001 - curly apostrophe on purpose
        "Cheenta mat kijiye, sab theek ho jaayega.",
        Action.BLOCK,
        "no-reassurance",
    ),
    (
        "phone number",
        "I'll call you back on +1 415 555 0132.",
        "Main aapko 98765 43210 par call karungi.",
        Action.REDACT,
        "pii",
    ),
    (
        "similar but safe words",
        "Take your tablets with water; ask the doctor before any dose change.",
        "Dawai paani ke saath lijiye, dose ke baare mein doctor se poochiye.",
        Action.ALLOW,
        None,
    ),
]


@pytest.mark.parametrize(
    ("scenario", "reply", "action", "rule_id"),
    [(c[0], c[1], c[3], c[4]) for c in CASES],
    ids=[c[0] for c in CASES],
)
async def test_english(scenario: str, reply: str, action: Action, rule_id: str | None):
    await _check(english, reply, action, rule_id)


@pytest.mark.parametrize(
    ("scenario", "reply", "action", "rule_id"),
    [(c[0], c[2], c[3], c[4]) for c in CASES],
    ids=[c[0] for c in CASES],
)
async def test_hinglish(scenario: str, reply: str, action: Action, rule_id: str | None):
    await _check(hinglish, reply, action, rule_id)


async def _check(guard: Guard, reply: str, action: Action, rule_id: str | None) -> None:
    verdict = await guard.check(reply)
    assert verdict.action is action, verdict.violations
    assert verdict.rule_ids == ((rule_id,) if rule_id else ())
    if action is Action.BLOCK:
        assert verdict.text == FALLBACK
    elif action is Action.REDACT:
        assert "[phone number]" in verdict.text


async def test_word_lists_cannot_read_meaning():
    """A known limit of layer 1, in both languages: negations still match.

    The classifier and LLM-judge layers exist for this. If this test starts failing,
    the rule layer has learned something it shouldn't try to do.
    """
    assert (await english.check("Please don't stop taking it.")).blocked
    assert (await hinglish.check("Dawai band kar do, aisa kabhi mat sochiye.")).blocked
