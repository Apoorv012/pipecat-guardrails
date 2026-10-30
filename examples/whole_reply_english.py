"""English health bot: check whole replies before speaking them, without a Pipecat pipeline.

This is how an app with its own LLM loop uses the guard: generate a reply, check it,
then speak `verdict.text` (or regenerate when it is blocked).

Run it with:  uv run python examples/whole_reply_english.py

The same bot in Hinglish: examples/whole_reply_hinglish.py
"""

import asyncio

from pipecat_guardrails import Action, Guard
from pipecat_guardrails.rules import PII, BannedTerms, pii

guard = Guard(
    rules=[
        # Each rule is opt-in. Pick the ones you need.
        BannedTerms(
            ["stop taking", "double the dose", "increase the dose", "skip your dose"],
            id="no-dosage-change",
        ),
        BannedTerms(
            ["don't worry", "everything will be fine", "nothing to worry about"],
            id="no-reassurance",
        ),
        PII(pii.email(), pii.phone(), pii.card(), id="pii"),  # redacts by default
    ],
    fallback="That's a question for your doctor. Shall I ask them to call you?",
)

REPLIES = [
    "Did you take your morning medicine today?",
    "You can stop taking it for a few days.",
    "Don’t worry, everything will be fine.",  # curly apostrophe, as LLMs often write it  # noqa: RUF001
    "I'll call you back on +1 415 555 0132.",
    # A limit of word lists: this safe sentence is blocked because it contains
    # "stop taking". The classifier and LLM-judge layers (coming next) handle meaning.
    "Please don't stop taking it without asking your doctor.",
]


async def main() -> None:
    for reply in REPLIES:
        verdict = await guard.check(reply)
        print(f"{verdict.action.value:>6}  {reply}")
        if verdict.action is not Action.ALLOW:
            print(f"        -> says: {verdict.text}   (rules: {', '.join(verdict.rule_ids)})")


if __name__ == "__main__":
    asyncio.run(main())
