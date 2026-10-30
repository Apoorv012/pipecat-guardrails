"""Hinglish health bot: check whole replies before speaking them, without a Pipecat pipeline.

This is how an app with its own LLM loop uses the guard: generate a reply, check it,
then speak `verdict.text` (or regenerate when it is blocked).

Run it with:  uv run python examples/whole_reply_hinglish.py

The same bot in English: examples/whole_reply_english.py
"""

import asyncio

from pipecat_guardrails import Action, Guard
from pipecat_guardrails.packs import india
from pipecat_guardrails.rules import PII, BannedTerms, pii

guard = Guard(
    rules=[
        # Each rule is opt-in. Pick the ones you need.
        BannedTerms(
            ["dawai band kar do", "dose double", "stop taking", "double the dose"],
            id="no-dosage-change",
            word_pattern=india.hinglish_word_pattern,  # also matches "davai", "dawaai", ...
            variants={"dawai": ["दवाई", "दवा"]},
        ),
        BannedTerms(
            ["chinta mat", "sab theek ho jayega", "don't worry"],
            id="no-reassurance",
            word_pattern=india.hinglish_word_pattern,
        ),
        PII(pii.email(), india.aadhaar(), india.pan(), india.phone(), id="pii"),
    ],
    fallback="Iske baare mein aap apne doctor se baat kijiye.",
)

REPLIES = [
    "Kya aapne aaj subah ki dawai le li?",
    "Aap davaai band kar do, koi baat nahi.",
    "Cheenta mat kijiye, sab theek ho jaayega.",
    "Main aapko 9876543210 se call karungi.",
]


async def main() -> None:
    for reply in REPLIES:
        verdict = await guard.check(reply)
        print(f"{verdict.action.value:>6}  {reply}")
        if verdict.action is not Action.ALLOW:
            print(f"        -> says: {verdict.text}   (rules: {', '.join(verdict.rule_ids)})")


if __name__ == "__main__":
    asyncio.run(main())
