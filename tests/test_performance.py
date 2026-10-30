"""Rules run before every spoken sentence, so they must stay fast.

The budget here is generous (CI machines are slow and noisy); it exists to catch large
regressions, such as a pattern that backtracks badly. Typical numbers on a laptop are
around 0.2 ms per sentence.
"""

import time

from pipecat_guardrails import Guard
from pipecat_guardrails.packs import india
from pipecat_guardrails.rules import PII, BannedTerms, pii

TERMS = [
    "dawai band kar do",
    "dose double",
    "stop taking",
    "you might have",
    "sab theek ho jayega",
    "chinta mat",
    "ghabraiye mat",
    "everything will be fine",
] * 5

SENTENCES = [
    "Namaste ji, main aapki dawai ke baare mein pooch rahi hoon.",
    "Please take your evening tablet after dinner, as the doctor prescribed.",
    "Aapka agla appointment Monday ko hai, 9876543210 par call aayega.",
    "दवाई समय पर लीजिए और डॉक्टर से बात कीजिए।",
    "a" * 400 + " " + "1 2 3 4 5 6 7 8 9 " * 20,  # long, digit-heavy text
]


async def test_rules_stay_fast():
    guard = Guard(
        [
            BannedTerms(TERMS, id="terms", word_pattern=india.hinglish_word_pattern),
            PII(pii.email(), pii.phone(), pii.card(), *india.all_recognizers()),
        ]
    )
    times: list[float] = []
    for _ in range(50):
        for sentence in SENTENCES:
            t = time.perf_counter()
            await guard.check(sentence)
            times.append((time.perf_counter() - t) * 1000)
    times.sort()
    p95 = times[int(len(times) * 0.95)]
    assert p95 < 5, f"p95 {p95:.2f} ms"
