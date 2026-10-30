"""India pack: Indian personal data and Hinglish spelling variants.

Everything here is opt-in. Example::

    from pipecat_guardrails.packs import india
    from pipecat_guardrails.rules import PII, BannedTerms

    PII(india.aadhaar(), india.pan(), india.upi(), india.phone(), id="pii-india")
    BannedTerms(
        ["dawai band kar do"],
        id="no-stopping-medicine",
        word_pattern=india.hinglish_word_pattern,   # also matches "davai", "dawaai", ...
        variants={"dawai": ["दवाई", "दवा"]},
    )
"""

from __future__ import annotations

import re

from pipecat_guardrails.rules.pii import Recognizer

# --- Verhoeff checksum (used by Aadhaar numbers) ----------------------------------------

_D = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)
_P = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)
_INV = (0, 4, 3, 2, 1, 5, 6, 7, 8, 9)


def _digits(text: str) -> list[int]:
    return [int(c) for c in text if c.isdigit()]


def verhoeff_valid(number: str) -> bool:
    """True if the digits in ``number`` (check digit last) pass the Verhoeff checksum."""
    digits = _digits(number)
    if not digits:
        return False
    c = 0
    for i, d in enumerate(reversed(digits)):
        c = _D[c][_P[i % 8][d]]
    return c == 0


def verhoeff_check_digit(number: str) -> int:
    """Return the Verhoeff check digit to append to the digits in ``number``."""
    c = 0
    for i, d in enumerate(reversed(_digits(number))):
        c = _D[c][_P[(i + 1) % 8][d]]
    return _INV[c]


# --- Recognizers ------------------------------------------------------------------------


def aadhaar(*, label: str = "[Aadhaar number]") -> Recognizer:
    """Aadhaar numbers: 12 digits, first digit 2-9, valid Verhoeff checksum.

    Matches ``2345 6789 0123``, ``2345-6789-0123`` and ``234567890123``.
    """
    return Recognizer(
        "aadhaar",
        label,
        r"(?<![\d-])[2-9]\d{3}([ -]?)\d{4}\1\d{4}(?![\d-])",
        validate=verhoeff_valid,
        score=0.95,
    )


def pan(*, label: str = "[PAN]") -> Recognizer:
    """PAN card numbers: ``AAAPA1234A``, where the 4th letter is a valid holder type."""
    return Recognizer(
        "pan",
        label,
        re.compile(r"(?<![A-Za-z0-9])[A-Z]{3}[PCHFATBLJG][A-Z]\d{4}[A-Z](?![A-Za-z0-9])", re.I),
        score=0.9,
    )


def upi(*, label: str = "[UPI ID]") -> Recognizer:
    """UPI ids such as ``name@okhdfcbank`` or ``98xxxxxx10@ybl`` (not email addresses)."""
    return Recognizer(
        "upi",
        label,
        r"(?<![\w.-])[\w.-]{2,256}@[A-Za-z][A-Za-z0-9]{1,63}(?![\w@-]|\.[A-Za-z])",
        score=0.85,
    )


def phone(*, label: str = "[phone number]") -> Recognizer:
    """Indian mobile numbers: optional +91/0 prefix, then 10 digits starting with 6-9.

    Digits may be grouped with single spaces or dashes: ``98765 43210``, ``987-654-3210``.
    """
    return Recognizer(
        "phone_in",
        label,
        r"(?<![\w+])(?:(?:\+|00)91[ -]?|0)?[6-9](?:[ -]?\d){9}(?![\w])",
        score=0.9,
    )


def all_recognizers() -> list[Recognizer]:
    """Every recognizer in this pack: Aadhaar, PAN, UPI and Indian phone numbers."""
    return [aadhaar(), pan(), upi(), phone()]


# --- Hinglish spelling variants ---------------------------------------------------------


def hinglish_word_pattern(word: str) -> str:
    """Regex for a Roman-script Hindi word that tolerates common spelling variants.

    Hinglish has no standard spelling: ``dawai``, ``davai``, ``dawaai`` are the same word.
    This handles long and short vowels (``a``/``aa``, ``i``/``ee``/``ii``, ``u``/``oo``),
    ``v``/``w``, ``j``/``z``, ``f``/``ph`` and doubled consonants (``acha``/``accha``).
    Words that are not plain ASCII letters (Devanagari, numbers) are matched exactly.

    Use it as ``BannedTerms(..., word_pattern=hinglish_word_pattern)``.
    """
    if not (word.isascii() and word.isalpha()):
        return re.escape(word)
    w = word.lower()
    parts: list[str] = []
    i = 0
    while i < len(w):
        c = w[i]
        run = 1
        while i + run < len(w) and w[i + run] == c:
            run += 1
        if c == "p" and run == 1 and w.startswith("ph", i):
            parts.append("(?:ph|f)")
            i += 2
            continue
        parts.append(_VARIANTS.get((c, min(run, 2)), f"{c}{{1,2}}"))
        i += run
    return "".join(parts)


_VARIANTS: dict[tuple[str, int], str] = {
    ("a", 1): "a{1,2}",
    ("a", 2): "a{1,2}",
    ("i", 1): "(?:i{1,2}|ee)",
    ("i", 2): "(?:i{1,2}|ee)",
    ("e", 1): "e",
    ("e", 2): "(?:ee|i{1,2})",
    ("u", 1): "(?:u{1,2}|oo)",
    ("u", 2): "(?:u{1,2}|oo)",
    ("o", 1): "o",
    ("o", 2): "(?:oo|u{1,2})",
    ("v", 1): "[vw]",
    ("v", 2): "[vw]{1,2}",
    ("w", 1): "[vw]",
    ("w", 2): "[vw]{1,2}",
    ("j", 1): "[jz]",
    ("z", 1): "[jz]",
    ("f", 1): "(?:f|ph)",
}
