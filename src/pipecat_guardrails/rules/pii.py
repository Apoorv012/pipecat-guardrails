"""Personal data detection (emails, phone numbers, card numbers, ...).

Nothing is detected unless you ask for it: pass the detectors you want to :class:`PII`.

Example::

    from pipecat_guardrails.packs import india
    from pipecat_guardrails.rules import PII, pii

    PII(pii.email(), pii.card(), india.aadhaar(), india.pan(), id="pii")
"""

from __future__ import annotations

import ipaddress
import re
from collections.abc import Callable
from dataclasses import dataclass

from pipecat_guardrails.rules.base import Match, Rule
from pipecat_guardrails.types import Action, CheckContext, OnError


@dataclass(frozen=True, slots=True)
class Detection:
    """One piece of personal data found by a :class:`Recognizer`."""

    start: int
    end: int
    score: float


class Recognizer:
    """Finds one kind of personal data with a regex and an optional validity check.

    Parameters:
        name: Kind of data, used in violation reasons (``"email"``).
        label: Replacement text used when redacting (``"[email]"``).
        pattern: Regex that finds candidates.
        validate: Extra check on the matched text (e.g. a checksum). Candidates that
            fail it are ignored.
        score: Confidence given to every detection, from 0 to 1. When detections
            overlap, the higher score wins.
    """

    def __init__(
        self,
        name: str,
        label: str,
        pattern: str | re.Pattern[str],
        *,
        validate: Callable[[str], bool] | None = None,
        score: float = 0.8,
    ):
        """Create a recognizer. See the class docstring for the arguments."""
        self.name = name
        self.label = label
        self.pattern = re.compile(pattern) if isinstance(pattern, str) else pattern
        self.validate = validate
        self.score = score

    def find(self, text: str) -> list[Detection]:
        """Return every valid detection in ``text``."""
        return [
            Detection(m.start(), m.end(), self.score)
            for m in self.pattern.finditer(text)
            if m.end() > m.start() and (self.validate is None or self.validate(m.group()))
        ]

    def __repr__(self) -> str:
        return f"Recognizer({self.name!r})"


class PII(Rule):
    """Fire when the text contains personal data found by any of the given recognizers.

    The default action is REDACT: each detection is replaced by its recognizer's label,
    e.g. ``"Call me at [phone number]"``. Use ``action="block"`` to stop the sentence instead.
    """

    def __init__(
        self,
        *recognizers: Recognizer,
        id: str = "pii",
        action: Action | str = Action.REDACT,
        on_error: OnError | None = None,
        enabled: bool = True,
    ):
        """Configure the recognizers.

        Args:
            *recognizers: The kinds of data to detect, e.g. ``pii.email()``. At least one.
            id: Unique rule id.
            action: What to do when personal data is found.
            on_error: What to do if detection raises.
            enabled: Set to False to keep the rule configured but switched off.
        """
        if not recognizers:
            raise ValueError("PII needs at least one recognizer, e.g. PII(pii.email())")
        super().__init__(
            id=id, action=action, replacement="[redacted]", on_error=on_error, enabled=enabled
        )
        self.recognizers: tuple[Recognizer, ...] = recognizers

    def find(self, text: str, context: CheckContext) -> list[Match]:
        """Return non-overlapping detections. Overlaps keep the higher score, then the longer."""
        found = [(d, r) for r in self.recognizers for d in r.find(text)]
        found.sort(key=lambda dr: (-dr[0].score, -(dr[0].end - dr[0].start), dr[0].start))
        chosen: list[tuple[Detection, Recognizer]] = []
        for d, r in found:
            if all(d.end <= c.start or d.start >= c.end for c, _ in chosen):
                chosen.append((d, r))
        chosen.sort(key=lambda dr: dr[0].start)
        return [
            Match(d.start, d.end, f"{r.name} detected", score=d.score, replacement=r.label)
            for d, r in chosen
        ]


# --- Checksums --------------------------------------------------------------------------


def luhn_valid(number: str) -> bool:
    """True if the digits in ``number`` pass the Luhn checksum used by payment cards."""
    digits = [int(c) for c in number if c.isdigit()]
    if not digits:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _digit_count(text: str) -> int:
    return sum(c.isdigit() for c in text)


# --- Generic recognizers ----------------------------------------------------------------


def email(*, label: str = "[email]") -> Recognizer:
    """Email addresses."""
    return Recognizer(
        "email",
        label,
        r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}(?![\w-])",
        score=0.95,
    )


_DATE = re.compile(r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}$|^\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}$")


def _valid_phone(text: str) -> bool:
    n = _digit_count(text)
    if _DATE.match(text.strip()):
        return False
    if text.startswith(("+", "00")):
        return 8 <= n <= 15
    return 10 <= n <= 15


def phone(*, label: str = "[phone number]") -> Recognizer:
    """International phone numbers: 8-15 digits with a ``+`` prefix, otherwise 10-15.

    Numbers with spaces, dots, dashes or brackets between digits are matched. For Indian
    numbers, :func:`pipecat_guardrails.packs.india.phone` is stricter and more precise.
    """
    return Recognizer(
        "phone",
        label,
        r"(?<![\w+])(?:\+|00)?\d(?:[ ().-]{0,2}\d){7,14}(?![\w])",
        validate=_valid_phone,
        score=0.5,
    )


def card(*, label: str = "[card number]") -> Recognizer:
    """Payment card numbers (13-19 digits, optionally grouped) that pass the Luhn check."""
    return Recognizer(
        "card",
        label,
        r"(?<![\d-])\d(?:[ -]?\d){12,18}(?![\d-])",
        validate=lambda s: 13 <= _digit_count(s) <= 19 and luhn_valid(s),
        score=0.9,
    )


def _valid_ip(text: str) -> bool:
    try:
        ipaddress.ip_address(text)
    except ValueError:
        return False
    return True


def ip_address(*, label: str = "[IP address]") -> Recognizer:
    """IPv4 addresses. Version strings like ``1.2.3.4`` also match, so use with care."""
    return Recognizer(
        "ip_address",
        label,
        r"(?<![\d.])\d{1,3}(?:\.\d{1,3}){3}(?![\d.])",
        validate=_valid_ip,
        score=0.6,
    )


def from_regex(
    name: str, pattern: str | re.Pattern[str], *, label: str | None = None, score: float = 0.8
) -> Recognizer:
    """Build a recognizer for your own kind of data, e.g. member or order ids."""
    return Recognizer(name, label or f"[{name}]", pattern, score=score)
