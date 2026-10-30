"""Regular-expression rule."""

from __future__ import annotations

import re
from collections.abc import Mapping

from pipecat_guardrails.rules.base import Match, Rule
from pipecat_guardrails.types import Action, CheckContext, OnError

PatternLike = str | re.Pattern[str]


class Regex(Rule):
    r"""Fire when any of the patterns matches.

    Patterns run on the original text, so offsets are exact. Prefer :class:`BannedTerms`
    for word lists; use this for structured text such as URLs or order numbers.

    Example::

        Regex({"url": r"https?://\S+"}, id="no-links", action="redact")
    """

    def __init__(
        self,
        patterns: PatternLike | Mapping[str, PatternLike],
        *,
        id: str,
        action: Action | str = Action.BLOCK,
        flags: re.RegexFlag = re.IGNORECASE,
        replacement: str = "[redacted]",
        on_error: OnError | None = None,
        enabled: bool = True,
    ):
        """Configure the patterns.

        Args:
            patterns: One pattern, or a mapping of name to pattern (the name appears in
                the violation reason).
            id: Unique rule id.
            action: What to do when a pattern matches.
            flags: Flags for patterns given as strings. Compiled patterns keep their own.
            replacement: Text used in place of a match when ``action`` is REDACT.
            on_error: What to do if matching raises.
            enabled: Set to False to keep the rule configured but switched off.
        """
        super().__init__(
            id=id, action=action, replacement=replacement, on_error=on_error, enabled=enabled
        )
        named = patterns if isinstance(patterns, Mapping) else {"pattern": patterns}
        if not named:
            raise ValueError("Regex needs at least one pattern")
        self._patterns: dict[str, re.Pattern[str]] = {
            name: p if isinstance(p, re.Pattern) else re.compile(p, flags)
            for name, p in named.items()
        }

    def find(self, text: str, context: CheckContext) -> list[Match]:
        """Return every non-empty match of every pattern."""
        return [
            Match(m.start(), m.end(), f"matched pattern {name!r}")
            for name, pattern in self._patterns.items()
            for m in pattern.finditer(text)
            if m.end() > m.start()
        ]
