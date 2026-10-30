"""Banned terms: whole-word phrase matching that works for English and Indic scripts."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping

from pipecat_guardrails.rules.base import Match, Rule
from pipecat_guardrails.text.normalize import is_word_char, normalize, word_char_class
from pipecat_guardrails.types import Action, CheckContext, OnError

WordPattern = Callable[[str], str]
"""Turns one normalised word into a regex fragment, e.g. to allow spelling variants."""


class BannedTerms(Rule):
    """Fire when the text contains any of the given words or phrases.

    Matching is case-insensitive, ignores extra whitespace and Unicode presentation
    differences, and only matches whole words: ``"ass"`` does not match ``"class"``, and
    ``"दवा"`` does not match inside ``"दवाई"``.

    Example::

        BannedTerms(
            ["dawai band kar do", "stop taking"],
            id="no-stopping-medicine",
            variants={"dawai": ["davai", "दवाई"]},
        )
    """

    def __init__(
        self,
        terms: Iterable[str],
        *,
        id: str,
        action: Action | str = Action.BLOCK,
        variants: Mapping[str, Iterable[str]] | None = None,
        word_pattern: WordPattern | None = None,
        replacement: str = "[redacted]",
        on_error: OnError | None = None,
        enabled: bool = True,
    ):
        """Configure the terms.

        Args:
            terms: Words or phrases to look for.
            id: Unique rule id.
            action: What to do when a term is found.
            variants: Alternatives for single words, applied inside every term. With
                ``{"dawai": ["davai", "दवाई"]}``, the term ``"dawai band"`` also matches
                ``"davai band"`` and ``"दवाई band"``.
            word_pattern: Turns each word into a regex fragment, for fuzzy spellings. See
                :func:`pipecat_guardrails.packs.india.hinglish_word_pattern`.
            replacement: Text used in place of a match when ``action`` is REDACT.
            on_error: What to do if matching raises.
            enabled: Set to False to keep the rule configured but switched off.
        """
        super().__init__(
            id=id, action=action, replacement=replacement, on_error=on_error, enabled=enabled
        )
        word_alts: dict[str, list[str]] = {}
        for word, alts in (variants or {}).items():
            key = normalize(word).text.strip()
            word_alts[key] = [key, *(normalize(a).text.strip() for a in alts)]

        self.terms: tuple[str, ...] = tuple(dict.fromkeys(t for t in terms if t.strip()))
        if not self.terms:
            raise ValueError("BannedTerms needs at least one non-empty term")
        to_regex = word_pattern or re.escape
        word = word_char_class()
        self._patterns: list[tuple[str, re.Pattern[str]]] = []
        for term in self.terms:
            norm = normalize(term).text.strip()
            body = " ".join(_alternation(to_regex, word_alts.get(w, [w])) for w in norm.split(" "))
            lead = f"(?<!{word})" if is_word_char(norm[0]) else ""
            trail = f"(?!{word})" if is_word_char(norm[-1]) else ""
            self._patterns.append((term, re.compile(lead + body + trail)))

    def find(self, text: str, context: CheckContext) -> list[Match]:
        """Return every occurrence of every term, with offsets in the original text."""
        norm = normalize(text)
        matches: list[Match] = []
        for term, pattern in self._patterns:
            for m in pattern.finditer(norm.text):
                if m.end() > m.start():
                    start, end = norm.to_original(m.start(), m.end())
                    matches.append(Match(start, end, f"banned term {term!r}"))
        return matches


def _alternation(to_regex: WordPattern, alternatives: list[str]) -> str:
    parts = [" ".join(to_regex(w) for w in alt.split(" ")) for alt in alternatives]
    return parts[0] if len(parts) == 1 else "(?:" + "|".join(parts) + ")"
