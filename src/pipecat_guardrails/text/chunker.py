"""Streaming sentence splitter.

The guard checks LLM output one sentence at a time, before it is spoken. Text arrives in
small pieces (tokens), so the splitter must decide where sentences end without seeing the
rest of the reply, and must not cut in the wrong place:

- ``Take 2.5 mg``, ``at 8.30``: a dot followed by a digit is not an end.
- ``Dr. Rao``, ``Tab. Crocin``, ``J. K.``: known abbreviations and single initials are not ends.
- ``Well... maybe``: an ellipsis followed by a lowercase word is not an end.
- ``।`` and ``॥`` (Devanagari danda), ``?``, ``!`` and newlines are ends.

Chunks are exact consecutive pieces of the input: joining every chunk returned by
:meth:`SentenceChunker.push` and :meth:`SentenceChunker.flush` gives back the input
(except trailing whitespace-only text, which ``flush`` drops). Whitespace after a sentence
end goes to the start of the next chunk.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

DEFAULT_ABBREVIATIONS: frozenset[str] = frozenset(
    {
        # titles and honorifics
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "shri", "smt",
        # common abbreviations ("no" is left out: "I said no." ends sentences far more often)
        "vs", "e.g", "i.e", "approx", "fig", "dept", "a.m", "p.m",
        # medicine and units (spoken by health agents)
        "mg", "mcg", "ml", "kg", "gm", "tab", "tabs", "cap", "caps", "inj", "syp",
    }
)  # fmt: skip
"""Words that end with a dot without ending the sentence (lowercase, without the final dot)."""

_TERMINATORS = ".?!।॥…。！？"  # noqa: RUF001 - full-width CJK punctuation is intentional
_CLOSERS = "\"')]}”’»"  # noqa: RUF001 - curly quotes are intentional
_CANDIDATE = re.compile(r"[.?!।॥…。！？\n]")  # noqa: RUF001
_LAST_WORD = re.compile(r"(\S+)$")


class SentenceChunker:
    """Split streamed text into sentences.

    Example::

        chunker = SentenceChunker()
        for token in llm_tokens:
            for sentence in chunker.push(token):
                check_and_speak(sentence)
        if rest := chunker.flush():
            check_and_speak(rest)
    """

    def __init__(self, *, abbreviations: Iterable[str] = DEFAULT_ABBREVIATIONS):
        """Create a chunker.

        Args:
            abbreviations: Words (lowercase, without their final dot) after which a dot
                does not end a sentence.
        """
        self._abbreviations = frozenset(a.lower().rstrip(".") for a in abbreviations)
        self._buf = ""
        self._scan = 0  # no sentence end can start before this offset in _buf

    @property
    def pending(self) -> str:
        """Text received but not yet returned as a sentence."""
        return self._buf

    def push(self, text: str) -> list[str]:
        """Add text and return every sentence that is now complete."""
        self._buf += text
        out: list[str] = []
        while True:
            cut = self._find_cut()
            if cut is None:
                break
            out.append(self._buf[:cut])
            self._buf = self._buf[cut:]
            self._scan = 0
        return out

    def flush(self) -> str | None:
        """Return the remaining text as a final sentence, or None if it is only whitespace."""
        rest, self._buf, self._scan = self._buf, "", 0
        return rest if rest.strip() else None

    def reset(self) -> None:
        """Drop any pending text, e.g. when the user interrupts the bot."""
        self._buf, self._scan = "", 0

    def _find_cut(self) -> int | None:
        """Return the offset after the first complete sentence in the buffer, if any.

        Also advances ``self._scan`` past candidates that are decided "not an end", so long
        replies are not rescanned on every token.
        """
        buf, n = self._buf, len(self._buf)
        for m in _CANDIDATE.finditer(buf, self._scan):
            i = m.start()
            if buf[i] == "\n":
                if buf[: i + 1].strip():
                    return i + 1
                continue  # leading blank lines stay with the next sentence

            # The run of terminators, then any closing quotes or brackets.
            j = i
            while j < n and buf[j] in _TERMINATORS:
                j += 1
            run_end = j
            while j < n and buf[j] in _CLOSERS:
                j += 1
            if j == n:
                self._scan = i
                return None  # undecided until more text arrives
            if not buf[j].isspace():
                continue  # "2.5", "file.txt", "?!x": not an end here

            run = buf[i:run_end]
            if set(run) <= {".", "…"}:
                if run == "." and self._is_abbreviation(buf[:i]):
                    continue
                # A dot or ellipsis followed by a lowercase word continues the sentence.
                k = j
                while k < n and buf[k].isspace():
                    k += 1
                if k == n:
                    self._scan = i
                    return None  # need the next word to decide
                if buf[k].islower():
                    continue
            if not buf[:j].strip(" \t\r\n" + _TERMINATORS):
                continue  # nothing but punctuation so far
            return j
        self._scan = n
        return None

    def _is_abbreviation(self, before: str) -> bool:
        m = _LAST_WORD.search(before)
        if m is None:
            return False
        word = m.group(1).lstrip("\"'([{“‘«")  # noqa: RUF001
        if len(word) == 1 and word.isalpha() and word.isupper():
            return True  # an initial, as in "J. K. Rowling"
        return word.lower() in self._abbreviations


def split_sentences(
    text: str, *, abbreviations: Iterable[str] = DEFAULT_ABBREVIATIONS
) -> list[str]:
    """Split a complete text into sentences, with the same rules as :class:`SentenceChunker`."""
    chunker = SentenceChunker(abbreviations=abbreviations)
    out = chunker.push(text)
    if rest := chunker.flush():
        out.append(rest)
    return out
