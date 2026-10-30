"""Text normalisation for matching, with a map back to the original offsets.

Matching runs on a normalised copy of the text (NFKC, case-folded, whitespace collapsed,
curly quotes made ASCII, invisible characters removed) so that ``Acme  Corp`` matches
``acme corp``. Redaction must edit the *original* text, so every normalised character
remembers where it came from.
"""

from __future__ import annotations

import functools
import unicodedata
from dataclasses import dataclass

# Removed after normalisation: zero-width space/joiners, soft hyphen, BOM, Devanagari nukta.
# The nukta is dropped because writers (and LLMs) use it inconsistently: ज़रूर / जरूर.
_DROP = frozenset(map(chr, (0x200B, 0x200C, 0x200D, 0x00AD, 0xFEFF, 0x093C)))

# Typographic punctuation that NFKC keeps, mapped to ASCII so that a term written as
# "don't worry" matches an LLM's "don't worry" written with a curly apostrophe (U+2019).
_ASCII = str.maketrans(
    {0x2018: "'", 0x2019: "'", 0x201B: "'", 0x02BC: "'"}  # single quotes
    | {0x201C: '"', 0x201D: '"', 0x201E: '"'}  # double quotes
    | {0x2013: "-", 0x2014: "-"}  # en and em dashes
)


def is_mark(ch: str) -> bool:
    """True for combining marks (Unicode category M*), e.g. Devanagari vowel signs."""
    return unicodedata.category(ch).startswith("M")


def is_word_char(ch: str) -> bool:
    r"""True for characters that belong inside a word.

    Python's ``\w`` misses combining marks, so ``\b`` breaks Devanagari words apart
    (``दवाई`` would split at the vowel signs). This counts marks as word characters.
    """
    return ch.isalnum() or ch == "_" or is_mark(ch)


@functools.cache
def word_char_class() -> str:
    """A regex character class matching :func:`is_word_char` characters.

    Built once on first use by scanning the Basic and Supplementary Multilingual Planes
    for combining marks.
    """
    ranges: list[str] = []
    start: int | None = None
    prev = 0
    for cp in range(0x300, 0x20000):
        if unicodedata.category(chr(cp)).startswith("M"):
            if start is None:
                start = cp
            elif cp != prev + 1:
                ranges.append(_range(start, prev))
                start = cp
            prev = cp
    if start is not None:
        ranges.append(_range(start, prev))
    return "[\\w" + "".join(ranges) + "]"


def _range(a: int, b: int) -> str:
    return f"\\U{a:08x}" if a == b else f"\\U{a:08x}-\\U{b:08x}"


@dataclass(frozen=True, slots=True)
class NormalizedText:
    """A normalised string plus, for each of its characters, the source span.

    Parameters:
        text: The normalised text.
        original: The text it was made from.
        starts: For each character of ``text``, the start offset of its source in
            ``original``.
        ends: For each character of ``text``, the end offset of its source in ``original``.
    """

    text: str
    original: str
    starts: tuple[int, ...]
    ends: tuple[int, ...]

    def to_original(self, start: int, end: int) -> tuple[int, int]:
        """Map a non-empty ``[start, end)`` span of ``text`` to a span of ``original``."""
        if not 0 <= start < end <= len(self.text):
            raise ValueError(f"invalid span ({start}, {end}) for length {len(self.text)}")
        return self.starts[start], self.ends[end - 1]


def normalize(text: str) -> NormalizedText:
    """Normalise ``text`` for matching and keep a map back to the original offsets.

    Steps: runs of whitespace become one space (leading whitespace is dropped); each
    base character together with its combining marks is NFKC-normalised and case-folded
    as one unit; curly quotes and en/em dashes become ASCII; zero-width characters, soft
    hyphens and the Devanagari nukta are removed.
    """
    out: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    i, n = 0, len(text)
    while i < n:
        if text[i].isspace():
            j = i + 1
            while j < n and text[j].isspace():
                j += 1
            if out:
                out.append(" ")
                starts.append(i)
                ends.append(j)
            i = j
            continue
        j = i + 1
        while j < n and is_mark(text[j]):
            j += 1
        for ch in unicodedata.normalize("NFKC", text[i:j]).casefold().translate(_ASCII):
            if ch not in _DROP:
                out.append(ch)
                starts.append(i)
                ends.append(j)
        i = j
    return NormalizedText("".join(out), text, tuple(starts), tuple(ends))
