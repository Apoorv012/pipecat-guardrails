import re

import pytest

from pipecat_guardrails import CheckContext
from pipecat_guardrails.packs.india import hinglish_word_pattern
from pipecat_guardrails.rules import PII, BannedTerms, Match, Regex, pii, rule

CTX = CheckContext()


def found(r: BannedTerms | Regex | PII, text: str) -> list[str]:
    return [text[m.start : m.end] for m in r.find(text, CTX)]


# --- BannedTerms ----------------------------------------------------------------------


def test_terms_whole_words_only():
    r = BannedTerms(["ass"], id="t")
    assert found(r, "This class is great") == []
    assert found(r, "what an ass.") == ["ass"]


def test_terms_case_and_whitespace_insensitive():
    r = BannedTerms(["acme corp"], id="t")
    assert found(r, "Try ACME   Corp today") == ["ACME   Corp"]


def test_terms_devanagari_whole_words():
    r = BannedTerms(["दवा"], id="t")
    assert found(r, "यह दवाई है") == []  # दवा is only a prefix of दवाई
    assert found(r, "यह दवा है") == ["दवा"]


def test_terms_with_punctuation_edges():
    r = BannedTerms(["$$$", "Dr."], id="t")
    assert found(r, "Make $$$ fast") == ["$$$"]
    assert found(r, "Ask Dr. Rao") == ["Dr."]


def test_terms_variants_apply_inside_phrases():
    r = BannedTerms(["dawai band kar do"], id="t", variants={"dawai": ["davai", "दवाई"]})
    assert found(r, "Aap davai band kar do.") == ["davai band kar do"]
    assert found(r, "दवाई band kar do") == ["दवाई band kar do"]
    assert found(r, "dawai mat band karo") == []


def test_terms_hinglish_spelling():
    r = BannedTerms(["dawai band kar do"], id="t", word_pattern=hinglish_word_pattern)
    assert found(r, "Davaai band kar do ji") == ["Davaai band kar do"]
    assert found(r, "chinta mat kijiye") == []


def test_hinglish_pattern_examples():
    pattern = re.compile(hinglish_word_pattern("chinta") + "$")
    for spelling in ["chinta", "cheenta", "chintaa", "chhinta"]:
        assert pattern.match(spelling), spelling
    assert not pattern.match("chanta")
    assert re.fullmatch(hinglish_word_pattern("accha"), "acha")
    assert re.fullmatch(hinglish_word_pattern("zaroor"), "jarur")
    assert hinglish_word_pattern("दवाई") == "दवाई"


def test_terms_require_a_term():
    with pytest.raises(ValueError, match="at least one"):
        BannedTerms(["  "], id="t")


# --- Regex ----------------------------------------------------------------------------


def test_regex_named_patterns():
    r = Regex({"url": r"https?://\S+"}, id="links")
    [m] = r.find("see https://x.io now", CTX)
    assert (m.start, m.end) == (4, 16)
    assert "url" in m.reason


def test_regex_compiled_pattern_keeps_flags():
    r = Regex(re.compile("ACME"), id="t")
    assert found(r, "acme ACME") == ["ACME"]


# --- PII ------------------------------------------------------------------------------


def test_pii_requires_recognizers():
    with pytest.raises(ValueError, match="at least one recognizer"):
        PII()


@pytest.mark.parametrize(
    ("recognizer", "text", "expected"),
    [
        (pii.email(), "mail me at a.b+c@example.co.in today", ["a.b+c@example.co.in"]),
        (pii.card(), "card 4111 1111 1111 1111 ok", ["4111 1111 1111 1111"]),
        (pii.card(), "card 4111 1111 1111 1112 ok", []),  # fails Luhn
        (pii.phone(), "call +1 (415) 555-0132 now", ["+1 (415) 555-0132"]),
        (pii.phone(), "on 2026-10-31 at noon", []),  # a date, not a phone number
        (pii.phone(), "take 500 mg", []),
        (pii.ip_address(), "host 192.168.1.20 down", ["192.168.1.20"]),
        (pii.ip_address(), "host 999.1.1.1 down", []),
    ],
)
def test_generic_recognizers(recognizer: pii.Recognizer, text: str, expected: list[str]):
    assert found(PII(recognizer), text) == expected


def test_pii_overlaps_keep_the_stronger_detection():
    r = PII(pii.phone(), pii.card())
    [m] = r.find("pay with 4111111111111111 now", CTX)
    assert m.reason == "card detected"
    assert m.replacement == "[card number]"


def test_custom_recognizer():
    r = PII(pii.from_regex("member_id", r"\bMBR-\d{6}\b"))
    [m] = r.find("your id is MBR-123456", CTX)
    assert m.replacement == "[member_id]"


# --- function rules -------------------------------------------------------------------


def test_function_rule_return_types():
    @rule()
    def shouting(text: str) -> bool:
        return text.isupper()

    assert shouting.id == "shouting"
    assert shouting.find("HELLO", CTX) == [Match(0, 5, "rule shouting fired")]
    assert shouting.find("hello", CTX) == []

    @rule(id="len")
    def too_long(text: str) -> str | None:
        return "too long" if len(text) > 3 else None

    assert [m.reason for m in too_long.find("long text", CTX)] == ["too long"]


def test_function_rule_can_use_context():
    @rule(id="lang")
    def english_only(text: str, context: CheckContext) -> bool:
        return context.language not in (None, "en")

    assert english_only.find("hi", CheckContext(language="hi"))
    assert not english_only.find("hi", CheckContext(language="en"))


def test_function_rule_rejects_bad_results():
    @rule(id="bad")  # pyright: ignore[reportArgumentType] - wrong return type on purpose
    def bad(text: str) -> list[str]:
        return ["oops"]

    with pytest.raises(TypeError):
        bad.find("x", CTX)
