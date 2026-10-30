import re

from hypothesis import given
from hypothesis import strategies as st

from pipecat_guardrails.text.normalize import is_word_char, normalize, word_char_class


def test_casefold_and_whitespace_collapse():
    n = normalize("  Acme \t\n  CORP ")
    assert n.text == "acme corp "


def test_nfkc_compatibility_characters():
    # full-width letters and the "fi" ligature normalise to plain ASCII
    assert normalize("ＡＣＭＥ ﬁle").text == "acme file"  # noqa: RUF001


def test_invisible_characters_and_nukta_removed():
    assert normalize(f"a{chr(0x200B)}b{chr(0xAD)}c").text == "abc"
    # ज़ (ja + nukta) and the precomposed form both become ज
    assert normalize("ज" + chr(0x93C) + "rur").text == normalize(chr(0x95B) + "rur").text == "जrur"


def test_typographic_quotes_and_dashes_become_ascii():
    curly = "Don" + chr(0x2019) + "t worry " + chr(0x2014) + " " + chr(0x201C) + "ok" + chr(0x201D)
    assert normalize(curly).text == 'don\'t worry - "ok"'


def test_map_back_to_original():
    original = "Hello  ACME Corp!"
    n = normalize(original)
    start = n.text.index("acme corp")
    s, e = n.to_original(start, start + len("acme corp"))
    assert original[s:e] == "ACME Corp"


def test_map_back_covers_whole_devanagari_cluster():
    original = "आप दवाई लीजिए"
    n = normalize(original)
    i = n.text.index("दवाई")
    s, e = n.to_original(i, i + len("दवाई"))
    assert original[s:e] == "दवाई"


def test_devanagari_vowel_signs_are_word_characters():
    assert is_word_char(chr(0x93E))  # the vowel sign ा
    assert is_word_char("क")
    assert not is_word_char(" ")
    assert not is_word_char("।")
    word = word_char_class()
    assert re.fullmatch(f"{word}+", "दवाई")
    # plain \b would split this word at its vowel signs
    assert re.findall(r"\b\w+\b", "दवाई") != ["दवाई"]


@given(st.text())
def test_spans_always_map_inside_original(text: str):
    n = normalize(text)
    assert len(n.starts) == len(n.ends) == len(n.text)
    for i in range(len(n.text)):
        s, e = n.to_original(i, i + 1)
        assert 0 <= s < e <= len(text)
