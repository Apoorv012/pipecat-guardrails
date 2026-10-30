import pytest
from hypothesis import given
from hypothesis import strategies as st

from pipecat_guardrails.text import SentenceChunker, split_sentences


def stripped(chunks: list[str]) -> list[str]:
    return [c.strip() for c in chunks]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hello there. How are you?", ["Hello there.", "How are you?"]),
        ("Take 2.5 mg daily. Then rest.", ["Take 2.5 mg daily.", "Then rest."]),
        ("Come at 8.30 tomorrow. Bye!", ["Come at 8.30 tomorrow.", "Bye!"]),
        ("Dr. Rao will call you. Okay?", ["Dr. Rao will call you.", "Okay?"]),
        ("Tab. Crocin lijiye. Aur paani.", ["Tab. Crocin lijiye.", "Aur paani."]),
        ("Smt. Sharma ji, namaste. Hi.", ["Smt. Sharma ji, namaste.", "Hi."]),
        ("J. K. Rowling wrote it. Yes.", ["J. K. Rowling wrote it.", "Yes."]),
        ("Well... maybe not. Okay.", ["Well... maybe not.", "Okay."]),
        ("Wait... That is it.", ["Wait...", "That is it."]),
        ("दवाई समय पर लीजिए। फिर आराम कीजिए।", ["दवाई समय पर लीजिए।", "फिर आराम कीजिए।"]),
        ("Really?! Yes.", ["Really?!", "Yes."]),
        ('He said "stop." Then left.', ['He said "stop."', "Then left."]),
        ("Line one\nLine two", ["Line one", "Line two"]),
        ("e.g. this one. Done.", ["e.g. this one.", "Done."]),
        ("I said no. Then left.", ["I said no.", "Then left."]),
        ("No ending", ["No ending"]),
        ("", []),
        ("   ", []),
    ],
)
def test_split_sentences(text: str, expected: list[str]):
    assert stripped(split_sentences(text)) == expected


def test_streaming_waits_for_decimal():
    c = SentenceChunker()
    assert c.push("Take 2.") == []  # could be "2.5"
    assert c.push("5 mg") == []
    assert c.push(" daily. ") == []  # need the next word to rule out "daily... and"
    assert stripped(c.push("Then")) == ["Take 2.5 mg daily."]
    assert c.flush() == " Then"


def test_reset_drops_pending_text():
    c = SentenceChunker()
    c.push("This was interrupt")
    c.reset()
    assert c.pending == ""
    assert c.flush() is None


def test_custom_abbreviations():
    assert stripped(split_sentences("Ask Capt. Singh. Ok.", abbreviations={"capt"})) == [
        "Ask Capt. Singh.",
        "Ok.",
    ]


SAMPLES = [
    "Take 2.5 mg daily. Dr. Rao will call... maybe. दवाई लीजिए। Okay?! Bye.\nNext line",
    "Aapki dawai ka time ho gaya hai. Kya aapne li? Haan ji.",
]


@pytest.mark.parametrize("text", SAMPLES)
@pytest.mark.parametrize("size", [1, 2, 3, 7])
def test_streaming_matches_whole_text(text: str, size: int):
    c = SentenceChunker()
    out: list[str] = []
    for i in range(0, len(text), size):
        out += c.push(text[i : i + size])
    if rest := c.flush():
        out.append(rest)
    assert out == split_sentences(text)


@given(st.text(alphabet=st.sampled_from("ab .?!।\n2…\"'")), st.integers(1, 5))
def test_chunks_are_lossless(text: str, size: int):
    c = SentenceChunker()
    out: list[str] = []
    for i in range(0, len(text), size):
        out += c.push(text[i : i + size])
    rest = c.flush()
    joined = "".join(out) + (rest or "")
    # Only trailing whitespace may be dropped.
    dropped = text[len(joined) :]
    assert joined == text or (rest is None and text.startswith(joined) and not dropped.strip())
    assert all(chunk.strip() for chunk in out)
