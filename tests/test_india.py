import pytest

from pipecat_guardrails import CheckContext
from pipecat_guardrails.packs import india
from pipecat_guardrails.rules import PII

CTX = CheckContext()


def found(r: PII, text: str) -> list[str]:
    return [text[m.start : m.end] for m in r.find(text, CTX)]


def make_aadhaar(first11: str) -> str:
    return first11 + str(india.verhoeff_check_digit(first11))


def test_verhoeff_reference_example():
    # Worked example from the Verhoeff algorithm's Wikipedia page: 236 -> check digit 3.
    assert india.verhoeff_check_digit("236") == 3
    assert india.verhoeff_valid("2363")
    assert not india.verhoeff_valid("2364")


def test_aadhaar_formats_and_checksum():
    number = make_aadhaar("23456789012")
    spaced = f"{number[:4]} {number[4:8]} {number[8:]}"
    dashed = spaced.replace(" ", "-")
    r = PII(india.aadhaar())
    assert found(r, f"mera aadhaar {number} hai") == [number]
    assert found(r, f"number: {spaced}.") == [spaced]
    assert found(r, f"number: {dashed}.") == [dashed]
    wrong = number[:-1] + str((int(number[-1]) + 1) % 10)
    assert found(r, f"number {wrong}") == []
    assert found(r, f"mixed {number[:4]} {number[4:8]}-{number[8:]}") == []
    assert found(r, "starts with 1: " + make_aadhaar("13456789012")) == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("PAN is ABCPE1234F ok", ["ABCPE1234F"]),
        ("pan abcpe1234f", ["abcpe1234f"]),
        ("not a PAN: ABCXE1234F", []),  # X is not a holder type
        ("code ABCPE1234FG", []),
    ],
)
def test_pan(text: str, expected: list[str]):
    assert found(PII(india.pan()), text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("pay rahul.k@okhdfcbank now", ["rahul.k@okhdfcbank"]),
        ("pay to 9876543210@ybl.", ["9876543210@ybl"]),
        ("mail rahul@gmail.com", []),  # an email, not a UPI id
    ],
)
def test_upi(text: str, expected: list[str]):
    assert found(PII(india.upi()), text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("call +91 98765 43210 now", ["+91 98765 43210"]),
        ("call 9876543210", ["9876543210"]),
        ("call 09876-543210", ["09876-543210"]),
        ("call 987 654 3210.", ["987 654 3210"]),
        ("ref 98765432101", []),  # 11 digits: not a mobile number
        ("call 1234567890", []),  # Indian mobiles start with 6-9
    ],
)
def test_indian_phone(text: str, expected: list[str]):
    assert found(PII(india.phone()), text) == expected


def test_all_recognizers_together():
    aadhaar = make_aadhaar("34567890123")
    r = PII(*india.all_recognizers())
    text = f"Aadhaar {aadhaar}, PAN ABCPE1234F, UPI a.b@oksbi, phone 9876543210."
    assert found(r, text) == [aadhaar, "ABCPE1234F", "a.b@oksbi", "9876543210"]
