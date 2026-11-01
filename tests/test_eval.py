"""The eval command: datasets, scoring, and the command line."""

import json
import textwrap
from pathlib import Path
from typing import Any

import pytest

from pipecat_guardrails import Guard
from pipecat_guardrails.eval import (
    Counts,
    Example,
    Outcome,
    category_of,
    load_dataset,
    load_guard,
    main,
    summarize,
)
from pipecat_guardrails.eval.report import DatasetError, diff_reports, percentile

ROOT = Path(__file__).parent.parent
SAMPLE = ROOT / "examples" / "eval" / "sample.jsonl"
GUARDS = ROOT / "examples" / "eval" / "guards.py"


def write(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(textwrap.dedent(content), encoding="utf-8")
    return path


def jsonl(tmp_path: Path, *rows: dict[str, Any]) -> Path:
    path = tmp_path / "data.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


# --- dataset ----------------------------------------------------------------------------


def test_load_dataset(tmp_path: Path):
    path = jsonl(
        tmp_path,
        {"text": "Fine.", "labels": []},
        {"text": "Bad.", "labels": ["x"], "lang": "en", "notes": "n",
         "history": [{"role": "user", "content": "hi"}]},
    )  # fmt: skip
    path.write_text(path.read_text() + "\n\n", encoding="utf-8")  # blank lines are skipped
    fine, bad = load_dataset(path)
    assert (fine.line, fine.labels) == (1, frozenset())
    assert bad.labels == {"x"}
    assert bad.context.language == "en"
    assert bad.context.history == [{"role": "user", "content": "hi"}]
    assert bad.notes == "n"


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("not json", "not valid JSON"),
        ("[1]", "JSON object"),
        ('{"labels": []}', '"text"'),
        ('{"text": "t", "labels": "x"}', '"labels"'),
        ('{"text": "t", "history": "x"}', '"history"'),
    ],
)
def test_bad_dataset_lines(tmp_path: Path, line: str, message: str):
    path = write(tmp_path, "d.jsonl", line + "\n")
    with pytest.raises(DatasetError, match=message):
        load_dataset(path)


def test_empty_or_missing_dataset(tmp_path: Path):
    with pytest.raises(DatasetError, match="no examples"):
        load_dataset(write(tmp_path, "d.jsonl", "\n"))
    with pytest.raises(DatasetError, match="cannot read"):
        load_dataset(tmp_path / "missing.jsonl")


def test_sample_dataset_loads():
    examples = load_dataset(SAMPLE)
    assert len(examples) > 40
    assert {label for e in examples for label in e.labels} == {
        "no-competitors",
        "no-med-advice",
        "pii",
    }


# --- scoring ----------------------------------------------------------------------------


def test_category_of():
    labels = ["no-med-advice", "pii"]
    assert category_of("no-med-advice", labels) == "no-med-advice"
    assert category_of("no-med-advice:terms", labels) == "no-med-advice"
    assert category_of("no-med-advice-2", labels) is None
    assert category_of("other", labels) is None


def test_counts():
    c = Counts()
    for expected, got in [(True, True), (True, True), (True, False), (False, True), (False, False)]:
        c.add(expected, got)
    assert (c.tp, c.fn, c.fp, c.tn) == (2, 1, 1, 1)
    assert c.precision == pytest.approx(2 / 3)
    assert c.recall == pytest.approx(2 / 3)
    assert c.f1 == pytest.approx(2 / 3)
    assert Counts().precision is None
    assert Counts().f1 is None
    assert Counts(fp=1, fn=1).f1 == 0.0


def test_percentile():
    assert percentile([], 50) is None
    assert percentile([3.0, 1.0, 2.0], 50) == 2.0
    assert percentile([float(i) for i in range(1, 101)], 95) == 95.0


def outcome(text: str, labels: set[str], fired: set[str], failed: set[str] = set()) -> Outcome:  # noqa: B006
    example = Example(line=1, text=text, labels=frozenset(labels))
    return Outcome(example, frozenset(fired), frozenset(failed), {"total": 1.0})


def test_summarize():
    report = summarize(
        "g",
        [
            outcome("a", {"x"}, {"x:terms"}),
            outcome("b", {"x"}, set()),
            outcome("c", set(), {"other"}),
            outcome("d", set(), set(), failed={"x"}),
        ],
        ["x"],
    )
    assert report["labels"]["x"] == Counts(tp=1, fn=1, tn=2).to_dict()
    assert report["any"]["tp"] == 1
    assert report["any"]["fp"] == 1
    assert report["unlabelled_rails"] == {"other": 1}
    assert report["errors"] == {"checks": 1, "rate": 0.25, "by_rail": {"x": 1}}
    assert [m["text"] for m in report["mistakes"]] == ["b", "c"]
    assert report["mistakes"][1]["rails"] == ["other"]
    assert report["latency_ms"]["total"]["p50"] == 1.0


def test_diff_reports_flags_regressions():
    def rep(f1: float) -> dict[str, Any]:
        c = {"f1": f1}
        return {"guards": [{"name": "g", "labels": {"x": c}, "any": c}]}

    text, regressed = diff_reports(rep(0.8), rep(0.9))
    assert regressed
    assert "REGRESSION" in text
    assert not diff_reports(rep(0.85), rep(0.9), max_drop=0.1)[1]
    assert not diff_reports(rep(0.9), rep(0.8))[1]
    assert "nothing to compare" in diff_reports(rep(0.9), {"guards": []})[0]


# --- loading guards ---------------------------------------------------------------------


GUARD_FILE = """
    from fakes import FakeClassifier

    from pipecat_guardrails import Guard
    from pipecat_guardrails.rules import BannedTerms

    plain = Guard([BannedTerms(["acme"], id="no-competitors")])
    not_a_guard = 42

    def judged():
        judge = FakeClassifier(judge=lambda state, name: 0.9 if "pills" in str(state) else 0.1)
        return Guard(
            [BannedTerms(["acme"], id="no-competitors")],
            policies={"no-med-advice": "Medication advice?"},
            judge=judge,
        )

    async def made_async():
        return plain
"""


async def test_load_guard_from_file(tmp_path: Path):
    path = write(tmp_path, "my_guards.py", GUARD_FILE)
    name, guard = await load_guard(f"{path}:plain")
    assert name == "plain"
    assert isinstance(guard, Guard)
    assert isinstance((await load_guard(f"{path}:judged"))[1], Guard)
    assert isinstance((await load_guard(f"{path}:made_async"))[1], Guard)


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ("no-colon", "must look like"),
        ("{file}:missing", "has no"),
        ("{file}:not_a_guard", "not a Guard"),
        ("nowhere/guards.py:g", "no such file"),
        ("no_such_module_xyz:g", "cannot import"),
    ],
)
async def test_load_guard_errors(tmp_path: Path, spec: str, message: str):
    from pipecat_guardrails.eval.cli import GuardSpecError

    path = write(tmp_path, "my_guards.py", GUARD_FILE)
    with pytest.raises(GuardSpecError, match=message):
        await load_guard(spec.format(file=path))


async def test_load_guard_from_module():
    name, guard = await load_guard("pipecat_guardrails.eval.cli:Guard")  # a callable
    assert name == "Guard"
    assert isinstance(guard, Guard)


# --- command line -----------------------------------------------------------------------


def test_cli_on_sample(capsys: pytest.CaptureFixture[str], tmp_path: Path):
    out_json = tmp_path / "report.json"
    status = main(["eval", str(SAMPLE), "--guard", f"{GUARDS}:rules_only", "--json", str(out_json)])
    assert status == 0
    out = capsys.readouterr().out
    assert "rules_only" in out
    assert "no-med-advice" in out
    report = json.loads(out_json.read_text(encoding="utf-8"))
    g = report["guards"][0]
    assert g["labels"]["no-competitors"]["recall"] == 1.0
    assert g["labels"]["pii"]["recall"] == 1.0
    # Word lists miss paraphrases: the reason for the classifier and judge layers.
    assert g["labels"]["no-med-advice"]["recall"] < 0.5


def test_cli_runs_policies_even_after_a_rule_blocks(tmp_path: Path):
    data = jsonl(
        tmp_path,
        {"text": "Try acme and stop the pills.", "labels": ["no-competitors", "no-med-advice"]},
        {"text": "Hello.", "labels": []},
    )
    guards = write(tmp_path, "my_guards.py", GUARD_FILE)
    out_json = tmp_path / "r.json"
    status = main(["eval", str(data), "--guard", f"{guards}:judged", "--json", str(out_json)])
    assert status == 0
    g = json.loads(out_json.read_text(encoding="utf-8"))["guards"][0]
    assert g["labels"]["no-med-advice"]["tp"] == 1
    assert g["labels"]["no-competitors"]["tp"] == 1
    assert "judge" in g["latency_ms"]


def test_cli_compares_guards_and_repeats(capsys: pytest.CaptureFixture[str], tmp_path: Path):
    data = jsonl(tmp_path, {"text": "stop the pills", "labels": ["no-med-advice"]})
    guards = write(tmp_path, "my_guards.py", GUARD_FILE)
    out_json = tmp_path / "r.json"
    args = ["eval", str(data), "--guard", f"{guards}:plain", "--guard", f"{guards}:judged"]
    assert main([*args, "--repeat", "3", "--json", str(out_json)]) == 0
    assert "Summary" in capsys.readouterr().out
    report = json.loads(out_json.read_text(encoding="utf-8"))
    assert report["repeat"] == 3
    assert report["guards"][1]["labels"]["no-med-advice"]["tp"] == 3


def test_cli_diff(capsys: pytest.CaptureFixture[str], tmp_path: Path):
    data = jsonl(tmp_path, {"text": "stop the pills", "labels": ["no-med-advice"]})
    guards = write(tmp_path, "my_guards.py", GUARD_FILE)
    baseline = tmp_path / "baseline.json"
    assert main(["eval", str(data), "--guard", f"{guards}:judged", "--json", str(baseline)]) == 0
    # Same guard name, but now without the judge: recall drops, so the diff fails.
    renamed = write(tmp_path, "renamed.py", "from my_guards import plain as judged\n")
    import sys

    sys.path.insert(0, str(tmp_path))
    try:
        status = main(["eval", str(data), "--guard", f"{renamed}:judged", "--diff", str(baseline)])
    finally:
        sys.path.remove(str(tmp_path))
    assert status == 1
    assert "REGRESSION" in capsys.readouterr().out


def test_cli_usage_errors(capsys: pytest.CaptureFixture[str], tmp_path: Path):
    guards = f"{GUARDS}:rules_only"
    assert main(["eval", str(tmp_path / "missing.jsonl"), "--guard", guards]) == 2
    assert main(["eval", str(SAMPLE), "--guard", "bad"]) == 2
    assert main(["eval", str(SAMPLE), "--guard", guards, "--repeat", "0"]) == 2
    assert main(["eval", str(SAMPLE), "--guard", guards, "--guard", guards]) == 2
    bad_baseline = write(tmp_path, "b.json", "not json")
    assert main(["eval", str(SAMPLE), "--guard", guards, "--diff", str(bad_baseline)]) == 2
    assert "error:" in capsys.readouterr().err


def test_cli_label_filter(tmp_path: Path):
    out_json = tmp_path / "r.json"
    args = ["eval", str(SAMPLE), "--guard", f"{GUARDS}:rules_only", "--labels", "pii"]
    assert main([*args, "--json", str(out_json)]) == 0
    g = json.loads(out_json.read_text(encoding="utf-8"))["guards"][0]
    assert list(g["labels"]) == ["pii"]
