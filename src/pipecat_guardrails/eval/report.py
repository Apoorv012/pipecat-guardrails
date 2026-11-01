"""Datasets, scoring and report formatting for the eval command."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from pipecat_guardrails.types import CheckContext, Verdict

REPORT_VERSION = 1


class DatasetError(ValueError):
    """The dataset file is missing or malformed."""


@dataclass(frozen=True, slots=True)
class Example:
    """One labelled line of the dataset.

    Parameters:
        line: Line number in the file, for error messages and the mistakes list.
        text: The bot reply to check.
        labels: Ids that should fire. Empty means the text is fine.
        context: Built from the optional ``history`` and ``lang`` fields.
        notes: Free text, shown next to mistakes.
    """

    line: int
    text: str
    labels: frozenset[str]
    context: CheckContext = field(default_factory=CheckContext)
    notes: str | None = None


def load_dataset(path: str | Path) -> list[Example]:
    """Read a JSONL dataset: one ``{"text": ..., "labels": [...]}`` object per line.

    Optional fields: ``lang``, ``history`` (OpenAI-format messages before the reply) and
    ``notes``. Blank lines are skipped.
    """
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as e:
        raise DatasetError(f"cannot read {path}: {e}") from e
    examples: list[Example] = []
    for number, raw in enumerate(lines, 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as e:
            raise DatasetError(f"{path}:{number}: not valid JSON ({e.msg})") from e
        if not isinstance(row, dict):
            raise DatasetError(f"{path}:{number}: expected a JSON object")
        row = cast(dict[str, Any], row)
        text = row.get("text")
        labels = row.get("labels", [])
        history = row.get("history", [])
        if not isinstance(text, str) or not text.strip():
            raise DatasetError(f'{path}:{number}: "text" must be a non-empty string')
        if not isinstance(labels, list) or not all(isinstance(x, str) for x in labels):  # pyright: ignore[reportUnknownVariableType]
            raise DatasetError(f'{path}:{number}: "labels" must be a list of strings')
        if not isinstance(history, list):
            raise DatasetError(f'{path}:{number}: "history" must be a list of messages')
        lang = row.get("lang")
        notes = row.get("notes")
        examples.append(
            Example(
                line=number,
                text=text,
                labels=frozenset(cast(list[str], labels)),
                context=CheckContext(
                    history=cast(list[dict[str, Any]], history),
                    language=lang if isinstance(lang, str) else None,
                ),
                notes=notes if isinstance(notes, str) else None,
            )
        )
    if not examples:
        raise DatasetError(f"{path}: no examples")
    return examples


def category_of(rail_id: str, categories: Iterable[str]) -> str | None:
    """The label a rail counts for: its own id, or the part before ``":"``.

    A rule ``no-med-advice:terms`` and a policy ``no-med-advice`` both count for the label
    ``no-med-advice``, so a dataset can be scored the same way whichever layer catches it.
    """
    for c in categories:
        if rail_id == c or rail_id.startswith(c + ":"):
            return c
    return None


@dataclass(slots=True)
class Counts:
    """Confusion counts for one label."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def add(self, expected: bool, got: bool) -> None:
        """Count one decision."""
        if expected and got:
            self.tp += 1
        elif got:
            self.fp += 1
        elif expected:
            self.fn += 1
        else:
            self.tn += 1

    @property
    def precision(self) -> float | None:
        """Of the texts flagged, the share that should have been. ``None`` if none were."""
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else None

    @property
    def recall(self) -> float | None:
        """Of the texts that should be flagged, the share that were. ``None`` if none."""
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else None

    @property
    def f1(self) -> float | None:
        """Harmonic mean of precision and recall. ``None`` if nothing was expected or flagged."""
        if not self.tp + self.fp + self.fn:
            return None
        return 2 * self.tp / (2 * self.tp + self.fp + self.fn)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready counts and scores."""
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


@dataclass(frozen=True, slots=True)
class Outcome:
    """What one guard did with one example, in one run."""

    example: Example
    fired: frozenset[str]  # rails that fired, not counting failed checks
    failed: frozenset[str]  # rails whose check failed
    latency_ms: Mapping[str, float]

    @classmethod
    def from_verdict(cls, example: Example, verdict: Verdict) -> Outcome:
        """Summarise a verdict."""
        return cls(
            example,
            frozenset(v.rule_id for v in verdict.violations if not v.error),
            frozenset(e.rule_id for e in verdict.errors),
            dict(verdict.latency_ms),
        )


def percentile(values: Sequence[float], q: float) -> float | None:
    """Nearest-rank percentile, ``q`` from 0 to 100. ``None`` for no values."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return ordered[rank - 1]


def summarize(name: str, outcomes: Sequence[Outcome], labels: Iterable[str]) -> dict[str, Any]:
    """Score one guard's outcomes against the labels."""
    categories = sorted(set(labels))
    per_label = {c: Counts() for c in categories}
    overall = Counts()
    unlabelled: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    latency: dict[str, list[float]] = {}
    mistakes: list[dict[str, Any]] = []
    checks_with_errors = 0

    for o in outcomes:
        got: set[str] = set()
        for rail in o.fired:
            c = category_of(rail, categories)
            if c is None:
                unlabelled[rail] += 1
            else:
                got.add(c)
        for c in categories:
            per_label[c].add(c in o.example.labels, c in got)
        overall.add(bool(o.example.labels), bool(o.fired))
        if o.failed:
            checks_with_errors += 1
            errors.update(o.failed)
        for key in ("total", "classifier", "judge"):
            if key in o.latency_ms:
                latency.setdefault(key, []).append(o.latency_ms[key])
        expected = set(o.example.labels)
        if got != expected or (not expected and o.fired):
            mistakes.append(
                {
                    "line": o.example.line,
                    "text": o.example.text,
                    "expected": sorted(expected),
                    "got": sorted(got | {r for r in o.fired if r in unlabelled}),
                    "rails": sorted(o.fired),
                    "failed": sorted(o.failed),
                    "notes": o.example.notes,
                }
            )

    return {
        "name": name,
        "labels": {c: per_label[c].to_dict() for c in categories},
        "any": overall.to_dict(),
        "latency_ms": {
            key: {
                "p50": percentile(values, 50),
                "p95": percentile(values, 95),
                "mean": sum(values) / len(values),
            }
            for key, values in latency.items()
        },
        "errors": {
            "checks": checks_with_errors,
            "rate": checks_with_errors / len(outcomes) if outcomes else 0.0,
            "by_rail": dict(errors),
        },
        "unlabelled_rails": dict(unlabelled),
        "mistakes": mistakes,
    }


# --- text output ------------------------------------------------------------------------


def _num(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]], indent: str = "  ") -> str:
    widths = [max(len(str(r[i])) for r in [header, *rows]) for i in range(len(header))]
    lines: list[str] = []
    for r in [header, *rows]:
        cells = [
            str(cell).ljust(w) if i == 0 else str(cell).rjust(w)
            for i, (cell, w) in enumerate(zip(r, widths, strict=True))
        ]
        lines.append(indent + "  ".join(cells).rstrip())
    return "\n".join(lines)


def format_report(report: Mapping[str, Any], *, verbose: bool = False) -> str:
    """The report as plain text for the terminal."""
    out: list[str] = []
    runs = report["repeat"]
    out.append(
        f"Dataset: {report['dataset']} ({report['examples']} examples, "
        f"{runs} run{'s' if runs != 1 else ''} each)"
    )
    for g in report["guards"]:
        out += ["", g["name"]]
        rows: list[list[str]] = []
        for label, c in [*g["labels"].items(), ("any (flagged at all)", g["any"])]:
            rows.append(
                [
                    label,
                    _num(c["precision"]),
                    _num(c["recall"]),
                    _num(c["f1"]),
                    str(c["tp"]),
                    str(c["fp"]),
                    str(c["fn"]),
                ]
            )
        out.append(_table(["label", "precision", "recall", "f1", "tp", "fp", "fn"], rows))
        lat = g["latency_ms"]
        parts = [
            f"{key} p50 {_num(lat[key]['p50'], 1)} / p95 {_num(lat[key]['p95'], 1)}"
            for key in ("total", "classifier", "judge")
            if key in lat
        ]
        out.append(f"  latency per check, ms: {'; '.join(parts)}")
        err = g["errors"]
        if err["checks"]:
            by_rail = ", ".join(f"{k} {v}" for k, v in sorted(err["by_rail"].items()))
            out.append(f"  errors: {err['checks']} checks ({err['rate']:.0%}); {by_rail}")
        else:
            out.append("  errors: none")
        if g["unlabelled_rails"]:
            fired = ", ".join(f"{k} {v}x" for k, v in sorted(g["unlabelled_rails"].items()))
            out.append(f"  fired without a matching label in the data: {fired}")
        mistakes = g["mistakes"]
        if not verbose:
            if mistakes:
                out.append(f"  mistakes: {len(mistakes)} (add --verbose to list them)")
        else:
            out.append(f"  mistakes: {len(mistakes)}")
            for m in mistakes:
                expected = ", ".join(m["expected"]) or "nothing"
                got = ", ".join(m["got"]) or "nothing"
                if m["rails"] and m["rails"] != m["got"]:
                    got += f" (by {', '.join(m['rails'])})"
                note = f"  [{m['notes']}]" if m["notes"] else ""
                failed = f"  (failed: {', '.join(m['failed'])})" if m["failed"] else ""
                out.append(f"    line {m['line']}: expected {expected}, got {got}{failed}{note}")
                out.append(f"      {m['text']}")

    if len(report["guards"]) > 1:
        out += ["", "Summary (any = flagged at all)"]
        rows = []
        for g in report["guards"]:
            total = g["latency_ms"].get("total", {})
            rows.append(
                [
                    g["name"],
                    _num(g["any"]["precision"]),
                    _num(g["any"]["recall"]),
                    _num(g["any"]["f1"]),
                    _num(total.get("p50"), 1),
                    _num(total.get("p95"), 1),
                    f"{g['errors']['rate']:.0%}",
                ]
            )
        out.append(
            _table(["guard", "precision", "recall", "f1", "p50 ms", "p95 ms", "errors"], rows)
        )
    return "\n".join(out)


def diff_reports(
    current: Mapping[str, Any], baseline: Mapping[str, Any], *, max_drop: float = 0.0
) -> tuple[str, bool]:
    """Compare F1 per guard and label with a baseline report.

    Returns:
        The comparison as text, and whether any F1 dropped by more than ``max_drop``.
    """
    base = {g["name"]: g for g in baseline.get("guards", [])}
    rows: list[list[str]] = []
    regressed = False
    for g in current["guards"]:
        old = base.get(g["name"])
        if old is None:
            continue
        pairs = [(label, c, old["labels"].get(label)) for label, c in g["labels"].items()]
        pairs.append(("any", g["any"], old["any"]))
        for label, new_c, old_c in pairs:
            new_f1 = new_c["f1"]
            old_f1 = old_c["f1"] if old_c else None
            if new_f1 is None or old_f1 is None:
                change = "-"
            else:
                delta = new_f1 - old_f1
                change = f"{delta:+.2f}"
                if delta < -max_drop - 1e-9:
                    regressed = True
                    change += "  REGRESSION"
            rows.append([g["name"], label, _num(old_f1), _num(new_f1), change])
    if not rows:
        return "Baseline has no guards with the same names; nothing to compare.", False
    text = "Compared with the baseline (f1)\n" + _table(
        ["guard", "label", "before", "after", "change"], rows
    )
    return text, regressed
