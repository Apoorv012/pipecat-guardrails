"""``pipecat-guardrails eval``: measure your rules and policies on your own labelled data.

Example::

    pipecat-guardrails eval data.jsonl --guard app.guards:guard
    pipecat-guardrails eval data.jsonl --guard guards.py:rules_only --guard guards.py:full

Each ``--guard`` names a :class:`~pipecat_guardrails.Guard`, or a function that returns
one, as ``module:name`` or ``path/to/file.py:name``.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import importlib.util
import inspect
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pipecat_guardrails.eval.report import (
    REPORT_VERSION,
    DatasetError,
    Example,
    Outcome,
    diff_reports,
    format_report,
    load_dataset,
    summarize,
)
from pipecat_guardrails.guard import Guard


class GuardSpecError(ValueError):
    """A ``--guard`` value that cannot be loaded."""


async def load_guard(spec: str) -> tuple[str, Guard]:
    """Load ``module:name`` or ``file.py:name``. Returns ``(name, guard)``."""
    target, sep, attr = spec.rpartition(":")
    if not sep or not target or not attr.isidentifier():
        raise GuardSpecError(f"--guard must look like module:name or file.py:name, not {spec!r}")
    if target.endswith(".py") or "/" in target or "\\" in target:
        path = Path(target)
        if not path.is_file():
            raise GuardSpecError(f"no such file: {target}")
        module_spec = importlib.util.spec_from_file_location(f"_guards_{path.stem}", path)
        if module_spec is None or module_spec.loader is None:
            raise GuardSpecError(f"cannot load {target}")
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_spec.name] = module
        module_spec.loader.exec_module(module)
    else:
        if "" not in sys.path:
            sys.path.insert(0, "")  # so that app.guards resolves from the current directory
        try:
            module = importlib.import_module(target)
        except ImportError as e:
            raise GuardSpecError(f"cannot import {target}: {e}") from e
    try:
        obj: Any = getattr(module, attr)
    except AttributeError as e:
        raise GuardSpecError(f"{target} has no {attr!r}") from e
    if callable(obj) and not isinstance(obj, Guard):
        obj = obj()
        if inspect.isawaitable(obj):
            obj = await obj
    if not isinstance(obj, Guard):
        raise GuardSpecError(f"{spec} is a {type(obj).__name__}, not a Guard")
    return attr, obj


async def run_guard(
    guard: Guard, examples: Sequence[Example], *, repeat: int = 1, concurrency: int = 4
) -> list[Outcome]:
    """Check every example ``repeat`` times, at most ``concurrency`` at a time.

    Every layer runs on every example, even after a rule blocks, so each rule and policy is
    scored on its own. Hooks and decision logs are skipped.
    """
    limit = asyncio.Semaphore(concurrency)

    async def one(example: Example) -> Outcome:
        async with limit:
            verdict = await guard._evaluate(  # pyright: ignore[reportPrivateUsage]
                example.text, example.context, stop_on_block=False
            )
        return Outcome.from_verdict(example, verdict)

    return list(await asyncio.gather(*(one(e) for e in examples for _ in range(repeat))))


def build_parser() -> argparse.ArgumentParser:
    """The command-line interface."""
    parser = argparse.ArgumentParser(
        prog="pipecat-guardrails",
        description="Output guardrails for Pipecat voice agents.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    ev = commands.add_parser(
        "eval",
        help="score guards on a labelled dataset",
        description=(
            "Score one or more guards on a JSONL dataset of bot replies. Each line is "
            '{"text": "...", "labels": ["rule-or-policy-id", ...]}; empty labels mean the '
            "reply is fine. A rail counts for a label if its id is the label or starts with "
            '"label:".'
        ),
    )
    ev.add_argument("dataset", help="JSONL file of labelled replies")
    ev.add_argument(
        "--guard",
        action="append",
        required=True,
        metavar="SPEC",
        help="module:name or file.py:name of a Guard (or a function returning one); "
        "repeat to compare guards",
    )
    ev.add_argument(
        "--labels",
        metavar="IDS",
        help="comma-separated labels to report (default: every label in the data)",
    )
    ev.add_argument("--repeat", type=int, default=1, help="check each example N times")
    ev.add_argument(
        "--concurrency", type=int, default=4, help="checks running at once (default: 4)"
    )
    ev.add_argument("--json", metavar="PATH", help="also write the report as JSON")
    ev.add_argument("--diff", metavar="BASELINE", help="compare F1 with an earlier --json report")
    ev.add_argument(
        "--max-drop",
        type=float,
        default=0.0,
        help="with --diff, exit with status 1 if any F1 drops by more than this (default: 0)",
    )
    ev.add_argument("-v", "--verbose", action="store_true", help="list every mistake")
    return parser


async def run_eval(args: argparse.Namespace) -> int:
    """Run the eval command. Returns the exit status."""
    if args.repeat < 1 or args.concurrency < 1:
        print("error: --repeat and --concurrency must be at least 1", file=sys.stderr)
        return 2
    try:
        examples = load_dataset(args.dataset)
        guards = [await load_guard(spec) for spec in args.guard]
    except (DatasetError, GuardSpecError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    names = [name for name, _ in guards]
    if len(set(names)) != len(names):
        print("error: two --guard values have the same name", file=sys.stderr)
        return 2

    labels: set[str] = {label for e in examples for label in e.labels}
    if args.labels:
        labels = {x.strip() for x in args.labels.split(",") if x.strip()}

    report: dict[str, Any] = {
        "version": REPORT_VERSION,
        "dataset": str(args.dataset),
        "examples": len(examples),
        "repeat": args.repeat,
        "guards": [],
    }
    for name, guard in guards:
        try:
            outcomes = await run_guard(
                guard, examples, repeat=args.repeat, concurrency=args.concurrency
            )
        finally:
            for classifier in (guard.classifier, guard.judge):
                if classifier is not None:
                    await classifier.cleanup()  # close API clients the classifier opened
        report["guards"].append(summarize(name, outcomes, labels))

    print(format_report(report, verbose=args.verbose))
    if args.json:
        text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        Path(args.json).write_bytes(text.encode("utf-8"))  # LF on every platform
        print(f"\nWrote {args.json}")
    if args.diff:
        try:
            baseline = json.loads(Path(args.diff).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            print(f"error: cannot read baseline {args.diff}: {e}", file=sys.stderr)
            return 2
        text, regressed = diff_reports(report, baseline, max_drop=args.max_drop)
        print("\n" + text)
        if regressed:
            return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the ``pipecat-guardrails`` command."""
    args = build_parser().parse_args(argv)
    # Print Devanagari and curly quotes even when output is piped or redirected, where
    # Windows would otherwise use a legacy code page.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None and not stream.isatty():
            reconfigure(encoding="utf-8", errors="replace")
    return asyncio.run(run_eval(args))


if __name__ == "__main__":
    sys.exit(main())
