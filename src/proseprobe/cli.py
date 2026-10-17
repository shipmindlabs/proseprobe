"""Command line for scoring a file or standard input.

``probe file.txt`` prints the report as sentences and ``probe file.txt --json``
prints the same report as JSON. The command renders what the library returns and
computes nothing of its own, so the two cannot disagree.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .calibrate import DEFAULT_CONFIDENCE, Score, score
from .explain import EXCERPT_LIMIT, explain
from .report import report_score
from .segment import split_words

__all__ = ["main"]

STDIN_NAME = "-"


def main(argv: Sequence[str] | None = None) -> int:
    """Score every named file, or standard input, and print the reports."""
    options = _parser().parse_args(argv)

    documents, unread = _read(options.paths)
    for name, reason in unread:
        print(f"probe: {name}: {reason}", file=sys.stderr)

    reports: list[dict[str, object]] = []
    blocks: list[str] = []
    for name, path, text in documents:
        result = score(text, confidence=options.confidence)
        if options.json:
            reports.append(
                report_score(
                    result,
                    name=name,
                    path=path,
                    words=len(split_words(text)),
                    limit=options.limit,
                    excerpt_limit=options.excerpt_limit,
                )
            )
        else:
            blocks.append(
                _render(
                    name if len(documents) > 1 else "",
                    result,
                    options.limit,
                    options.excerpt_limit,
                )
            )

    if options.json:
        payload = reports[0] if len(reports) == 1 else reports
        print(json.dumps(payload, indent=options.indent or None, ensure_ascii=False))
    elif blocks:
        print("\n\n".join(blocks))

    return 1 if unread else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="probe",
        description="Explainable scoring of how machine-like a text reads.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        metavar="FILE",
        help=f"files to score; none, or {STDIN_NAME!r}, reads standard input",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the report as JSON instead of sentences",
    )
    parser.add_argument(
        "--confidence",
        type=_level,
        default=DEFAULT_CONFIDENCE,
        metavar="LEVEL",
        help=f"confidence level for the interval (default {DEFAULT_CONFIDENCE})",
    )
    parser.add_argument(
        "--limit",
        type=_positive,
        default=None,
        metavar="N",
        help="keep only the N largest feature shifts in the explanation",
    )
    parser.add_argument(
        "--excerpt-limit",
        type=_positive,
        default=EXCERPT_LIMIT,
        metavar="CHARS",
        help=f"how long a quoted excerpt may run (default {EXCERPT_LIMIT})",
    )
    parser.add_argument(
        "--indent",
        type=_indent,
        default=2,
        metavar="N",
        help="JSON indentation; 0 prints one line",
    )
    parser.add_argument("--version", action="version", version=f"proseprobe {__version__}")
    return parser


def _read(paths: Sequence[str]) -> tuple[list[tuple[str, str, str]], list[tuple[str, str]]]:
    if not paths:
        return [(STDIN_NAME, "", sys.stdin.read())], []
    documents: list[tuple[str, str, str]] = []
    unread: list[tuple[str, str]] = []
    for argument in paths:
        if argument == STDIN_NAME:
            documents.append((STDIN_NAME, "", sys.stdin.read()))
            continue
        path = Path(argument)
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            unread.append((argument, str(error)))
            continue
        documents.append((path.name, str(path), text))
    return documents, unread


def _render(name: str, result: Score, limit: int | None, excerpt_limit: int) -> str:
    blocks = [f"{name}:"] if name else []
    blocks.append(explain(result, limit=limit, excerpt_limit=excerpt_limit).text())
    blocks.append(result.guard.summary())
    return "\n\n".join(blocks)


def _level(value: str) -> float:
    try:
        level = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a number") from None
    if not 0.0 < level < 1.0:
        raise argparse.ArgumentTypeError("a confidence level lies between 0 and 1")
    return level


def _positive(value: str) -> int:
    number = _whole(value)
    if number < 1:
        raise argparse.ArgumentTypeError("a count of at least 1 is needed")
    return number


def _indent(value: str) -> int:
    number = _whole(value)
    if number < 0:
        raise argparse.ArgumentTypeError("an indentation cannot be negative")
    return number


def _whole(value: str) -> int:
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a whole number") from None
