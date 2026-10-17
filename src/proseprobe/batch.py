"""Batch scoring with JSON reports.

A batch is a directory or a list of documents. Every document keeps its own
report in the structure ``report()`` returns: the point value with its interval
and band, every contribution with the prose it was read from, the guard that was
applied, the notes, and the explanation in sentences.

The batch gets one summary. It counts how the values fall across the bands and
the cautions, names the features that moved them most, and lists the files it
could not read. It ranks nothing, labels no document and holds no threshold, so
a batch of numbers cannot become a batch of verdicts.
"""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from .calibrate import BANDS, DEFAULT_CONFIDENCE, Score, score
from .guard import CAUTIONS
from .report import DIGITS, REPORT_VERSION, report_score
from .segment import split_words

__all__ = [
    "DEFAULT_PATTERNS",
    "SUMMARY_FILENAME",
    "BatchReport",
    "DocumentReport",
    "Unread",
    "score_directory",
    "score_documents",
    "score_paths",
    "write_reports",
]

SUMMARY_FILENAME = "summary.json"
DEFAULT_PATTERNS: tuple[str, ...] = ("*.txt", "*.md", "*.markdown")
WIDE_INTERVAL = 0.5

_UNSAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True, slots=True)
class Unread:
    """A file the batch could not read, and the reason it gave."""

    name: str
    path: str
    reason: str

    def to_dict(self) -> dict[str, object]:
        """The record as a JSON-ready dictionary."""
        return {"name": self.name, "path": self.path, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class DocumentReport:
    """One document's score, ready to be written as its own report."""

    name: str
    path: str
    words: int
    score: Score

    def to_dict(self) -> dict[str, object]:
        """The whole score as a JSON-ready dictionary, explanation included."""
        return report_score(self.score, name=self.name, path=self.path, words=self.words)

    def to_json(self, *, indent: int = 2) -> str:
        """The report rendered as JSON text."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


@dataclass(frozen=True, slots=True)
class BatchReport:
    """Every document that was scored, and every file that could not be read."""

    documents: tuple[DocumentReport, ...]
    unread: tuple[Unread, ...]
    confidence: float

    def filenames(self) -> tuple[str, ...]:
        """One report filename per document, unique and in document order."""
        taken = {SUMMARY_FILENAME}
        names: list[str] = []
        for document in self.documents:
            slug = _UNSAFE_RE.sub("-", document.name).strip("-.") or "document"
            candidate = f"{slug}.json"
            attempt = 1
            while candidate in taken:
                attempt += 1
                candidate = f"{slug}-{attempt}.json"
            taken.add(candidate)
            names.append(candidate)
        return tuple(names)

    def headline(self) -> str:
        """One line: how many documents were scored, and where their values fell."""
        if not self.documents:
            line = "No document was scored."
        else:
            values = [document.score.value for document in self.documents]
            line = (
                f"{_count(len(self.documents), 'document')} scored at "
                f"{self.confidence:.0%} confidence: mean {statistics.fmean(values):.2f}, "
                f"median {statistics.median(values):.2f}, from {min(values):.2f} to "
                f"{max(values):.2f} on the 0..1 scale."
            )
        if self.unread:
            line += f" {_count(len(self.unread), 'file')} could not be read."
        return line

    def notes(self) -> tuple[str, ...]:
        """What the summary can and cannot be read as, in plain sentences."""
        notes = [
            "This summary counts how the scores fall. It ranks nothing, labels no "
            "document and holds no threshold."
        ]
        total = len(self.documents)
        cautioned = sum(1 for item in self.documents if item.score.guard.strength > 0.0)
        if cautioned:
            notes.append(
                f"{_count(cautioned, 'document')} of {total} raised a non-native English "
                "caution, so those scores are held toward the middle of the scale and "
                "their intervals widened."
            )
        wide = sum(1 for item in self.documents if item.score.width > WIDE_INTERVAL)
        if wide:
            notes.append(
                f"{_count(wide, 'document')} of {total} carry an interval wider than half "
                "the scale; read those point values as indicative only."
            )
        if self.unread:
            notes.append(
                "Not scored, these files could not be read as UTF-8 text: "
                + ", ".join(record.name for record in self.unread)
                + "."
            )
        return tuple(notes)

    def summary(self) -> dict[str, object]:
        """The batch as a JSON-ready dictionary."""
        values = [document.score.value for document in self.documents]
        widths = [document.score.width for document in self.documents]
        bands = Counter(document.score.band.name for document in self.documents)
        cautions = Counter(
            document.score.guard.caution.name for document in self.documents
        )
        return {
            "report_version": REPORT_VERSION,
            "headline": self.headline(),
            "confidence": self.confidence,
            "documents": len(self.documents),
            "words": sum(document.words for document in self.documents),
            "scale": _distribution(values, widths),
            "bands": {band.name: bands[band.name] for band in BANDS},
            "cautions": {caution.name: cautions[caution.name] for caution in CAUTIONS},
            "features_moved": self._movers(),
            "reports": [
                {
                    "name": document.name,
                    "file": filename,
                    "value": _round(document.score.value),
                    "interval": [_round(document.score.low), _round(document.score.high)],
                    "band": document.score.band.name,
                    "caution": document.score.guard.caution.name,
                }
                for document, filename in zip(self.documents, self.filenames())
            ],
            "unread": [record.to_dict() for record in self.unread],
            "notes": list(self.notes()),
        }

    def to_json(self, *, indent: int = 2) -> str:
        """The summary rendered as JSON text."""
        return json.dumps(self.summary(), indent=indent, ensure_ascii=False)

    def _movers(self) -> list[dict[str, object]]:
        shifts: dict[str, list[float]] = {}
        for document in self.documents:
            for contribution in document.score.contributions:
                shifts.setdefault(contribution.name, []).append(contribution.logit_shift)
        ordered = sorted(
            shifts.items(),
            key=lambda item: -statistics.fmean([abs(shift) for shift in item[1]]),
        )
        return [
            {
                "name": name,
                "documents": len(measured),
                "mean_logit_shift": _round(statistics.fmean(measured)),
                "mean_absolute_shift": _round(
                    statistics.fmean([abs(shift) for shift in measured])
                ),
            }
            for name, measured in ordered
        ]


def score_documents(
    documents: Mapping[str, str] | Iterable[tuple[str, str]],
    *,
    confidence: float = DEFAULT_CONFIDENCE,
) -> BatchReport:
    """Score a name-to-text mapping, or an iterable of ``(name, text)`` pairs."""
    items = documents.items() if isinstance(documents, Mapping) else documents
    scored = tuple(_report(name, "", text, confidence) for name, text in items)
    return BatchReport(documents=scored, unread=(), confidence=confidence)


def score_paths(
    paths: Iterable[str | Path], *, confidence: float = DEFAULT_CONFIDENCE
) -> BatchReport:
    """Score the files at ``paths``, each named by its filename."""
    return _read_and_score([Path(path) for path in paths], None, confidence)


def score_directory(
    directory: str | Path,
    *,
    patterns: Iterable[str] = DEFAULT_PATTERNS,
    confidence: float = DEFAULT_CONFIDENCE,
) -> BatchReport:
    """Score every file under ``directory`` matching ``patterns``, named by its path."""
    base = Path(directory)
    found: set[Path] = set()
    for pattern in patterns:
        found.update(path for path in base.rglob(pattern) if path.is_file())
    return _read_and_score(sorted(found), base, confidence)


def write_reports(
    batch: BatchReport, directory: str | Path, *, indent: int = 2
) -> tuple[Path, ...]:
    """Write one JSON report per document plus the summary, and return the paths."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for document, filename in zip(batch.documents, batch.filenames()):
        path = target / filename
        path.write_text(document.to_json(indent=indent) + "\n", encoding="utf-8")
        written.append(path)
    summary = target / SUMMARY_FILENAME
    summary.write_text(batch.to_json(indent=indent) + "\n", encoding="utf-8")
    written.append(summary)
    return tuple(written)


def _read_and_score(paths: list[Path], base: Path | None, confidence: float) -> BatchReport:
    documents: list[DocumentReport] = []
    unread: list[Unread] = []
    for path in paths:
        name = path.relative_to(base).as_posix() if base is not None else path.name
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            unread.append(Unread(name=name, path=str(path), reason=str(error)))
            continue
        documents.append(_report(name, str(path), text, confidence))
    return BatchReport(tuple(documents), tuple(unread), confidence)


def _report(name: str, path: str, text: str, confidence: float) -> DocumentReport:
    return DocumentReport(
        name=name,
        path=path,
        words=len(split_words(text)),
        score=score(text, confidence=confidence),
    )


def _distribution(values: list[float], widths: list[float]) -> dict[str, object]:
    if not values:
        return {
            "mean": None,
            "median": None,
            "lowest": None,
            "highest": None,
            "mean_interval_width": None,
        }
    return {
        "mean": _round(statistics.fmean(values)),
        "median": _round(statistics.median(values)),
        "lowest": _round(min(values)),
        "highest": _round(max(values)),
        "mean_interval_width": _round(statistics.fmean(widths)),
    }


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}" if number == 1 else f"{number} {noun}s"


def _round(value: float) -> float:
    return round(value, DIGITS)
