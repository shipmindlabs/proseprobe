"""One report for one text, ready to be rendered as JSON.

A report carries the whole score: the point value with its interval and band,
every contribution with the prose it was read from, the guard that was applied,
the notes, and the explanation in sentences. Batch scoring puts the same
structure under each document and the command line prints it unchanged, so a
report has one shape wherever it came from.
"""

from __future__ import annotations

from .calibrate import DEFAULT_CONFIDENCE, Band, Contribution, Score, score
from .explain import EXCERPT_LIMIT, explain
from .guard import Caution, Guard, Reading
from .segment import split_words

__all__ = ["REPORT_VERSION", "report", "report_score"]

REPORT_VERSION = 1
DIGITS = 6


def report(
    text: str,
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    name: str = "",
    path: str = "",
    limit: int | None = None,
    excerpt_limit: int = EXCERPT_LIMIT,
) -> dict[str, object]:
    """Score ``text`` and return the whole report as a JSON-ready dictionary."""
    return report_score(
        score(text, confidence=confidence),
        name=name,
        path=path,
        words=len(split_words(text)),
        limit=limit,
        excerpt_limit=excerpt_limit,
    )


def report_score(
    result: Score,
    *,
    name: str = "",
    path: str = "",
    words: int = 0,
    limit: int | None = None,
    excerpt_limit: int = EXCERPT_LIMIT,
) -> dict[str, object]:
    """Render a ``Score`` already in hand in the same report structure."""
    return {
        "report_version": REPORT_VERSION,
        "name": name,
        "path": path,
        "words": words,
        "score": _score_dict(result),
        "explanation": explain(result, limit=limit, excerpt_limit=excerpt_limit).text(),
    }


def _score_dict(result: Score) -> dict[str, object]:
    return {
        "summary": result.summary(),
        "value": _round(result.value),
        "interval": {
            "low": _round(result.low),
            "high": _round(result.high),
            "width": _round(result.width),
            "confidence": result.confidence,
        },
        "band": _range_dict(result.band),
        "spanned_bands": [band.name for band in result.spanned_bands],
        "contributions": [
            _contribution_dict(contribution) for contribution in result.contributions
        ],
        "guard": _guard_dict(result.guard),
        "notes": list(result.notes),
    }


def _range_dict(named: Band | Caution) -> dict[str, object]:
    return {
        "name": named.name,
        "low": named.low,
        "high": named.high,
        "description": named.description,
    }


def _contribution_dict(contribution: Contribution) -> dict[str, object]:
    return {
        "name": contribution.name,
        "value": _round(contribution.value),
        "unit": contribution.unit,
        "description": contribution.description,
        "human_mean": contribution.human_mean,
        "machine_mean": contribution.machine_mean,
        "z_human": _round(contribution.z_human),
        "z_machine": _round(contribution.z_machine),
        "logit_shift": _round(contribution.logit_shift),
        "direction": contribution.direction(),
        "evidence": list(contribution.evidence),
    }


def _guard_dict(guard: Guard) -> dict[str, object]:
    return {
        "summary": guard.summary(),
        "strength": _round(guard.strength),
        "reliability": _round(guard.reliability),
        "caution": _range_dict(guard.caution),
        "words": guard.words,
        "sentences": guard.sentences,
        "readings": [_reading_dict(reading) for reading in guard.readings],
        "deferred": list(guard.deferred),
        "notes": list(guard.notes),
    }


def _reading_dict(reading: Reading) -> dict[str, object]:
    return {
        "name": reading.name,
        "sentence": reading.sentence(),
        "value": _round(reading.value),
        "unit": reading.unit,
        "note": reading.note,
        "threshold": reading.threshold,
        "saturation": reading.saturation,
        "weight": reading.weight,
        "strength": _round(reading.strength),
        "evidence": list(reading.evidence),
    }


def _round(value: float) -> float:
    return round(value, DIGITS)
