"""Plain-language explanation of a score.

An explanation restates what the scale already reported: which calibrated
features moved the score, which reference each one pulled toward, how large its
shift was, and one short quoted excerpt from the prose the measurement was read
from. It introduces no arithmetic of its own, so it cannot disagree with the
score it describes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .calibrate import Contribution, Score

__all__ = ["MAGNITUDES", "Explanation", "Magnitude", "Reason", "explain"]

EXCERPT_LIMIT = 120
NO_EXCERPT = "no span to quote for it; the measurement is over the whole text"


@dataclass(frozen=True, slots=True)
class Magnitude:
    """A named size for a shift, from ``floor`` up to ``ceiling`` in log-odds."""

    name: str
    floor: float
    ceiling: float


MAGNITUDES: tuple[Magnitude, ...] = (
    Magnitude("negligible", 0.0, 0.05),
    Magnitude("slight", 0.05, 0.15),
    Magnitude("moderate", 0.15, 0.40),
    Magnitude("strong", 0.40, math.inf),
)


@dataclass(frozen=True, slots=True)
class Reason:
    """One feature's effect on the score, in words, with the prose behind it."""

    name: str
    value: float
    unit: str
    description: str
    human_mean: float
    machine_mean: float
    logit_shift: float
    share: float
    magnitude: Magnitude
    direction: str
    excerpt: str

    def sentence(self) -> str:
        """The measurement, the size of its shift and its direction, as one line."""
        movement = self.direction if self.logit_shift else "pulling toward neither reference"
        return (
            f"{self.name} is {_number(self.value)} {self.unit}: a {self.magnitude.name} "
            f"shift of {self.logit_shift:+.3f} in log-odds, {movement}, "
            f"{self.share:.0%} of the movement; the references sit at "
            f"{_number(self.human_mean)} human and {_number(self.machine_mean)} machine."
        )

    def quotation(self) -> str:
        """The excerpt in quotation marks, or a statement that there is none."""
        if not self.excerpt:
            return f"({NO_EXCERPT})"
        return f"\u201c{self.excerpt}\u201d"


@dataclass(frozen=True, slots=True)
class Explanation:
    """A score retold: a headline, one block per feature, and the score's notes."""

    headline: str
    reasons: tuple[Reason, ...]
    notes: tuple[str, ...]
    omitted: int

    def text(self) -> str:
        """The whole explanation as plain text."""
        blocks = [self.headline]
        blocks.extend(f"{reason.sentence()}\n    {reason.quotation()}" for reason in self.reasons)
        if self.omitted:
            blocks.append(f"{self.omitted} smaller shifts are measured but not shown.")
        blocks.extend(self.notes)
        return "\n\n".join(blocks)

    def __str__(self) -> str:
        return self.text()


def explain(
    result: Score, *, limit: int | None = None, excerpt_limit: int = EXCERPT_LIMIT
) -> Explanation:
    """Restate ``result`` as sentences, with one quoted excerpt per feature."""
    movement = sum(abs(contribution.logit_shift) for contribution in result.contributions)
    kept = result.contributions if limit is None else result.contributions[:limit]
    return Explanation(
        headline=result.summary(),
        reasons=tuple(_reason(contribution, movement, excerpt_limit) for contribution in kept),
        notes=result.notes,
        omitted=len(result.contributions) - len(kept),
    )


def _reason(contribution: Contribution, movement: float, excerpt_limit: int) -> Reason:
    return Reason(
        name=contribution.name,
        value=contribution.value,
        unit=contribution.unit,
        description=contribution.description,
        human_mean=contribution.human_mean,
        machine_mean=contribution.machine_mean,
        logit_shift=contribution.logit_shift,
        share=abs(contribution.logit_shift) / movement if movement else 0.0,
        magnitude=_magnitude(contribution.logit_shift),
        direction=contribution.direction(),
        excerpt=_excerpt(contribution.evidence, excerpt_limit),
    )


def _magnitude(shift: float) -> Magnitude:
    size = abs(shift)
    for magnitude in MAGNITUDES:
        if size < magnitude.ceiling:
            return magnitude
    return MAGNITUDES[-1]


def _excerpt(evidence: Sequence[str], limit: int) -> str:
    for quote in evidence:
        collapsed = " ".join(quote.split())
        if collapsed:
            return _shorten(collapsed, limit)
    return ""


def _shorten(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[: limit - 1]
    boundary = cut.rfind(" ")
    if boundary > limit // 2:
        cut = cut[:boundary]
    return cut.rstrip(" ,;:") + "\u2026"


def _number(value: float) -> str:
    return f"{value:.3g}"
