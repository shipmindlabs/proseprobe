"""Calibrated 0..1 scale with a confidence band.

The scale places a text between the two bundled reference distributions: zero is
the human end, one is the machine end. Every point value arrives with an
interval, the bands that interval touches, the per-feature shifts that produced
it, and a note wherever the text was too short to measure something. Nothing
here returns a boolean: a number with a stated interval can be argued with, a
yes cannot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

from .features import FeatureSet, extract_features
from .reference import REFERENCES, Reference

__all__ = [
    "BANDS",
    "DEFAULT_CONFIDENCE",
    "Band",
    "Contribution",
    "Score",
    "score",
    "score_features",
]

DEFAULT_CONFIDENCE = 0.95
EVIDENCE_CEILING = 2.5
LOGIT_GAIN = 2.6
IRREDUCIBLE_SPREAD = 0.18
STABLE_WORDS = 250


@dataclass(frozen=True, slots=True)
class Band:
    """A named stretch of the scale, from ``low`` up to ``high``."""

    name: str
    low: float
    high: float
    description: str


BANDS: tuple[Band, ...] = (
    Band("very low", 0.0, 0.2, "Sits in the human reference range on the calibrated features."),
    Band("low", 0.2, 0.4, "Mostly inside the human reference range."),
    Band("middle", 0.4, 0.6, "Between the two reference ranges, or the features disagree."),
    Band("high", 0.6, 0.8, "Mostly inside the machine reference range."),
    Band("very high", 0.8, 1.0, "Sits in the machine reference range on the calibrated features."),
)


@dataclass(frozen=True, slots=True)
class Contribution:
    """One feature's share of the score, with the prose it was read from."""

    name: str
    value: float
    unit: str
    description: str
    human_mean: float
    machine_mean: float
    z_human: float
    z_machine: float
    logit_shift: float
    evidence: tuple[str, ...]

    def direction(self) -> str:
        """Which reference population this feature pulled the score toward."""
        if self.logit_shift > 0.0:
            return "toward the machine reference"
        if self.logit_shift < 0.0:
            return "toward the human reference"
        return "neutral"


@dataclass(frozen=True, slots=True)
class Score:
    """A position on the 0..1 scale, its interval, and how it was reached."""

    value: float
    low: float
    high: float
    confidence: float
    band: Band
    spanned_bands: tuple[Band, ...]
    contributions: tuple[Contribution, ...]
    notes: tuple[str, ...]

    @property
    def interval(self) -> tuple[float, float]:
        """The confidence interval as a pair."""
        return (self.low, self.high)

    @property
    def width(self) -> float:
        """How much of the scale the interval covers."""
        return self.high - self.low

    def moved_by(self, limit: int = 3) -> tuple[Contribution, ...]:
        """The features that shifted the score most, largest shift first."""
        return self.contributions[:limit]

    def summary(self) -> str:
        """One line: point value, interval, confidence level and band."""
        return (
            f"{self.value:.2f} of 1.00 ({self.low:.2f} to {self.high:.2f} "
            f"at {self.confidence:.0%} confidence, band '{self.band.name}')"
        )


def score(text: str, *, confidence: float = DEFAULT_CONFIDENCE) -> Score:
    """Place ``text`` on the calibrated scale with a confidence band."""
    return score_features(extract_features(text), confidence=confidence)


def score_features(features: FeatureSet, *, confidence: float = DEFAULT_CONFIDENCE) -> Score:
    """Place an already measured ``FeatureSet`` on the calibrated scale."""
    word_count = features.value("word_count")
    sentence_count = features.value("sentence_count")

    measured: list[tuple[Reference, float]] = []
    deferred: list[str] = []
    for reference in REFERENCES:
        if word_count < reference.min_words or sentence_count < reference.min_sentences:
            deferred.append(reference.feature)
            continue
        ratio = _log_likelihood_ratio(features.value(reference.feature), reference)
        measured.append((reference, ratio))

    notes: list[str] = []
    if deferred:
        notes.append("Not counted, the text is shorter than they need: " + ", ".join(deferred) + ".")
    if not measured:
        notes.append("No calibrated feature could be measured, so the scale reports its prior.")
        return Score(
            value=0.5,
            low=0.0,
            high=1.0,
            confidence=confidence,
            band=_band_at(0.5),
            spanned_bands=BANDS,
            contributions=(),
            notes=tuple(notes),
        )

    weight = sum(reference.weight for reference, _ in measured)
    mean_evidence = sum(reference.weight * ratio for reference, ratio in measured) / weight
    deviation = NormalDist().inv_cdf(0.5 + confidence / 2.0) * _spread(
        measured, mean_evidence, weight, word_count
    )

    value = _logistic(LOGIT_GAIN * mean_evidence)
    low = _logistic(LOGIT_GAIN * (mean_evidence - deviation))
    high = _logistic(LOGIT_GAIN * (mean_evidence + deviation))
    spanned = _bands_spanned(low, high)

    if word_count < STABLE_WORDS:
        notes.append(
            f"{int(word_count)} words is under the {STABLE_WORDS}-word floor where the "
            "reference spreads settle, so the interval is widened."
        )
    if len(spanned) > 2:
        notes.append(
            f"The interval covers {len(spanned)} bands; read the point value as indicative only."
        )

    return Score(
        value=value,
        low=low,
        high=high,
        confidence=confidence,
        band=_band_at(value),
        spanned_bands=spanned,
        contributions=_contributions(features, measured, weight),
        notes=tuple(notes),
    )


def _contributions(
    features: FeatureSet, measured: list[tuple[Reference, float]], weight: float
) -> tuple[Contribution, ...]:
    collected: list[Contribution] = []
    for reference, ratio in measured:
        feature = features[reference.feature]
        collected.append(
            Contribution(
                name=feature.name,
                value=feature.value,
                unit=feature.unit,
                description=feature.description,
                human_mean=reference.human_mean,
                machine_mean=reference.machine_mean,
                z_human=(feature.value - reference.human_mean) / reference.human_stdev,
                z_machine=(feature.value - reference.machine_mean) / reference.machine_stdev,
                logit_shift=LOGIT_GAIN * reference.weight * ratio / weight,
                evidence=feature.evidence,
            )
        )
    collected.sort(key=lambda item: -abs(item.logit_shift))
    return tuple(collected)


def _spread(
    measured: list[tuple[Reference, float]],
    mean_evidence: float,
    weight: float,
    word_count: float,
) -> float:
    if len(measured) > 1:
        variance = sum(
            reference.weight * (ratio - mean_evidence) ** 2 for reference, ratio in measured
        ) / weight
        standard_error = math.sqrt(variance / (len(measured) - 1))
    else:
        standard_error = EVIDENCE_CEILING
    widening = math.sqrt(STABLE_WORDS / word_count) if 0 < word_count < STABLE_WORDS else 1.0
    return math.hypot(standard_error, IRREDUCIBLE_SPREAD) * widening


def _log_likelihood_ratio(value: float, reference: Reference) -> float:
    ratio = _log_density(value, reference.machine_mean, reference.machine_stdev) - _log_density(
        value, reference.human_mean, reference.human_stdev
    )
    return max(-EVIDENCE_CEILING, min(EVIDENCE_CEILING, ratio))


def _log_density(value: float, mean: float, stdev: float) -> float:
    z = (value - mean) / stdev
    return -0.5 * z * z - math.log(stdev)


def _logistic(logit: float) -> float:
    if logit >= 0.0:
        return 1.0 / (1.0 + math.exp(-logit))
    exponential = math.exp(logit)
    return exponential / (1.0 + exponential)


def _band_at(value: float) -> Band:
    for band in BANDS:
        if value < band.high:
            return band
    return BANDS[-1]


def _bands_spanned(low: float, high: float) -> tuple[Band, ...]:
    return tuple(band for band in BANDS if band.low <= high and band.high >= low)
