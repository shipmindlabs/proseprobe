"""Non-native English guard.

Detectors of machine-like prose systematically over-score competent non-native
English: measured pacing, textbook connectives and a thin stock of contractions
read as machine habits whoever wrote them. This module measures the patterns
that point at a non-native author and reports how far the score should be
trusted.

Every signal is a row in ``SIGNALS`` with the value where it starts to count,
the value where it counts fully, the weight it carries and the least text it
needs, and the guard reports the word and sentence count the rates were computed
over, so its own arithmetic can be checked. The scale uses the result to shrink
its evidence toward the prior and widen its interval, so a caution can only make
a score less certain, never larger. The guard names no language and makes no
claim about the author.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .segment import Span, split_sentences, split_words

__all__ = [
    "ARTICLES",
    "CAUTIONS",
    "FORMAL_CONNECTIVES",
    "SIGNALS",
    "TRANSFER_PATTERNS",
    "UNEXAMINED",
    "Caution",
    "Guard",
    "Reading",
    "Signal",
    "inspect_language",
]

ARTICLES: frozenset[str] = frozenset({"a", "an", "the"})

FORMAL_CONNECTIVES: frozenset[str] = frozenset(
    {
        "besides", "consequently", "firstly", "furthermore", "hence",
        "lastly", "moreover", "nowadays", "secondly", "therefore",
        "thirdly", "thus",
    }
)

TRANSFER_PATTERNS: tuple[str, ...] = (
    "according to me",
    "advices",
    "as a conclusion",
    "depend of",
    "depends of",
    "discuss about",
    "equipments",
    "evidences",
    "explain me",
    "feedbacks",
    "i am agree",
    "in my point of view",
    "in nowadays",
    "in the other hand",
    "informations",
    "knowledges",
    "more better",
    "since long time",
    "softwares",
    "the most biggest",
)

_CONTRACTION_RE = re.compile(
    r"\b(?:ai|ca|could|did|do|does|had|has|have|is|are|might|must|need|sha|should"
    r"|was|were|wo|would)n['\u2019]t\b"
    r"|\b(?:he|here|i|it|let|she|that|there|they|this|we|what|who|you)"
    r"['\u2019](?:d|ll|m|re|s|ve)\b",
    re.IGNORECASE,
)

_PATTERN_RE = re.compile(
    r"\b(?:"
    + "|".join(
        r"\s+".join(re.escape(word) for word in pattern.split())
        for pattern in TRANSFER_PATTERNS
    )
    + r")\b",
    re.IGNORECASE,
)

_MIN_QUOTE_WORDS = 6


@dataclass(frozen=True, slots=True)
class Signal:
    """One pattern the guard looks for, and how far it counts."""

    name: str
    unit: str
    description: str
    note: str
    threshold: float
    saturation: float
    weight: float
    min_words: int
    min_sentences: int

    def strength_at(self, value: float) -> float:
        """How far ``value`` has travelled from ``threshold`` to ``saturation``."""
        reach = self.saturation - self.threshold
        return max(0.0, min(1.0, (value - self.threshold) / reach))


SIGNALS: tuple[Signal, ...] = tuple(
    Signal(*row)
    for row in (
        ("article_rate", "per 100 words", "Rate of a, an and the.",
         "Many first languages have no article system, so articles thin out.",
         5.5, 2.0, 1.3, 80, 3),
        ("contraction_rate", "per 100 words",
         "Rate of verb contractions such as isn't or we'll.",
         "Contractions are learned late and avoided in careful writing.",
         0.3, 0.0, 0.4, 120, 4),
        ("repeated_opener_ratio", "ratio",
         "Share of sentence openers that repeat another sentence's opener.",
         "A small stock of sentence openers is reused across the text.",
         0.35, 0.7, 0.8, 100, 6),
        ("formal_connective_rate", "per 100 words",
         "Rate of connectives such as moreover, hence or nowadays.",
         "Textbook connectives are taught as the way to join paragraphs, and the "
         "same words are what machine prose over-uses.",
         1.2, 3.0, 0.9, 100, 4),
        ("transfer_pattern_rate", "per 1000 words",
         "Rate of wordings recorded in learner-English corpora.",
         "Wordings carried over from another language, such as informations or "
         "in the other hand.",
         0.6, 4.0, 1.4, 60, 2),
        ("foreign_letter_rate", "per 100 words",
         "Rate of word tokens carrying letters outside ASCII.",
         "Letters from another orthography reach the page unconverted.",
         0.4, 2.5, 0.6, 60, 2),
    )
)

TOTAL_WEIGHT = sum(signal.weight for signal in SIGNALS)


@dataclass(frozen=True, slots=True)
class Caution:
    """A named stretch of guard strength, from ``low`` up to ``high``."""

    name: str
    low: float
    high: float
    description: str


CAUTIONS: tuple[Caution, ...] = (
    Caution("none", 0.0, 0.15, "Nothing stood out; the score stands as measured."),
    Caution("slight", 0.15, 0.35, "One pattern stood out; read the score as a little soft."),
    Caution("substantial", 0.35, 0.65,
            "Several patterns stood out; the score is held back and its interval widened."),
    Caution("severe", 0.65, 1.0,
            "The patterns dominate; treat the score as carrying little information."),
)


@dataclass(frozen=True, slots=True)
class Reading:
    """One signal as measured, with the prose it was read from."""

    name: str
    value: float
    unit: str
    description: str
    note: str
    threshold: float
    saturation: float
    weight: float
    strength: float
    evidence: tuple[str, ...]

    def sentence(self) -> str:
        """The measurement and how far it counts, as one line."""
        if not self.strength:
            return (
                f"{self.name} is {_number(self.value)} {self.unit}, inside the range where "
                f"it does not count against the score."
            )
        return (
            f"{self.name} is {_number(self.value)} {self.unit}, {self.strength:.0%} of the "
            f"way from {_number(self.threshold)} where it starts to count to "
            f"{_number(self.saturation)} where it counts fully: {self.note}"
        )


@dataclass(frozen=True, slots=True)
class Guard:
    """How far the score should be trusted, which patterns said so, over how much text."""

    strength: float
    caution: Caution
    words: int
    sentences: int
    readings: tuple[Reading, ...]
    deferred: tuple[str, ...]
    notes: tuple[str, ...]

    @property
    def reliability(self) -> float:
        """The share of the measured evidence the scale is allowed to keep."""
        return 1.0 - self.strength

    def fired(self) -> tuple[Reading, ...]:
        """The signals that counted, largest share of the strength first."""
        return tuple(reading for reading in self.readings if reading.strength > 0.0)

    def summary(self) -> str:
        """One line: the caution, the reliability left, and what raised it."""
        if not self.words:
            return "Non-native English guard: did not run, it was given no prose to read."
        fired = self.fired()
        if not fired:
            if not self.readings:
                return (
                    f"Non-native English guard: {self.words} words is too little text to "
                    "weigh any signal."
                )
            return "Non-native English guard: nothing stood out, reliability 100%."
        return (
            f"Non-native English guard: {self.caution.name} caution at "
            f"{self.reliability:.0%} reliability ("
            + ", ".join(reading.name for reading in fired)
            + ")."
        )


UNEXAMINED = Guard(
    strength=0.0,
    caution=CAUTIONS[0],
    words=0,
    sentences=0,
    readings=(),
    deferred=tuple(signal.name for signal in SIGNALS),
    notes=(
        "The non-native English guard did not run: scoring started from measured "
        "features, which no longer carry the text the guard reads.",
    ),
)


def inspect_language(text: str) -> Guard:
    """Measure the non-native English signals in ``text`` and weigh them."""
    words = split_words(text)
    sentences = split_sentences(text)
    word_count = len(words)
    sentence_count = len(sentences)
    measurements = _measure(text, words, sentences)

    readings: list[Reading] = []
    deferred: list[str] = []
    for signal in SIGNALS:
        if word_count < signal.min_words or sentence_count < signal.min_sentences:
            deferred.append(signal.name)
            continue
        value, evidence = measurements[signal.name]
        strength = signal.strength_at(value)
        readings.append(
            Reading(
                name=signal.name,
                value=value,
                unit=signal.unit,
                description=signal.description,
                note=signal.note,
                threshold=signal.threshold,
                saturation=signal.saturation,
                weight=signal.weight,
                strength=strength,
                evidence=evidence if strength > 0.0 else (),
            )
        )
    readings.sort(key=lambda reading: -reading.weight * reading.strength)

    strength = sum(reading.weight * reading.strength for reading in readings) / TOTAL_WEIGHT
    caution = _caution_at(strength)
    fired = [reading for reading in readings if reading.strength > 0.0]

    notes: list[str] = []
    if fired:
        notes.append(
            "Non-native English patterns are present ("
            + ", ".join(reading.name for reading in fired)
            + f"): {caution.description}"
        )
        notes.append(
            "Machine-prose detectors over-score non-native English, so this is a "
            "statement about how far to trust the score, not about the author."
        )
    elif readings:
        notes.append(
            "No non-native English pattern stood out, so the score is reported as measured."
        )
    if deferred:
        notes.append(
            "The non-native English guard could not weigh these, the text is shorter than "
            f"they need ({word_count} words, {sentence_count} sentences): "
            + ", ".join(deferred)
            + "."
        )

    return Guard(
        strength=strength,
        caution=caution,
        words=word_count,
        sentences=sentence_count,
        readings=tuple(readings),
        deferred=tuple(deferred),
        notes=tuple(notes),
    )


def _measure(
    text: str, words: Sequence[Span], sentences: Sequence[Span]
) -> Mapping[str, tuple[float, tuple[str, ...]]]:
    word_count = len(words)
    lowered = [word.text.lower() for word in words]

    def rate(count: float, per: float) -> float:
        return 0.0 if not word_count else per * count / word_count

    articles = sum(1 for word in lowered if word in ARTICLES)
    connectives = sum(1 for word in lowered if word in FORMAL_CONNECTIVES)
    contractions = len(_CONTRACTION_RE.findall(text))
    transfers = len(_PATTERN_RE.findall(text))
    foreign = sum(1 for word in lowered if not word.isascii())

    return {
        "article_rate": (rate(articles, 100.0), _articleless_quotes(sentences)),
        "contraction_rate": (rate(contractions, 100.0), ()),
        "repeated_opener_ratio": _openers(sentences),
        "formal_connective_rate": (
            rate(connectives, 100.0),
            _quotes_with_vocabulary(sentences, FORMAL_CONNECTIVES),
        ),
        "transfer_pattern_rate": (
            rate(transfers, 1000.0),
            _quotes_matching(sentences, _PATTERN_RE),
        ),
        "foreign_letter_rate": (rate(foreign, 100.0), _foreign_quotes(sentences)),
    }


def _articleless_quotes(sentences: Sequence[Span], limit: int = 2) -> tuple[str, ...]:
    picked: list[str] = []
    for span in sentences:
        tokens = [word.text.lower() for word in split_words(span.text)]
        if len(tokens) >= _MIN_QUOTE_WORDS and not any(token in ARTICLES for token in tokens):
            picked.append(span.quote())
        if len(picked) == limit:
            break
    return tuple(picked)


def _openers(sentences: Sequence[Span], limit: int = 2) -> tuple[float, tuple[str, ...]]:
    openers: list[tuple[str, Span]] = []
    for span in sentences:
        tokens = split_words(span.text)
        if tokens:
            openers.append((tokens[0].text.lower(), span))
    if not openers:
        return (0.0, ())
    counts = Counter(opener for opener, _ in openers)
    ratio = 1.0 - len(counts) / len(openers)
    opener, repeats = counts.most_common(1)[0]
    if repeats < 2:
        return (ratio, ())
    quotes = tuple(span.quote() for name, span in openers if name == opener)
    return (ratio, quotes[:limit])


def _quotes_with_vocabulary(
    sentences: Sequence[Span], vocabulary: frozenset[str], limit: int = 2
) -> tuple[str, ...]:
    picked: list[str] = []
    for span in sentences:
        if any(word.text.lower() in vocabulary for word in split_words(span.text)):
            picked.append(span.quote())
        if len(picked) == limit:
            break
    return tuple(picked)


def _quotes_matching(
    sentences: Sequence[Span], pattern: re.Pattern[str], limit: int = 2
) -> tuple[str, ...]:
    return tuple(span.quote() for span in sentences if pattern.search(span.text))[:limit]


def _foreign_quotes(sentences: Sequence[Span], limit: int = 2) -> tuple[str, ...]:
    picked: list[str] = []
    for span in sentences:
        if any(not word.text.isascii() for word in split_words(span.text)):
            picked.append(span.quote())
        if len(picked) == limit:
            break
    return tuple(picked)


def _caution_at(strength: float) -> Caution:
    for caution in CAUTIONS:
        if strength < caution.high:
            return caution
    return CAUTIONS[-1]


def _number(value: float) -> str:
    return f"{value:.3g}"
