"""Stylometric feature extraction.

Every feature has a stable name, a unit, a one-line description and, where the
measurement points at specific prose, the quoted spans it was read from. These
are measurements; none of them is a verdict.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field

from .segment import LineKind, Span, split_lines, split_paragraphs, split_sentences, split_words

__all__ = [
    "DISCOURSE_MARKERS",
    "FEATURES",
    "FEATURE_SPECS",
    "FIRST_PERSON_WORDS",
    "FUNCTION_WORDS",
    "Feature",
    "FeatureSet",
    "FeatureSpec",
    "extract_features",
]

SHORT_SENTENCE_WORDS = 8
LONG_SENTENCE_WORDS = 30

FUNCTION_WORDS: frozenset[str] = frozenset(
    {
        "a", "about", "above", "across", "after", "again", "against", "all",
        "along", "also", "although", "am", "among", "an", "and", "another",
        "any", "anyone", "anything", "are", "as", "at", "be", "because",
        "been", "before", "behind", "being", "below", "between", "beyond",
        "both", "but", "by", "can", "could", "did", "do", "does", "during",
        "each", "either", "even", "every", "everyone", "few", "for", "from",
        "had", "has", "have", "he", "her", "hers", "herself", "him", "himself",
        "his", "how", "however", "i", "if", "in", "indeed", "into", "is", "it",
        "its", "itself", "just", "least", "less", "many", "may", "me", "might",
        "mine", "more", "most", "much", "must", "my", "myself", "neither",
        "no", "nor", "not", "nothing", "of", "on", "only", "or", "other",
        "our", "ours", "ourselves", "out", "over", "quite", "rather",
        "several", "shall", "she", "should", "since", "so", "some", "someone",
        "something", "still", "such", "than", "that", "the", "their", "theirs",
        "them", "themselves", "then", "there", "these", "they", "this",
        "those", "though", "through", "thus", "to", "too", "toward",
        "towards", "under", "unless", "until", "up", "upon", "us", "very",
        "was", "we", "were", "what", "when", "where", "whether", "which",
        "while", "who", "whom", "whose", "why", "will", "with", "within",
        "without", "would", "yet", "you", "your", "yours", "yourself",
    }
)

FIRST_PERSON_WORDS: frozenset[str] = frozenset(
    {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"}
)

DISCOURSE_MARKERS: frozenset[str] = frozenset(
    {
        "accordingly", "additionally", "consequently", "crucially",
        "essentially", "furthermore", "however", "importantly", "moreover",
        "nevertheless", "nonetheless", "notably", "overall", "significantly",
        "subsequently", "therefore", "thus", "ultimately",
    }
)

_MARKER_RE = re.compile(r"^\s*(?:#{1,6}\s+|(?:[-*+\u2022]|\(?\d{1,3}[.)]|\(?[a-zA-Z][.)])\s+)")
_DASH_RE = re.compile(r"[\u2014\u2013]|(?<=\s)-{1,2}(?=\s)")
_ELLIPSIS_RE = re.compile(r"\u2026|\.{3,}")
_QUOTE_CHARS = "\"\u201c\u201d\u00ab\u00bb"
_PUNCTUATION_MARKS = (".", ",", ";", ":", "!", "?", "(", "\u2014", "\u2013", "\u2026", "\"", "'")


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """The definition of a feature: its name, its unit and what it measures."""

    name: str
    unit: str
    description: str


FEATURES: tuple[FeatureSpec, ...] = tuple(
    FeatureSpec(name, unit, description)
    for name, unit, description in (
        ("word_count", "words", "Alphabetic word tokens in the text."),
        ("sentence_count", "sentences", "Sentences found; a heading or a list item counts as one."),
        ("paragraph_count", "paragraphs", "Blocks separated by blank lines."),
        ("mean_sentence_length", "words per sentence", "Arithmetic mean of sentence lengths in words."),
        ("sentence_length_stdev", "words", "Population standard deviation of sentence lengths."),
        ("sentence_length_cv", "ratio", "Standard deviation of sentence length divided by its mean."),
        ("burstiness", "coefficient", "(stdev - mean) / (stdev + mean) over sentence lengths, in [-1, 1]; higher means more uneven pacing."),
        ("sentence_length_range", "words", "Longest sentence minus shortest sentence, in words."),
        ("short_sentence_ratio", "ratio", "Share of sentences shorter than 8 words."),
        ("long_sentence_ratio", "ratio", "Share of sentences longer than 30 words."),
        ("mean_paragraph_sentences", "sentences per paragraph", "Mean number of sentences per paragraph."),
        ("function_word_ratio", "ratio", "Share of word tokens that are closed-class function words; detail holds each function word's rate per 1000 words."),
        ("function_word_diversity", "ratio", "Distinct function words divided by function-word tokens."),
        ("first_person_ratio", "ratio", "Share of word tokens that are first-person pronouns."),
        ("discourse_marker_rate", "per 100 words", "Rate of single-word connectives such as however or therefore."),
        ("comma_rate", "per 100 words", "Commas per 100 words."),
        ("semicolon_rate", "per 100 words", "Semicolons per 100 words."),
        ("colon_rate", "per 100 words", "Colons per 100 words, with list and heading markers removed first."),
        ("dash_rate", "per 100 words", "Em dashes, en dashes and spaced hyphens per 100 words."),
        ("quote_rate", "per 100 words", "Quotation pairs per 100 words."),
        ("ellipsis_rate", "per 100 words", "Ellipses per 100 words."),
        ("question_ratio", "ratio", "Share of sentences containing a question mark."),
        ("exclamation_ratio", "ratio", "Share of sentences containing an exclamation mark."),
        ("punctuation_diversity", "marks", "Count of distinct punctuation marks the text uses at all."),
        ("list_item_ratio", "ratio", "Share of non-blank lines that are list items."),
        ("list_item_length_cv", "ratio", "Coefficient of variation of list item lengths; near zero means templated items."),
        ("heading_ratio", "ratio", "Share of non-blank lines that are headings."),
    )
)

FEATURE_SPECS: Mapping[str, FeatureSpec] = {spec.name: spec for spec in FEATURES}


@dataclass(frozen=True, slots=True)
class Feature:
    """A measured feature together with the evidence that produced it."""

    name: str
    value: float
    unit: str
    description: str
    evidence: tuple[str, ...] = ()
    detail: Mapping[str, float] = field(default_factory=dict)


class FeatureSet(Mapping[str, Feature]):
    """Features keyed by name, in extraction order."""

    def __init__(self, features: Sequence[Feature]) -> None:
        self._features = {feature.name: feature for feature in features}

    def __getitem__(self, name: str) -> Feature:
        return self._features[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._features)

    def __len__(self) -> int:
        return len(self._features)

    def value(self, name: str) -> float:
        """The numeric value of one feature."""
        return self._features[name].value

    def to_dict(self) -> dict[str, float]:
        """Plain name-to-number mapping, without units or evidence."""
        return {name: feature.value for name, feature in self._features.items()}


def extract_features(text: str) -> FeatureSet:
    """Measure every documented stylometric feature of ``text``."""
    lines = split_lines(text)
    sentences = split_sentences(text)
    paragraphs = split_paragraphs(text)
    words = split_words(text)

    word_count = len(words)
    sentence_count = len(sentences)
    lowered = [word.text.lower() for word in words]
    lengths = [len(split_words(sentence.text)) for sentence in sentences]
    body = "\n".join(_MARKER_RE.sub("", sentence.text) for sentence in sentences)

    collected: list[Feature] = []

    def add(
        name: str,
        value: float,
        evidence: Sequence[str] = (),
        detail: Mapping[str, float] | None = None,
    ) -> None:
        spec = FEATURE_SPECS[name]
        collected.append(
            Feature(spec.name, float(value), spec.unit, spec.description, tuple(evidence), dict(detail or {}))
        )

    def per_100_words(count: float) -> float:
        return 0.0 if word_count == 0 else 100.0 * count / word_count

    def word_share(count: int) -> float:
        return 0.0 if word_count == 0 else count / word_count

    def sentence_share(count: int) -> float:
        return 0.0 if sentence_count == 0 else count / sentence_count

    add("word_count", word_count)
    add("sentence_count", sentence_count)
    add("paragraph_count", len(paragraphs))

    mean_length = statistics.fmean(lengths) if lengths else 0.0
    stdev_length = statistics.pstdev(lengths) if len(lengths) > 1 else 0.0
    extremes = _extreme_sentences(sentences, lengths)
    add("mean_sentence_length", mean_length, extremes)
    add("sentence_length_stdev", stdev_length, extremes)
    add("sentence_length_cv", stdev_length / mean_length if mean_length else 0.0, extremes)
    spread = stdev_length + mean_length
    add("burstiness", (stdev_length - mean_length) / spread if spread else 0.0, extremes)
    add("sentence_length_range", max(lengths) - min(lengths) if lengths else 0, extremes)
    add(
        "short_sentence_ratio",
        sentence_share(sum(1 for n in lengths if n < SHORT_SENTENCE_WORDS)),
        _quotes_by_length(sentences, lengths, lambda n: n < SHORT_SENTENCE_WORDS),
    )
    add(
        "long_sentence_ratio",
        sentence_share(sum(1 for n in lengths if n > LONG_SENTENCE_WORDS)),
        _quotes_by_length(sentences, lengths, lambda n: n > LONG_SENTENCE_WORDS),
    )
    add("mean_paragraph_sentences", sentence_count / len(paragraphs) if paragraphs else 0.0)

    function_tokens = [word for word in lowered if word in FUNCTION_WORDS]
    counts = Counter(function_tokens)
    profile = {
        word: round(1000.0 * count / word_count, 4) for word, count in sorted(counts.items())
    } if word_count else {}
    add("function_word_ratio", word_share(len(function_tokens)), _densest_sentence(sentences), profile)
    add(
        "function_word_diversity",
        len(counts) / len(function_tokens) if function_tokens else 0.0,
    )
    add(
        "first_person_ratio",
        word_share(sum(1 for word in lowered if word in FIRST_PERSON_WORDS)),
        _quotes_with_vocabulary(sentences, FIRST_PERSON_WORDS),
    )
    add(
        "discourse_marker_rate",
        per_100_words(sum(1 for word in lowered if word in DISCOURSE_MARKERS)),
        _quotes_with_vocabulary(sentences, DISCOURSE_MARKERS),
    )

    add("comma_rate", per_100_words(body.count(",")), _quotes_containing(sentences, ","))
    add("semicolon_rate", per_100_words(body.count(";")), _quotes_containing(sentences, ";"))
    add("colon_rate", per_100_words(body.count(":")), _quotes_containing(sentences, ":"))
    add("dash_rate", per_100_words(len(_DASH_RE.findall(body))), _quotes_matching(sentences, _DASH_RE))
    add(
        "quote_rate",
        per_100_words(sum(body.count(char) for char in _QUOTE_CHARS) / 2.0),
        _quotes_containing(sentences, _QUOTE_CHARS[0]),
    )
    add(
        "ellipsis_rate",
        per_100_words(len(_ELLIPSIS_RE.findall(body))),
        _quotes_matching(sentences, _ELLIPSIS_RE),
    )
    add(
        "question_ratio",
        sentence_share(sum(1 for sentence in sentences if "?" in sentence.text)),
        _quotes_containing(sentences, "?"),
    )
    add(
        "exclamation_ratio",
        sentence_share(sum(1 for sentence in sentences if "!" in sentence.text)),
        _quotes_containing(sentences, "!"),
    )
    add("punctuation_diversity", sum(1 for mark in _PUNCTUATION_MARKS if mark in body))

    visible = [line for line in lines if line.kind not in (LineKind.BLANK, LineKind.RULE)]
    list_items = [line for line in visible if line.kind is LineKind.LIST_ITEM]
    headings = [line for line in visible if line.kind is LineKind.HEADING]

    def line_share(count: int) -> float:
        return 0.0 if not visible else count / len(visible)

    item_lengths = [len(split_words(_MARKER_RE.sub("", line.span.text))) for line in list_items]
    item_mean = statistics.fmean(item_lengths) if item_lengths else 0.0
    item_cv = statistics.pstdev(item_lengths) / item_mean if len(item_lengths) > 1 and item_mean else 0.0
    item_quotes = tuple(line.span.quote() for line in list_items[:3])
    add("list_item_ratio", line_share(len(list_items)), item_quotes)
    add("list_item_length_cv", item_cv, item_quotes)
    add("heading_ratio", line_share(len(headings)), tuple(line.span.quote() for line in headings[:3]))

    return FeatureSet(collected)


def _extreme_sentences(sentences: Sequence[Span], lengths: Sequence[int]) -> tuple[str, ...]:
    if not lengths:
        return ()
    shortest = min(range(len(lengths)), key=lengths.__getitem__)
    longest = max(range(len(lengths)), key=lengths.__getitem__)
    if shortest == longest:
        return (sentences[longest].quote(),)
    return (sentences[shortest].quote(), sentences[longest].quote())


def _quotes_by_length(
    sentences: Sequence[Span],
    lengths: Sequence[int],
    predicate: Callable[[int], bool],
    limit: int = 2,
) -> tuple[str, ...]:
    picked = [span.quote() for span, length in zip(sentences, lengths) if predicate(length)]
    return tuple(picked[:limit])


def _quotes_containing(sentences: Sequence[Span], needle: str, limit: int = 2) -> tuple[str, ...]:
    return tuple(span.quote() for span in sentences if needle in span.text)[:limit]


def _quotes_matching(sentences: Sequence[Span], pattern: re.Pattern[str], limit: int = 2) -> tuple[str, ...]:
    return tuple(span.quote() for span in sentences if pattern.search(span.text))[:limit]


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


def _densest_sentence(sentences: Sequence[Span], minimum_words: int = 5) -> tuple[str, ...]:
    best: Span | None = None
    best_share = -1.0
    for span in sentences:
        tokens = [word.text.lower() for word in split_words(span.text)]
        if len(tokens) < minimum_words:
            continue
        share = sum(1 for token in tokens if token in FUNCTION_WORDS) / len(tokens)
        if share > best_share:
            best, best_share = span, share
    return (best.quote(),) if best is not None else ()
