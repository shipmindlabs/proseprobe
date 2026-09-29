"""Explainable scoring of how machine-like a text reads."""

from .calibrate import (
    BANDS,
    DEFAULT_CONFIDENCE,
    Band,
    Contribution,
    Score,
    score,
    score_features,
)
from .features import (
    DISCOURSE_MARKERS,
    FEATURE_SPECS,
    FEATURES,
    FIRST_PERSON_WORDS,
    FUNCTION_WORDS,
    Feature,
    FeatureSet,
    FeatureSpec,
    extract_features,
)
from .reference import REFERENCE_INDEX, REFERENCE_NOTE, REFERENCES, Reference
from .segment import (
    Line,
    LineKind,
    Span,
    split_lines,
    split_paragraphs,
    split_sentences,
    split_words,
)

__all__ = [
    "BANDS",
    "DEFAULT_CONFIDENCE",
    "DISCOURSE_MARKERS",
    "FEATURES",
    "FEATURE_SPECS",
    "FIRST_PERSON_WORDS",
    "FUNCTION_WORDS",
    "REFERENCES",
    "REFERENCE_INDEX",
    "REFERENCE_NOTE",
    "Band",
    "Contribution",
    "Feature",
    "FeatureSet",
    "FeatureSpec",
    "Line",
    "LineKind",
    "Reference",
    "Score",
    "Span",
    "__version__",
    "extract_features",
    "score",
    "score_features",
    "split_lines",
    "split_paragraphs",
    "split_sentences",
    "split_words",
]

__version__ = "0.0.1"
