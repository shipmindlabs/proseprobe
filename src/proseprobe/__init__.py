"""Explainable scoring of how machine-like a text reads."""

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
    "DISCOURSE_MARKERS",
    "FEATURES",
    "FEATURE_SPECS",
    "FIRST_PERSON_WORDS",
    "FUNCTION_WORDS",
    "Feature",
    "FeatureSet",
    "FeatureSpec",
    "Line",
    "LineKind",
    "Span",
    "__version__",
    "extract_features",
    "split_lines",
    "split_paragraphs",
    "split_sentences",
    "split_words",
]

__version__ = "0.0.1"
