"""Bundled reference distributions for the calibrated scale.

Each row is one indicator feature with the mean and standard deviation it takes
in two reference populations: human prose, and prose from instruction-tuned
language models at default settings. A row also carries the least amount of text
below which the measurement is too noisy to count, and a weight for how much
separation the feature earns.

A feature with no row here is still measured, it just does not move the score, so
an uncalibrated feature is a missing row rather than a hidden term.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

__all__ = ["REFERENCE_INDEX", "REFERENCE_NOTE", "REFERENCES", "Reference"]

REFERENCE_NOTE = (
    "Provisional pre-alpha summary statistics: coarse means and standard "
    "deviations per feature, not a fitted model. They are bundled in source form "
    "so any score can be reproduced and argued with; replace REFERENCES to "
    "calibrate against your own corpus."
)


@dataclass(frozen=True, slots=True)
class Reference:
    """One feature's distribution in each reference population."""

    feature: str
    human_mean: float
    human_stdev: float
    machine_mean: float
    machine_stdev: float
    weight: float
    min_words: int
    min_sentences: int
    note: str


REFERENCES: tuple[Reference, ...] = tuple(
    Reference(*row)
    for row in (
        ("burstiness", -0.30, 0.12, -0.52, 0.10, 1.4, 16, 3,
         "Machine prose paces evenly, so sentence lengths cluster."),
        ("sentence_length_cv", 0.55, 0.18, 0.32, 0.10, 1.2, 16, 3,
         "The same evenness, seen as spread over the mean."),
        ("mean_sentence_length", 17.5, 5.5, 20.5, 4.0, 0.8, 16, 3,
         "Machine sentences run slightly longer."),
        ("short_sentence_ratio", 0.18, 0.12, 0.06, 0.06, 0.7, 16, 3,
         "Very short sentences are a human habit models rarely take up."),
        ("long_sentence_ratio", 0.10, 0.08, 0.14, 0.08, 0.5, 16, 3,
         "Weak on its own; long sentences lean machine only marginally."),
        ("mean_paragraph_sentences", 3.4, 1.6, 3.0, 1.2, 0.3, 60, 4,
         "Model paragraphs are shorter and more uniform."),
        ("function_word_ratio", 0.48, 0.06, 0.45, 0.05, 0.6, 60, 3,
         "Models carry marginally more content words per token."),
        ("first_person_ratio", 0.022, 0.020, 0.005, 0.008, 0.9, 60, 3,
         "First-person pronouns are suppressed by default assistant style."),
        ("discourse_marker_rate", 0.45, 0.35, 1.20, 0.55, 1.1, 60, 3,
         "Connectives such as however or furthermore are over-used."),
        ("comma_rate", 5.4, 1.9, 6.6, 1.6, 0.7, 60, 3,
         "Model clauses are more heavily comma-separated."),
        ("semicolon_rate", 0.15, 0.25, 0.30, 0.30, 0.3, 60, 3,
         "Rare in both populations, slightly less so in models."),
        ("dash_rate", 0.35, 0.45, 0.85, 0.65, 0.6, 60, 3,
         "Em dashes are a pronounced model habit, and a human style too."),
        ("quote_rate", 0.55, 0.80, 0.20, 0.40, 0.3, 60, 3,
         "Quoted material appears more often in human writing."),
        ("exclamation_ratio", 0.04, 0.07, 0.01, 0.03, 0.4, 60, 3,
         "Exclamations are flattened out of default model style."),
        ("question_ratio", 0.07, 0.09, 0.03, 0.05, 0.3, 60, 3,
         "Rhetorical questions are more of a human move."),
        ("punctuation_diversity", 8.2, 1.9, 7.3, 1.7, 0.4, 60, 3,
         "Human texts reach for a slightly wider set of marks."),
    )
)

REFERENCE_INDEX: Mapping[str, Reference] = {row.feature: row for row in REFERENCES}
