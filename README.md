# proseprobe

Explainable scoring of how machine-like a text reads.

proseprobe reports a calibrated scale with confidence bands, names the
stylometric features that moved the score and quotes the spans they came from,
and applies a guard for non-native English, where detectors systematically fail.
It never returns a binary verdict: "written by a machine" is not a decision this
library is willing to make on your behalf.

## Status

Pre-alpha. Feature extraction, the calibrated scale with its confidence bands,
the quoted explanation and the non-native English guard are in place; the bundled
reference distributions and the guard's thresholds are provisional estimates
carried in source form so they can be read and replaced.

## Install

```bash
pip install proseprobe
```

The core runs on the standard library. NumPy is optional and only speeds up the
numeric paths:

```bash
pip install "proseprobe[numpy]"
```

Python 3.11 or newer is required.

## Usage

```python
from proseprobe import score

result = score(text)

print(result.summary())
print(result.value, result.interval, result.band.name)

for contribution in result.moved_by():
    print(contribution.name, round(contribution.logit_shift, 3), contribution.direction())
    for quote in contribution.evidence:
        print("  ", quote)

for note in result.notes:
    print(note)
```

The guard travels with the score:

```python
print(result.guard.summary())
print(result.guard.caution.name, round(result.guard.reliability, 2))

for reading in result.guard.fired():
    print(reading.sentence())
    for quote in reading.evidence:
        print("  ", quote)
```

The same result told as sentences, with one short quote under each feature:

```python
from proseprobe import explain, score

explanation = explain(score(text))

print(explanation)

for reason in explanation.reasons:
    print(reason.name, reason.magnitude.name, reason.direction)
    print("  ", reason.excerpt)
```

The measurements are available on their own:

```python
from proseprobe import extract_features

features = extract_features(text)

print(features.value("burstiness"))

moved = features["burstiness"]
print(moved.description, moved.unit)
for quote in moved.evidence:
    print("  ", quote)
```

Every feature carries its `name`, `value`, `unit`, a one-line `description` and,
where the measurement points at specific prose, the `evidence` it was read from.
`features.to_dict()` gives a plain name-to-number mapping, and `score_features()`
scores a `FeatureSet` you already have.

## The scale

The scale runs from 0 to 1: 0 is the human end of the bundled reference
distributions, 1 is the machine end. There is no boolean in the result and no
threshold that turns the number into a verdict.

A `Score` carries the point `value`; `low`, `high`, `interval` and `width` at the
requested `confidence` (0.95 by default, so `score(text, confidence=0.99)` gives
a wider band); the `band` the point falls in and every band the interval touches
as `spanned_bands`; one `Contribution` per calibrated feature, largest shift
first, with its distance from both reference means in standard deviations, a
signed `logit_shift` and the quoted spans behind it; the `guard` that was applied;
and `notes` in plain sentences about what limited the result.

| band | range |
| --- | --- |
| very low | 0.0 to 0.2 |
| low | 0.2 to 0.4 |
| middle | 0.4 to 0.6 |
| high | 0.6 to 0.8 |
| very high | 0.8 to 1.0 |

The shifts sum to the log-odds behind `value`, so the arithmetic is checkable.
An interval spanning three or more bands means the point value carries little
information, and it says so in `notes`.

Short text is handled by saying so rather than by guessing. Each reference row
declares the least text it needs, rows under that minimum are skipped and named
in `notes`, the interval is widened below 250 words, and a text where nothing can
be measured returns the prior of 0.5 across the full 0..1 interval.

## The non-native English guard

Detectors of machine-like prose systematically over-score competent non-native
English: measured pacing, textbook connectives and a thin stock of contractions
read as machine habits whoever wrote them. The guard measures the patterns that
point at a non-native author and reports how far the score should be trusted. It
names no language and makes no claim about the author.

`inspect_language(text)` returns a `Guard` on its own, and `score(text)` runs it
and carries it on `result.guard`. Each signal becomes a `Reading` with its
measured value and unit, the value where it starts to count, the value where it
counts fully, its `strength` between those two and the quoted spans it was read
from. `guard.strength` is the weighted mean of those strengths over all signals,
`guard.reliability` is what is left of 1, `guard.caution` names the stretch the
strength falls in, and `guard.words` and `guard.sentences` give the denominator
the rates were computed over.

| signal | unit | starts to count | counts fully | weight |
| --- | --- | --- | --- | --- |
| article_rate | per 100 words | 5.5 | 2.0 | 1.3 |
| contraction_rate | per 100 words | 0.3 | 0.0 | 0.4 |
| repeated_opener_ratio | ratio | 0.35 | 0.7 | 0.8 |
| formal_connective_rate | per 100 words | 1.2 | 3.0 | 0.9 |
| transfer_pattern_rate | per 1000 words | 0.6 | 4.0 | 1.4 |
| foreign_letter_rate | per 100 words | 0.4 | 2.5 | 0.6 |

| caution | strength |
| --- | --- |
| none | 0.00 to 0.15 |
| slight | 0.15 to 0.35 |
| substantial | 0.35 to 0.65 |
| severe | 0.65 to 1.00 |

A caution can only make a score less certain, never larger. The scale keeps
`1 - strength` of the measured evidence, which pulls the point value toward the
prior of 0.5, and widens the interval in proportion to the strength. A text whose
patterns dominate lands near the middle of the scale with a wide band and a note
saying why, instead of high on it.

Signals the text is too short for are deferred, named in `guard.deferred` and
reported in the notes together with the word and sentence count that fell short.
`score_features()` scores a `FeatureSet`, which no longer carries the prose the
guard reads, so it reports `UNEXAMINED`: no caution, every signal deferred, and a
note that the guard did not run.

The authoritative definitions live in `proseprobe.SIGNALS`, with the vocabularies
they read in `proseprobe.ARTICLES`, `proseprobe.FORMAL_CONNECTIVES` and
`proseprobe.TRANSFER_PATTERNS`. A pattern that is not in one of those tables
cannot raise a caution, so a missing heuristic is a missing row rather than a
hidden rule.

## The explanation

`explain(result)` restates a `Score` in sentences. Each feature that moved the
score becomes a `Reason` with its measured value and unit, both reference means,
the signed `logit_shift` the scale already reported, that shift's `share` of the
total movement, a named `magnitude`, the reference it pulled `direction`, and one
short quoted `excerpt` from the prose the measurement was read from. A feature
that points at no particular span says so instead of quoting something unrelated.

| magnitude | shift in log-odds |
| --- | --- |
| negligible | 0.00 to 0.05 |
| slight | 0.05 to 0.15 |
| moderate | 0.15 to 0.40 |
| strong | 0.40 and above |

`explanation.text()`, which `str()` also gives you, renders the headline, one
block per reason and the score's notes, so whatever the guard said arrives with
the explanation. `explain(result, limit=3)` keeps the three largest shifts and
says how many smaller ones it left out, and `excerpt_limit` sets how long a quote
may run. The explanation performs no arithmetic of its own, so it cannot disagree
with the score it describes.

## Reference distributions

`proseprobe.REFERENCES` is the table the scale is calibrated against: one row per
indicator feature with its mean and standard deviation in each reference
population, a weight, the minimum text it needs and a note on what it reflects.
`proseprobe.REFERENCE_NOTE` states their provenance. Sixteen of the twenty-seven
measured features have a row; the rest are reported but do not move the score, so
an uncalibrated feature is a missing row rather than a hidden term.

## Features

Pacing and length: `word_count`, `sentence_count`, `paragraph_count`,
`mean_sentence_length`, `sentence_length_stdev`, `sentence_length_cv`,
`burstiness`, `sentence_length_range`, `short_sentence_ratio`,
`long_sentence_ratio`, `mean_paragraph_sentences`.

Lexicon: `function_word_ratio` (its `detail` map holds the per-word profile as a
rate per 1000 words), `function_word_diversity`, `first_person_ratio`,
`discourse_marker_rate`.

Punctuation habits: `comma_rate`, `semicolon_rate`, `colon_rate`, `dash_rate`,
`quote_rate`, `ellipsis_rate`, `question_ratio`, `exclamation_ratio`,
`punctuation_diversity`.

Layout: `list_item_ratio`, `list_item_length_cv`, `heading_ratio`.

The authoritative definitions live in `proseprobe.FEATURES`, a tuple of
`FeatureSpec` entries with a name, a unit and a description. A feature that is
not in that tuple cannot be produced, so a missing measurement is a missing row
rather than a silent gap.

## Development

```bash
pip install -e ".[numpy]"
python -m pytest
```

## License

MIT. See [LICENSE](LICENSE).

Maintained by [Shipmind Labs](https://shipmindlabs.com).
