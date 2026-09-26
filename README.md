# proseprobe

Explainable scoring of how machine-like a text reads.

proseprobe reports a calibrated scale with confidence bands, names the
stylometric features that moved the score and quotes the spans they came from,
and applies a guard for non-native English, where detectors systematically fail.
It never returns a binary verdict: "written by a machine" is not a decision this
library is willing to make on your behalf.

## Status

Pre-alpha. Feature extraction is in place; calibration, confidence bands and the
non-native guard are being built in the open.

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
`features.to_dict()` gives a plain name-to-number mapping.

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
