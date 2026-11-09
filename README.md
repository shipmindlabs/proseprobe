# proseprobe

Explainable scoring of how machine-like a text reads.

proseprobe reports a calibrated scale with confidence bands, names the
stylometric features that moved the score and quotes the spans they came from,
and applies a guard for non-native English, where detectors systematically fail.
It never returns a binary verdict: "written by a machine" is not a decision this
library is willing to make on your behalf.

## Status

Pre-alpha. Feature extraction, the calibrated scale with its confidence bands,
the quoted explanation, the non-native English guard, the one-call report, the
command line and batch scoring are in place; the bundled reference distributions
and the guard's thresholds are provisional estimates carried in source form so
they can be read and replaced.

## Install

```bash
pip install proseprobe
```

Installing also puts a `probe` command on your path.

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

## Limits and forbidden uses

The score measures style, not authorship. It states how close a text's pacing,
lexicon and punctuation habits sit to each of two reference distributions, and
nothing in the method observes who wrote the text, what tool was open while it
was written, or whether a draft was revised afterwards. A high score means the
prose carries habits that are common in the machine reference population and
uncommon in the human one. Style is imitable, teachable, and often just the
voice of a genre, so that is weaker evidence than it looks.

Four properties of the method set the ceiling on what a score can support.

- Base rates are not in the result. The scale compares one text with two
  populations; it is never told how much of the corpus you are scoring was
  machine-written. Where machine prose is rare, most of the texts that land high
  on the scale are still human, and no setting inside the library can correct
  for that, because the library does not know the mix.
- Style moves under light editing, in both directions. Reordering sentences,
  cutting connectives or varying sentence length pulls a score down; asking a
  model for uneven pacing pulls it down too. Anyone who wants a particular score
  can have one, so a low score is not a clearance and a high score is not a
  confession.
- The references are provisional. `REFERENCE_NOTE` says as much in the package:
  coarse summary statistics, not a fitted model, bundled in source form so they
  can be read and replaced. A score is reproducible and auditable; it is not
  validated against a corpus large enough to carry a decision about anyone.
- Whole genres sit near the machine end for reasons of their own: edited
  corporate prose, technical documentation, translated text, writing in a second
  language, and anything written to a template. The scale cannot tell a house
  style from a model.

So a score must never be the sole evidence for a decision about a person, and
there are uses it is unfit for at any confidence level. These are design limits,
not gaps waiting to be closed:

- Academic misconduct findings, grading penalties, or any disciplinary outcome.
- Hiring, admissions, promotion or dismissal decisions.
- Moderation that removes content, withholds payment or suspends an account.
- Any claim about a named author, an identifiable group or a language community.
- Any decision that is irreversible, unappealable, or invisible to the person it
  affects.

What a score is good for is reading a text more closely than you otherwise
would: deciding where to look first in a large corpus, opening a conversation
with an author about a draft rather than closing one, watching a corpus drift
over time, and studying the features themselves. In each of those the score is
one input among others, and the person it concerns can see it and answer it.

The library hands you no threshold and no boolean, in the result, the report or
the command line, because the step from a number to a verdict is where the harm
happens and it is not a step this library takes on your behalf. Writing
`result.value > 0.7` in your own code restores the verdict the scale declined to
make, and the responsibility that comes with it.

Read a result as the whole of what it carries: the interval rather than the
point value, every band the interval touches, the notes on what limited it, and
the quoted spans behind each contribution. An interval across three or more
bands means the point value carries little information, and the notes say so.
Below a few hundred words there is little to measure, and the notes say that
too.

The failure this library takes most seriously is not symmetric. The features
that separate the reference populations also separate first-language writers
from second-language ones: measured pacing, textbook connectives and a sparing
use of contractions are taught as careful writing and are also what default
model prose looks like. Left uncorrected, a score of machine-likeness doubles as
a score of foreign authorship, and the people it misreads are the least likely
to be in a position to appeal it. The guard described next measures those
patterns and spends them on caution only: the scale keeps `1 - strength` of the
measured evidence, which pulls the point value toward the prior of 0.5, and
widens the interval in proportion, so a caution can never raise a score. It
names no language and makes no claim about the author. It is also not a shield
against the limits above: a text with no caution is still a text whose style was
measured and whose authorship was not.

## The non-native English guard

Detectors of machine-like prose systematically over-score competent non-native
English: measured pacing, textbook connectives and a thin stock of contractions
read as machine habits whoever wrote them. The guard measures the patterns that
point at a non-native author and reports how far the score should be trusted. It
names no language and makes no claim about the author.

`inspect_language(text)` returns a `Guard` on its own, and `score(text)` runs it
and carries it on `result.guard`. It reads the text with code blocks, inline
code and URLs masked out, as the features do. Each signal becomes a `Reading`
with its measured value and unit, the value where it starts to count, the value
where it counts fully, its `strength` between those two and the quoted spans it
was read from. `guard.strength` is the weighted mean of those strengths over all
signals, `guard.reliability` is what is left of 1, `guard.caution` names the
stretch the strength falls in, and `guard.words` and `guard.sentences` give the
denominator the rates were computed over.

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

## One report

`report(text)` is the whole library in one call: it measures the text, places it
on the scale, runs the guard, writes the explanation and returns a JSON-ready
dictionary.

```python
import json

from proseprobe import report

print(json.dumps(report(text, name="essay.txt"), indent=2))
```

A report carries a `report_version`, the `calibration` block described under
batch scoring, the `name` and `path` it was given, the `words` it was scored
over, the `score` whole — the point value, the interval at its confidence, the
band and every band the interval touches, each contribution with its reference
means, its signed shift and the spans it was read from, the guard with each of
its readings, and the notes — and the `explanation` as text. Nothing is
summarised away, so a report can be argued with on its own.

It accepts the same `confidence` as `score()` and the same `limit` and
`excerpt_limit` as `explain()`. `report_score(result)` renders a `Score` already
in hand in the same structure.

There is one report structure and three ways to reach it: this function, the
command line, which prints it unchanged, and batch scoring, which puts it under
each document. `report(text, name="a.txt")` and that document's entry in a batch
are the same dictionary.

## Command line

```bash
probe essay.txt
probe essay.txt --json
cat essay.txt | probe
```

Without `--json` the report is printed as sentences: the headline, one block per
feature with its quoted excerpt, the notes, and the guard's line. With `--json`
the dictionary `report()` returns is printed as it stands. The command renders
what the library gives it and computes nothing of its own, so the two cannot
disagree.

| option | effect |
| --- | --- |
| `--json` | print the report as JSON instead of sentences |
| `--confidence LEVEL` | confidence level for the interval (default 0.95) |
| `--limit N` | keep only the N largest feature shifts in the explanation |
| `--excerpt-limit CHARS` | how long a quoted excerpt may run (default 120) |
| `--indent N` | JSON indentation; `0` prints one line |
| `--version` | print the version |

Several files can be named at once and each keeps its own report, so `--json`
prints one object for a single file and a list, in the order the files were
given, for more than one. With no file, or with `-`, the text is read from
standard input. A file that cannot be read is named on standard error, the rest
are scored anyway, and the exit status is 1. `python -m proseprobe` runs the same
command.

## Batch scoring

A directory or a list of documents is scored in one pass, and every document
keeps its own report.

```python
from proseprobe import score_directory, write_reports

batch = score_directory("corpus")

print(batch.headline())

for document in batch.documents:
    print(document.name, round(document.score.value, 2), document.score.band.name)

write_reports(batch, "reports")
```

`score_directory(path)` reads `*.txt`, `*.md` and `*.markdown` below `path`,
which `patterns` overrides, and names each document by its path relative to the
directory. `score_paths([...])` takes the files you name, and
`score_documents({"name": text})` takes prose you already hold. All three accept
the same `confidence` as `score()` and return a `BatchReport`.

`write_reports(batch, directory)` writes one `<document>.json` per document plus
`summary.json`, and returns the paths it wrote. Nothing is written unless you ask
for it: `document.to_dict()` and `batch.summary()` return JSON-ready
dictionaries, and `to_json()` renders them.

A document report carries the score whole, in the structure `report()` returns,
so nothing is summarised away and a report can be argued with on its own.

The summary states a distribution: how many documents were scored and over how
many words, the mean, median, lowest and highest value with the mean interval
width, how many documents fell in each band and each caution, the features that
moved the scores most by mean absolute shift, one line per document pointing at
its report file, the files that could not be read as UTF-8 text, and notes in
plain sentences. It ranks nothing and holds no threshold, so a batch of numbers
does not become a batch of verdicts.

Every document report and the batch summary carry the same `calibration` block:
a `fingerprint`, which is the sha256 digest of canonical JSON over the reference
distributions, the guard's signals, the bands, the cautions and the constants of
the scale, and `note`, the provenance line `REFERENCE_NOTE` states. The
fingerprint identifies the tables that were in force when the report was
written; it says nothing about a corpus they were fitted to, because they are
not fitted. Two reports with the same fingerprint were scored against the same
tables, and replacing `REFERENCES` or moving a guard threshold changes it.
`calibration_fingerprint()` returns it on its own, so a stored report can be
checked against the tables you hold now.

A file that cannot be read is named in `batch.unread` with the reason, and the
rest of the batch is scored anyway.

## Reference distributions

`proseprobe.REFERENCES` is the table the scale is calibrated against: one row per
indicator feature with its mean and standard deviation in each reference
population, a weight, the minimum text it needs and a note on what it reflects.
`proseprobe.REFERENCE_NOTE` states their provenance. Sixteen of the twenty-eight
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

What was left out: `protected_word_count`.

The authoritative definitions live in `proseprobe.FEATURES`, a tuple of
`FeatureSpec` entries with a name, a unit and a description. A feature that is
not in that tuple cannot be produced, so a missing measurement is a missing row
rather than a silent gap.

## Code and links are not prose

Fenced code blocks, inline code between single backticks and URLs are masked out
before anything is measured, so a semicolon-heavy snippet or a query string does
not read as punctuation habit. `protected_spans(text)` returns those spans with
their offsets, and `mask_protected(text)` replaces every non-newline character
inside them with a space, which keeps the offsets and the line breaks in place so
the surrounding prose segments exactly as it did before. A fence runs from the
line that opens it to its closing fence, or to the end of the text if it is never
closed, and a URL leaves its trailing `.,;:!?)` to the sentence around it.

The word tokens inside those spans are reported as `protected_word_count` rather
than dropped in silence, and `score()` says so in its notes whenever the count is
above zero. Quotation marks around inline quotations are left alone, since
`quote_rate` is calibrated on them.

## Development

```bash
pip install -e ".[numpy]"
python -m pytest
```

## License

MIT. See [LICENSE](LICENSE).

Maintained by [Shipmind Labs](https://shipmindlabs.com).
