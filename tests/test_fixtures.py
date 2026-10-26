import json
from pathlib import Path
from statistics import fmean

import pytest

from proseprobe import REFERENCES, extract_features, score

FIXTURES = Path(__file__).parent / "fixtures"
LABELS = json.loads((FIXTURES / "labels.json").read_text(encoding="utf-8"))
CASES = LABELS["fixtures"]
IDS = [case["file"] for case in CASES]
SEPARATION = 0.1
MIN_WORDS = max(reference.min_words for reference in REFERENCES)
MIN_SENTENCES = max(reference.min_sentences for reference in REFERENCES)


def _text(case):
    return (FIXTURES / case["file"]).read_text(encoding="utf-8")


def _labelled(label):
    return [case for case in CASES if case["label"] == label]


def test_the_labelled_set_holds_both_populations():
    assert {case["label"] for case in CASES} == {"human", "generated"}
    assert len(_labelled("human")) >= 3
    assert len(_labelled("generated")) >= 3
    assert len({case["file"] for case in CASES}) == len(CASES)
    assert LABELS["note"].strip()


def test_every_fixture_is_labelled_and_every_label_names_a_file():
    assert {path.name for path in FIXTURES.glob("*.txt")} == {case["file"] for case in CASES}
    for case in CASES:
        assert case["source"].strip()
        assert 0.0 <= case["expect"]["low"] < case["expect"]["high"] <= 1.0
        assert 0.0 <= case["expect"]["max_guard_strength"] <= 1.0
        assert _text(case).strip()


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_every_fixture_is_long_enough_for_every_reference_row(case):
    features = extract_features(_text(case))
    assert features.value("word_count") >= MIN_WORDS
    assert features.value("sentence_count") >= MIN_SENTENCES
    assert not [note for note in score(_text(case)).notes if note.startswith("Not counted")]


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_every_fixture_scores_inside_its_expected_range(case):
    result = score(_text(case))
    expect = case["expect"]
    assert expect["low"] <= result.value <= expect["high"], (
        f"{case['file']} ({case['label']}) scored {result.value:.3f} in band "
        f"'{result.band.name}', outside the expected {expect['low']}..{expect['high']}"
    )
    assert 0.0 <= result.low <= result.value <= result.high <= 1.0


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_the_guard_stays_within_the_expected_caution(case):
    guard = score(_text(case)).guard
    assert guard.strength <= case["expect"]["max_guard_strength"], (
        f"{case['file']} raised a {guard.caution.name} caution at strength "
        f"{guard.strength:.2f} from {sorted(r.name for r in guard.fired())}"
    )


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_every_fixture_quotes_the_prose_its_contributions_were_read_from(case):
    text = _text(case)
    collapsed = " ".join(text.split())
    result = score(text)
    assert result.contributions
    assert any(contribution.evidence for contribution in result.contributions)
    for contribution in result.contributions:
        for quote in contribution.evidence:
            assert quote.rstrip("\u2026") in collapsed


def test_the_human_fixtures_score_below_the_generated_ones():
    human = [score(_text(case)).value for case in _labelled("human")]
    generated = [score(_text(case)).value for case in _labelled("generated")]
    assert fmean(human) + SEPARATION <= fmean(generated), (
        f"human mean {fmean(human):.3f} and generated mean {fmean(generated):.3f} "
        f"are closer than {SEPARATION}"
    )
