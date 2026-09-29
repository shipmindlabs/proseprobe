import dataclasses
import math

from proseprobe import BANDS, FEATURE_SPECS, REFERENCES, extract_features, score, score_features

UNIFORM = (
    "The system works well. The system runs fast. The system stays calm. "
    "The system holds up. The system scales out."
)
BURSTY = (
    "Yes. The argument, which had been rehearsed for weeks in rooms that smelled "
    "of old coffee and older ambition, collapsed the moment anyone asked it a "
    "plain question about the data. Odd. Nobody minded much."
)


def test_the_point_value_sits_inside_its_interval():
    for text in (UNIFORM, BURSTY, ""):
        result = score(text)
        assert 0.0 <= result.low <= result.value <= result.high <= 1.0
        assert result.interval == (result.low, result.high)
        assert result.band in result.spanned_bands


def test_the_result_carries_no_boolean_verdict():
    result = score(BURSTY)
    for field in dataclasses.fields(result):
        assert not isinstance(getattr(result, field.name), bool)
    for contribution in result.contributions:
        for field in dataclasses.fields(contribution):
            assert not isinstance(getattr(contribution, field.name), bool)
    assert not [name for name in dir(result) if name.startswith(("is_", "has_"))]


def test_even_pacing_scores_higher_than_bursty_pacing():
    assert score(UNIFORM).value > score(BURSTY).value


def test_a_wider_confidence_level_gives_a_wider_band():
    assert score(BURSTY, confidence=0.99).width > score(BURSTY, confidence=0.80).width


def test_text_below_every_minimum_reports_the_prior_and_says_why():
    result = score("Hello there.")
    assert result.value == 0.5
    assert result.interval == (0.0, 1.0)
    assert result.contributions == ()
    assert result.spanned_bands == BANDS
    assert any("prior" in note for note in result.notes)


def test_contributions_are_ordered_and_add_up_to_the_score():
    result = score(BURSTY)
    shifts = [contribution.logit_shift for contribution in result.contributions]
    assert shifts
    assert shifts == sorted(shifts, key=abs, reverse=True)
    assert math.isclose(sum(shifts), math.log(result.value / (1.0 - result.value)), abs_tol=1e-6)
    assert result.moved_by(2) == result.contributions[:2]


def test_contributions_quote_the_prose_they_were_read_from():
    result = score(BURSTY)
    assert any(contribution.evidence for contribution in result.contributions)
    assert all(
        contribution.direction()
        in ("toward the machine reference", "toward the human reference", "neutral")
        for contribution in result.contributions
    )


def test_scoring_a_feature_set_matches_scoring_the_text():
    assert score_features(extract_features(UNIFORM)).value == score(UNIFORM).value


def test_every_reference_row_names_a_documented_feature():
    for reference in REFERENCES:
        assert reference.feature in FEATURE_SPECS
        assert reference.human_stdev > 0.0
        assert reference.machine_stdev > 0.0
        assert reference.human_mean != reference.machine_mean
        assert reference.note.strip()


def test_bands_tile_the_whole_scale():
    assert BANDS[0].low == 0.0
    assert BANDS[-1].high == 1.0
    for earlier, later in zip(BANDS, BANDS[1:]):
        assert earlier.high == later.low
        assert earlier.description.strip()


def test_summary_names_the_band_and_the_interval():
    result = score(UNIFORM)
    line = result.summary()
    assert result.band.name in line
    assert f"{result.low:.2f}" in line
