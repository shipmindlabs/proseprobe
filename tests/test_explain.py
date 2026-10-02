import dataclasses
import math

from proseprobe import MAGNITUDES, explain, score

BURSTY = (
    "Yes. The argument, which had been rehearsed for weeks in rooms that smelled "
    "of old coffee and older ambition, collapsed the moment anyone asked it a "
    "plain question about the data. Odd. Nobody minded much."
)


def test_every_contribution_becomes_a_reason():
    result = score(BURSTY)
    explanation = explain(result)
    assert [reason.name for reason in explanation.reasons] == [
        contribution.name for contribution in result.contributions
    ]
    assert explanation.headline == result.summary()
    assert explanation.notes == result.notes
    assert explanation.omitted == 0


def test_each_reason_states_a_direction_and_a_size():
    for reason in explain(score(BURSTY)).reasons:
        assert reason.direction in (
            "toward the machine reference",
            "toward the human reference",
            "neutral",
        )
        assert reason.magnitude.floor <= abs(reason.logit_shift) < reason.magnitude.ceiling
        assert reason.magnitude.name in reason.sentence()
        assert f"{reason.logit_shift:+.3f}" in reason.sentence()


def test_shares_of_the_movement_add_up():
    explanation = explain(score(BURSTY))
    assert explanation.reasons
    assert math.isclose(sum(reason.share for reason in explanation.reasons), 1.0, abs_tol=1e-9)


def test_each_reason_quotes_prose_or_says_there_is_none():
    collapsed = " ".join(BURSTY.split())
    explanation = explain(score(BURSTY))
    rendered = explanation.text()
    for reason in explanation.reasons:
        assert reason.sentence() in rendered
        assert reason.quotation() in rendered
        if reason.excerpt:
            assert reason.excerpt.rstrip("\u2026") in collapsed
        else:
            assert reason.quotation().startswith("(")
    assert any(reason.excerpt for reason in explanation.reasons)


def test_excerpts_stay_within_the_requested_length():
    for reason in explain(score(BURSTY), excerpt_limit=40).reasons:
        assert len(reason.excerpt) <= 40


def test_a_limit_says_how_many_shifts_it_left_out():
    result = score(BURSTY)
    explanation = explain(result, limit=2)
    assert len(explanation.reasons) == 2
    assert explanation.omitted == len(result.contributions) - 2
    assert explanation.omitted > 0
    assert f"{explanation.omitted} smaller shifts" in explanation.text()


def test_the_text_keeps_the_headline_and_the_notes():
    result = score(BURSTY)
    rendered = str(explain(result))
    assert rendered.startswith(result.summary())
    for note in result.notes:
        assert note in rendered


def test_a_text_below_every_minimum_has_no_reasons_but_keeps_its_notes():
    explanation = explain(score("Hello there."))
    assert explanation.reasons == ()
    assert explanation.notes
    assert "prior" in explanation.text()


def test_magnitudes_tile_the_sizes():
    assert MAGNITUDES[0].floor == 0.0
    assert MAGNITUDES[-1].ceiling == math.inf
    for earlier, later in zip(MAGNITUDES, MAGNITUDES[1:]):
        assert earlier.ceiling == later.floor
        assert earlier.name.strip()


def test_an_explanation_carries_no_boolean_verdict():
    explanation = explain(score(BURSTY))
    for field in dataclasses.fields(explanation):
        assert not isinstance(getattr(explanation, field.name), bool)
    for reason in explanation.reasons:
        for field in dataclasses.fields(reason):
            assert not isinstance(getattr(reason, field.name), bool)
