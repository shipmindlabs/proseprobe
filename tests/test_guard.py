import dataclasses

from proseprobe import (
    CAUTIONS,
    SIGNALS,
    explain,
    extract_features,
    inspect_language,
    score,
    score_features,
)

NON_NATIVE = (
    "Nowadays, many company want to use artificial intelligence for improve their "
    "process. Moreover, managers believe that new tool will reduce cost and also "
    "increase quality of service. Furthermore, workers receive informations about "
    "change very late, therefore they feel confused. Moreover, training is given "
    "only one time, and after that nobody explain me how system must be used in "
    "daily work. In the other hand, some department have good experience, because "
    "their leader discuss about problem with all team before taking decision. "
    "Besides, consultants say that according to me this approach is correct, but "
    "evidences are not strong. Hence, company must prepare better plan, and must "
    "give advices to every employee who use new tool. As a conclusion, artificial "
    "intelligence can help industry, but only when people understand process and "
    "receive support since long time. Nowadays this question is discussed in many "
    "conference, and many researcher try to find answer which satisfy both manager "
    "and worker."
)

NATIVE = (
    "I didn't expect the meeting to run long, but it did. The room was too warm, "
    "the coffee had gone cold an hour earlier, and somebody's laptop kept chirping "
    "at the wrong moments. We talked about the deadline. Then we talked around it. "
    "By the time the whiteboard was full, nobody could remember which of the three "
    "plans we'd actually agreed to, so I wrote the whole thing down on a napkin and "
    "read it back. That's when the argument started again. Priya thought the second "
    "plan was cheaper; Tom said it wasn't cheaper, it was just slower, which isn't "
    "the same thing at all. They're both right, in a way. The budget sheet only "
    "shows what we spend this quarter, and the slow plan pushes half the cost into "
    "the next one. I'll write the summary tonight. It won't please anyone, and that "
    "is probably the sign of a fair compromise. The napkin is in my coat pocket, "
    "slightly torn, and it is still the best record we have of what we decided."
)


def test_native_prose_raises_no_caution():
    guard = inspect_language(NATIVE)
    assert guard.strength == 0.0
    assert guard.reliability == 1.0
    assert guard.caution.name == "none"
    assert guard.fired() == ()
    assert guard.deferred == ()
    assert any("stood out" in note for note in guard.notes)


def test_non_native_patterns_raise_a_caution():
    guard = inspect_language(NON_NATIVE)
    fired = {reading.name for reading in guard.fired()}
    assert {
        "article_rate",
        "contraction_rate",
        "formal_connective_rate",
        "transfer_pattern_rate",
    } <= fired
    assert guard.strength > 0.5
    assert guard.caution.name in ("substantial", "severe")
    assert guard.deferred == ()
    assert guard.caution.name in guard.summary()


def test_fired_signals_quote_the_prose_they_were_read_from():
    collapsed = " ".join(NON_NATIVE.split())
    guard = inspect_language(NON_NATIVE)
    assert any(reading.evidence for reading in guard.fired())
    for reading in guard.fired():
        assert reading.name in reading.sentence()
        for quote in reading.evidence:
            assert quote.rstrip("\u2026") in collapsed


def test_the_guard_damps_the_score_instead_of_inflating_it():
    damped = score(NON_NATIVE)
    undamped = score_features(extract_features(NON_NATIVE))
    assert damped.guard.strength > 0.0
    assert abs(damped.value - 0.5) < abs(undamped.value - 0.5)
    assert damped.width > undamped.width
    assert 0.0 <= damped.low <= damped.value <= damped.high <= 1.0
    assert any("non-native" in note.lower() for note in damped.notes)


def test_the_caution_reaches_the_summary_and_the_explanation():
    result = score(NON_NATIVE)
    assert result.guard.caution.name in result.summary()
    rendered = explain(result).text()
    for note in result.guard.notes:
        assert note in rendered


def test_scoring_from_features_alone_says_the_guard_did_not_run():
    result = score_features(extract_features(NON_NATIVE))
    assert result.guard.strength == 0.0
    assert any("did not run" in note for note in result.notes)


def test_short_text_defers_every_signal_and_says_so():
    guard = inspect_language("Hello there.")
    assert guard.strength == 0.0
    assert guard.readings == ()
    assert set(guard.deferred) == {signal.name for signal in SIGNALS}
    assert any("shorter" in note for note in guard.notes)


def test_every_signal_row_is_documented_and_ramped():
    for signal in SIGNALS:
        assert signal.description.strip()
        assert signal.note.strip()
        assert signal.unit.strip()
        assert signal.weight > 0.0
        assert signal.min_words > 0
        assert signal.threshold != signal.saturation
        assert signal.strength_at(signal.threshold) == 0.0
        assert signal.strength_at(signal.saturation) == 1.0


def test_cautions_tile_the_strength_range():
    assert CAUTIONS[0].low == 0.0
    assert CAUTIONS[-1].high == 1.0
    for earlier, later in zip(CAUTIONS, CAUTIONS[1:]):
        assert earlier.high == later.low
        assert earlier.description.strip()


def test_the_guard_carries_no_boolean_verdict():
    guard = inspect_language(NON_NATIVE)
    for field in dataclasses.fields(guard):
        assert not isinstance(getattr(guard, field.name), bool)
    for reading in guard.readings:
        for field in dataclasses.fields(reading):
            assert not isinstance(getattr(reading, field.name), bool)
    assert not [name for name in dir(guard) if name.startswith(("is_", "has_"))]
