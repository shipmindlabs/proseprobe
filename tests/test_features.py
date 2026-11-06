from proseprobe import FEATURE_SPECS, extract_features, inspect_language, score, split_words

UNIFORM = (
    "The system works well. The system runs fast. The system stays calm. "
    "The system holds up. The system scales out."
)
BURSTY = (
    "Yes. The argument, which had been rehearsed for weeks in rooms that smelled "
    "of old coffee and older ambition, collapsed the moment anyone asked it a "
    "plain question about the data. Odd. Nobody minded much."
)
STRUCTURED = (
    "Setup checklist:\n"
    "\n"
    "- Install the package\n"
    "- Run the tests\n"
    "- Ship it\n"
    "\n"
    "This takes a minute; it is, however, worth doing.\n"
)
PROSE = (
    "Field notes\n"
    "\n"
    "We walked the river at dusk; nobody spoke for a while.\n"
    "The water was low. A heron lifted off the far bank, and the dog froze.\n"
    "Later we argued about the map and then agreed to disagree.\n"
    "\n"
    "The track climbs for a mile past the old weir, where the gravel gives way\n"
    "to mud and the going slows. I hadn't brought the right boots, so the last\n"
    "stretch was a matter of stubbornness rather than skill. A farmer waved from\n"
    "a gate and told us the bridge had been out since the spring floods, which\n"
    "explained the detour we had seen marked on a hand-drawn sign. We turned back\n"
    "at the second bend, happy enough with what the afternoon had given us, and\n"
    "the dog slept in the car all the way home.\n"
)
MIXED = PROSE + (
    "\n"
    "```python\n"
    "totals = [alpha; beta; gamma];\n"
    "print(totals); print(totals); print(totals);\n"
    "```\n"
    "\n"
    "https://example.org/field-notes?ref=a;b;c\n"
)


def test_every_feature_is_named_and_documented():
    features = extract_features(UNIFORM)
    assert set(features) == set(FEATURE_SPECS)
    for feature in features.values():
        assert feature.description.strip()
        assert feature.unit.strip()


def test_burstiness_separates_uniform_from_varied_prose():
    assert extract_features(BURSTY).value("burstiness") > extract_features(UNIFORM).value("burstiness")
    assert extract_features(UNIFORM).value("sentence_length_stdev") < 1.0
    assert extract_features(BURSTY).value("sentence_length_range") > 20


def test_features_quote_the_spans_they_were_read_from():
    feature = extract_features(BURSTY)["burstiness"]
    assert feature.evidence
    assert any(quote.startswith("Yes.") for quote in feature.evidence)


def test_function_word_profile_is_a_rate_per_1000_words():
    features = extract_features(UNIFORM)
    assert 0.0 < features.value("function_word_ratio") < 1.0
    assert features["function_word_ratio"].detail["the"] == 250.0


def test_punctuation_and_layout_features():
    features = extract_features(STRUCTURED)
    assert features.value("list_item_ratio") > 0.5
    assert features.value("heading_ratio") > 0.0
    assert features.value("semicolon_rate") > 0.0
    assert features.value("comma_rate") > 0.0
    assert features.value("discourse_marker_rate") > 0.0
    assert features["list_item_ratio"].evidence


def test_templated_list_items_have_low_length_variation():
    features = extract_features("- alpha beta gamma\n- delta epsilon zeta\n- eta theta iota\n")
    assert features.value("list_item_length_cv") == 0.0


def test_empty_text_yields_zeroed_features():
    features = extract_features("")
    assert set(features) == set(FEATURE_SPECS)
    assert all(value == 0.0 for value in features.to_dict().values())


def test_a_fenced_block_and_a_url_do_not_reach_the_measurements():
    plain = extract_features(PROSE)
    mixed = extract_features(MIXED)
    for name in ("word_count", "sentence_count", "semicolon_rate", "comma_rate"):
        assert mixed.value(name) == plain.value(name)
    assert inspect_language(PROSE).readings
    assert inspect_language(MIXED).readings == inspect_language(PROSE).readings


def test_the_words_left_out_are_counted():
    masked_tokens = len(split_words(MIXED)) - len(split_words(PROSE))
    assert masked_tokens > 0
    assert extract_features(MIXED).value("protected_word_count") == masked_tokens
    assert extract_features(PROSE).value("protected_word_count") == 0.0


def test_the_score_notes_how_many_words_it_left_out():
    count = int(extract_features(MIXED).value("protected_word_count"))
    note = f"{count} words inside code and URLs were left out of every measurement."
    assert note in score(MIXED).notes
    assert not [line for line in score(PROSE).notes if "left out of every measurement" in line]
