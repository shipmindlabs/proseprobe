from proseprobe import FEATURE_SPECS, extract_features

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
