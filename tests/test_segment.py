from proseprobe import LineKind, split_lines, split_paragraphs, split_sentences, split_words


def test_words_are_alphabetic_tokens():
    assert [word.text for word in split_words("It cost 42 dollars, roughly.")] == [
        "It",
        "cost",
        "dollars",
        "roughly",
    ]


def test_spans_point_back_at_the_source():
    text = "First sentence. Second sentence."
    for span in split_sentences(text):
        assert text[span.start : span.end] == span.text


def test_abbreviations_do_not_end_a_sentence():
    sentences = split_sentences("Dr. Smith went home. He slept.")
    assert [span.text for span in sentences] == ["Dr. Smith went home.", "He slept."]


def test_lowercase_continuation_is_not_a_boundary():
    assert len(split_sentences("We shipped v1.2 yesterday. It works.")) == 2


def test_headings_and_list_items_are_classified():
    text = "Key points\n\n- first item\n- second item\n\nBody text follows here.\n"
    kinds = [line.kind for line in split_lines(text)]
    assert kinds[0] is LineKind.HEADING
    assert kinds.count(LineKind.LIST_ITEM) == 2
    assert kinds[-1] in (LineKind.TEXT, LineKind.BLANK)


def test_list_items_stay_whole_without_end_punctuation():
    sentences = split_sentences("- first item\n- second item\n")
    assert [span.text for span in sentences] == ["- first item", "- second item"]


def test_paragraphs_split_on_blank_lines():
    assert len(split_paragraphs("One.\n\nTwo.\n\n\nThree.")) == 3
