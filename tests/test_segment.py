from proseprobe import (
    LineKind,
    mask_protected,
    protected_spans,
    split_lines,
    split_paragraphs,
    split_sentences,
    split_words,
)

FENCED = "Before.\n\n```python\ntotals = [1; 2];\n```\n\nAfter.\n"


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


def test_a_fenced_block_is_protected_from_its_opening_fence_to_its_closing_one():
    spans = protected_spans(FENCED)
    assert len(spans) == 1
    assert spans[0].text.startswith("```python")
    assert spans[0].text.endswith("```")


def test_an_unclosed_fence_runs_to_the_end_of_the_text():
    text = "Before.\n\n~~~\nstill code\n"
    span = protected_spans(text)[0]
    assert span.start == text.index("~~~")
    assert span.end == len(text)


def test_inline_code_and_urls_are_protected_without_trailing_punctuation():
    text = "Call `run(x; y)` and read https://example.org/docs, then stop."
    assert [span.text for span in protected_spans(text)] == [
        "`run(x; y)`",
        "https://example.org/docs",
    ]


def test_masking_keeps_the_offsets_and_the_sentence_ends():
    masked = mask_protected(FENCED)
    assert len(masked) == len(FENCED)
    assert masked.count("\n") == FENCED.count("\n")
    assert ";" not in masked
    assert [span.text for span in split_sentences(masked)] == ["Before.", "After."]


def test_prose_without_code_or_links_is_left_alone():
    text = "One sentence. Another sentence."
    assert protected_spans(text) == []
    assert mask_protected(text) == text
