"""Segmentation of raw text into lines, paragraphs, sentences and words.

Every segment carries its character offsets into the original text, so a feature
can quote the exact span that produced it instead of paraphrasing it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "Line",
    "LineKind",
    "Span",
    "split_lines",
    "split_paragraphs",
    "split_sentences",
    "split_words",
]

WORD_RE = re.compile(r"[^\W\d_]+(?:['\u2019-][^\W\d_]+)*")

_SENTENCE_END_RE = re.compile(r"[.!?\u2026]+[\"'\u201d\u2019)\]]*")
_ATX_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+\S")
_RULE_RE = re.compile(r"^\s{0,3}(?:=+|-{2,}|\*{3,}|_{3,})\s*$")
_LIST_RE = re.compile(r"^\s{0,8}(?:[-*+\u2022]|\(?\d{1,3}[.)]|\(?[a-zA-Z][.)])\s+\S")
_PARAGRAPH_BREAK_RE = re.compile(r"\n[ \t]*\n\s*")

_ABBREVIATIONS = frozenset(
    {
        "a.m", "al", "approx", "cf", "co", "dept", "dr", "e.g", "est", "etc",
        "fig", "i.e", "inc", "jr", "ltd", "mr", "mrs", "ms", "no", "p.m",
        "ph.d", "prof", "sr", "st", "u.k", "u.s", "viz", "vs",
    }
)

HEADING_MAX_CHARS = 80
HEADING_MAX_WORDS = 10


@dataclass(frozen=True, slots=True)
class Span:
    """A character range in the source text, with the text it covers."""

    start: int
    end: int
    text: str

    def __len__(self) -> int:
        return self.end - self.start

    def quote(self, limit: int = 160) -> str:
        """Whitespace-collapsed excerpt, truncated to ``limit`` characters."""
        collapsed = " ".join(self.text.split())
        if len(collapsed) <= limit:
            return collapsed
        return collapsed[: limit - 1].rstrip() + "\u2026"


class LineKind(str, Enum):
    """What a single line of the source looks like."""

    BLANK = "blank"
    RULE = "rule"
    HEADING = "heading"
    LIST_ITEM = "list_item"
    TEXT = "text"


@dataclass(frozen=True, slots=True)
class Line:
    """A source line and its kind."""

    span: Span
    kind: LineKind


def _trimmed(text: str, start: int, end: int) -> Span | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start >= end:
        return None
    return Span(start, end, text[start:end])


def split_words(text: str) -> list[Span]:
    """Alphabetic word tokens; bare digits and symbols are not words."""
    return [Span(m.start(), m.end(), m.group(0)) for m in WORD_RE.finditer(text)]


def split_paragraphs(text: str) -> list[Span]:
    """Blocks separated by one or more blank lines."""
    spans: list[Span] = []
    cursor = 0
    for match in _PARAGRAPH_BREAK_RE.finditer(text):
        span = _trimmed(text, cursor, match.start())
        if span is not None:
            spans.append(span)
        cursor = match.end()
    tail = _trimmed(text, cursor, len(text))
    if tail is not None:
        spans.append(tail)
    return spans


def split_lines(text: str) -> list[Line]:
    """Lines with their kind: blank, rule, heading, list item or body text."""
    bounds: list[tuple[int, int]] = []
    position = 0
    for raw in text.splitlines(keepends=True):
        stripped = raw.rstrip("\r\n")
        bounds.append((position, position + len(stripped)))
        position += len(raw)

    contents = [text[start:end].strip() for start, end in bounds]
    kinds: list[LineKind] = []
    for index, (start, end) in enumerate(bounds):
        raw_line = text[start:end]
        if not contents[index]:
            kinds.append(LineKind.BLANK)
        elif _ATX_HEADING_RE.match(raw_line):
            kinds.append(LineKind.HEADING)
        elif _RULE_RE.match(raw_line):
            kinds.append(LineKind.RULE)
            if index and kinds[index - 1] is LineKind.TEXT:
                kinds[index - 1] = LineKind.HEADING
        elif _LIST_RE.match(raw_line):
            kinds.append(LineKind.LIST_ITEM)
        else:
            kinds.append(LineKind.TEXT)

    for index, kind in enumerate(kinds):
        if kind is LineKind.TEXT and _looks_like_heading(contents, kinds, index):
            kinds[index] = LineKind.HEADING

    return [
        Line(Span(start, end, text[start:end]), kind)
        for (start, end), kind in zip(bounds, kinds)
    ]


def _looks_like_heading(contents: list[str], kinds: list[LineKind], index: int) -> bool:
    content = contents[index]
    if len(content) > HEADING_MAX_CHARS or content[-1] in ".!?\u2026,;":
        return False
    if not (content[:1].isupper() or content[:1].isdigit()):
        return False
    if len(WORD_RE.findall(content)) > HEADING_MAX_WORDS:
        return False
    before = kinds[index - 1] if index else LineKind.BLANK
    if before not in (LineKind.BLANK, LineKind.RULE):
        return False
    after = kinds[index + 1] if index + 1 < len(kinds) else LineKind.BLANK
    return after in (LineKind.BLANK, LineKind.LIST_ITEM) or content.endswith(":")


def split_sentences(text: str) -> list[Span]:
    """Sentences; headings and list items stay whole even without end marks."""
    spans: list[Span] = []
    for block in _blocks(split_lines(text)):
        spans.extend(_split_region(text, block[0].span.start, block[-1].span.end))
    return spans


def _blocks(lines: Iterable[Line]) -> Iterator[list[Line]]:
    block: list[Line] = []
    for line in lines:
        if line.kind in (LineKind.BLANK, LineKind.RULE):
            if block:
                yield block
                block = []
        elif line.kind in (LineKind.HEADING, LineKind.LIST_ITEM):
            if block:
                yield block
                block = []
            yield [line]
        else:
            block.append(line)
    if block:
        yield block


def _split_region(text: str, start: int, end: int) -> list[Span]:
    spans: list[Span] = []
    cursor = start
    for match in _SENTENCE_END_RE.finditer(text, start, end):
        if not _is_boundary(text, match, end):
            continue
        span = _trimmed(text, cursor, match.end())
        if span is not None:
            spans.append(span)
        cursor = match.end()
    tail = _trimmed(text, cursor, end)
    if tail is not None:
        spans.append(tail)
    return spans


def _is_boundary(text: str, match: re.Match[str], end: int) -> bool:
    position = match.end()
    if position >= end:
        return True
    if not text[position].isspace():
        return False
    following = position
    while following < end and text[following].isspace():
        following += 1
    if following < end and text[following].islower():
        return False
    if match.group(0) != ".":
        return True
    token = _token_before(text, match.start()).strip(".")
    if not token:
        return True
    if len(token) == 1 and token.isupper():
        return False
    return token.lower() not in _ABBREVIATIONS


def _token_before(text: str, index: int) -> str:
    chars: list[str] = []
    position = index - 1
    while position >= 0 and (text[position].isalpha() or text[position] == "."):
        chars.append(text[position])
        position -= 1
    return "".join(reversed(chars))
