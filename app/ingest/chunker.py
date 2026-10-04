"""Heading-aware chunking (spec §8.1, FR-2).

1. Walk the pages line by line and cut at section headings: numbered
   ("4. Vesting", "4.1 Cliff") or ALL-CAPS ("SCHEDULE A"). The current
   section carries over to the next page until a new heading appears.
2. Text is grouped by (page, section), so no chunk ever spans two pages and
   every citation points to one page.
3. A group whose embed_text fits the embedding model's input window becomes
   one chunk. Longer groups are split with RecursiveCharacterTextSplitter,
   measured in model tokens and preferring clause boundaries.

Chunk sizes deviate from the 500/80 in spec §8.1 on purpose: all-MiniLM-L6-v2
reads at most 256 tokens and silently truncates the rest, so chunks are
capped to fit the model (see learning/phase-03-ingestion.md).
"""

import re
from collections.abc import Callable
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ingest.loader import PageText

# Spec §8.1 regex, fixed with `\.?` so "4. Vesting" matches as well as "4.1 Cliff".
HEADING_NUMBER_RE = re.compile(r"^\d+(\.\d+)*\.?\s")
# Headings are short; wrapped body lines run the full page width (~95-105 characters).
MAX_HEADING_CHARS = 80
# Dates such as "1 January 2026 (cliff)" in grant-letter tables look like "<number> <Title>".
DATE_RE = re.compile(r"^\d{1,2} [A-Z][a-z]+ \d{4}\b")
# Characters allowed in an ALL-CAPS heading besides uppercase letters.
ALL_CAPS_RE = re.compile(r"^[A-Z][A-Z &'/,()-]*$")
MIN_ALL_CAPS_LETTERS = 3  # "TAX" is a heading; "OR" is not

PREAMBLE = "Preamble"  # section name for text before the first heading

# Split sizes in model tokens. A chunk's embed_text must fit the model window
# (254 tokens for MiniLM); the title/section prefix takes ~10-25 of those.
CHUNK_TOKENS = 200
CHUNK_OVERLAP_TOKENS = 50
# Prefer cutting before a numbered clause ("\n4.2 ..."), then after a line that ends a
# sentence (a paragraph end, since pypdf joins wrapped lines with "\n"), then at any
# line, sentence, word. Non-capturing groups only: the splitter adds its own.
SEPARATORS = [r"\n(?=\d+(?:\.\d+)+\s)", r"(?<=[.:;])\n", r"\n", r"(?<=\.) ", " ", ""]

TokenCounter = Callable[[str], int]


@dataclass(frozen=True)
class Chunk:
    """One retrievable unit: text from a single page and section."""

    text: str
    page: int
    section: str
    embed_text: str


def is_heading(line: str) -> bool:
    """True if a (normalised) line is a numbered or ALL-CAPS section heading.

    The number regex alone also matches wrapped body lines ("15 November 2024. The ..."),
    table cells ("48 months") and dates ("1 January 2026"), so a heading must also be
    short, start with a capital letter after the number, not be a date, and not end
    like a sentence.
    """
    if len(line) > MAX_HEADING_CHARS or line.endswith((".", ",", ";", ":")) or DATE_RE.match(line):
        return False
    if match := HEADING_NUMBER_RE.match(line):
        rest = line[match.end():]
        return rest[:1].isupper()
    letters = sum(c.isalpha() for c in line)
    return bool(ALL_CAPS_RE.match(line)) and letters >= MIN_ALL_CAPS_LETTERS


def split_sections(pages: list[PageText]) -> list[tuple[int, str, str]]:
    """Group lines into (page, section, body) blocks, in document order.

    A heading line becomes the section name (not part of the body). Blocks
    with an empty body (e.g. a heading that is the last line of a page) are dropped.
    """
    blocks: list[tuple[int, str, str]] = []
    section = PREAMBLE
    for page, text in pages:
        body: list[str] = []
        for line in text.splitlines():
            if is_heading(line):
                if body:
                    blocks.append((page, section, "\n".join(body)))
                section, body = line, []
            else:
                body.append(line)
        if body:  # page ends: close the block; the section carries over to the next page
            blocks.append((page, section, "\n".join(body)))
    return blocks


def embed_prefix(doc_title: str, section: str) -> str:
    """The '<doc title> > <section>' line put in front of every chunk before embedding."""
    return f"{doc_title} > {section}\n"


def make_splitter(count_tokens: TokenCounter, chunk_tokens: int, overlap_tokens: int) -> RecursiveCharacterTextSplitter:
    """A RecursiveCharacterTextSplitter whose sizes are measured in model tokens."""
    return RecursiveCharacterTextSplitter(
        separators=SEPARATORS,
        is_separator_regex=True,
        keep_separator="start",
        chunk_size=chunk_tokens,
        chunk_overlap=overlap_tokens,
        length_function=count_tokens,
    )


def chunk_document(
    pages: list[PageText],
    doc_title: str,
    count_tokens: TokenCounter | None = None,
    max_tokens: int | None = None,
) -> list[Chunk]:
    """Split a document's pages into heading-aware, single-page chunks.

    count_tokens and max_tokens default to the embedding model's tokenizer and
    input window; tests pass simple stand-ins so they need no model.
    """
    if count_tokens is None or max_tokens is None:
        from app.ingest import embedder  # lazy: loads config and the model

        count_tokens = count_tokens or embedder.count_tokens
        max_tokens = max_tokens or embedder.max_input_tokens()

    chunks: list[Chunk] = []
    for page, section, body in split_sections(pages):
        prefix = embed_prefix(doc_title, section)
        if count_tokens(prefix + body) <= max_tokens:
            pieces = [body]
        else:
            # The prefix is embedded with every piece, so it comes out of the budget.
            size = min(CHUNK_TOKENS, max_tokens - count_tokens(prefix))
            pieces = make_splitter(count_tokens, size, min(CHUNK_OVERLAP_TOKENS, size // 4)).split_text(body)
        chunks.extend(Chunk(text=p, page=page, section=section, embed_text=prefix + p) for p in pieces)
    return chunks
