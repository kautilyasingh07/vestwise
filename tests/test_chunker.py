"""Chunker and loader tests (spec §12, §8.1).

Spec cases: splits on numbered headings; no chunk spans two pages; every chunk
has page and section. Plus: the fixed heading regex matches "4. Vesting", and
the build_pdfs.py footer never reaches a chunk.

Most tests count tokens as words, so they need neither the model nor `.env`.
Only test_real_chunks_fit_model_window loads the real tokenizer.
"""

import re
from pathlib import Path

import pytest

from app.ingest.chunker import HEADING_NUMBER_RE, PREAMBLE, Chunk, chunk_document, is_heading, split_sections
from app.ingest.loader import clean_page_text, load_pdf, pdf_title

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "data" / "docs"
POLICY_SRC = ROOT / "data" / "docs_src" / "esop_policy.md"
FOOTER_PATTERN = re.compile(r"\|\s*Page \d+ of \d+")


def word_count(text: str) -> int:
    """Stand-in tokenizer: one token per whitespace-separated word."""
    return len(text.split())


def chunk(pages: list[tuple[int, str]], max_tokens: int = 254, title: str = "Doc") -> list[Chunk]:
    """chunk_document with the word-count tokenizer."""
    return chunk_document(pages, title, count_tokens=word_count, max_tokens=max_tokens)


def all_pdfs() -> list[Path]:
    """Every built PDF, including the compliance drafts."""
    return sorted(DOCS_DIR.rglob("*.pdf"))


# --- heading detection ---


@pytest.mark.parametrize("line", ["4. Vesting", "4 Vesting", "4.1 Cliff", "10. Transfer Restrictions", "7.3.1 Notice"])
def test_heading_regex_matches_numbered_headings(line: str) -> None:
    assert HEADING_NUMBER_RE.match(line)


def test_spec_regex_missed_dotted_headings() -> None:
    """Regression: the original spec §8.1 regex has no `\\.?`, so "4. Vesting" never matched."""
    assert re.match(r"^\d+(\.\d+)*\s", "4. Vesting") is None
    assert HEADING_NUMBER_RE.match("4. Vesting")


@pytest.mark.parametrize("line", [
    "4. Vesting",
    "7. Termination of Employment: Good Leavers and Bad Leavers",
    "4.1 Cliff",
    "DEFINITIONS",
    "TAX",
    "SCHEDULE A - VESTING TABLE",
])
def test_is_heading_accepts_headings(line: str) -> None:
    assert is_heading(line)


@pytest.mark.parametrize("line", [
    "48 months",                                    # table cell: lowercase after the number
    "1 January 2026 (cliff)",                       # table cell: a date
    "15 March 2027",
    "4,800",
    "15 November 2024. The total number of Options outstanding and exercised under this Plan may never",
    '2.3 "Share" means one equity share of the Company with a face value of Rs 10.',
    "3.5 Grant Letter.",                            # ends like a sentence
    "RESOLVED THAT pursuant to section 62(1)(b) of the Companies Act, 2013",
    "CIN: U29299KA2021PTC145678",
    "OR",
    "Preamble text that is ordinary prose",
])
def test_is_heading_rejects_body_lines(line: str) -> None:
    assert not is_heading(line)


# --- spec §12 cases on synthetic pages ---


def test_splits_on_numbered_headings() -> None:
    pages = [(1, "Intro line\n1. Name\nThis plan is called the Plan.\n2. Vesting\nOptions vest monthly.")]
    chunks = chunk(pages)
    assert [c.section for c in chunks] == [PREAMBLE, "1. Name", "2. Vesting"]
    assert chunks[2].text == "Options vest monthly."


def test_splits_on_all_caps_headings() -> None:
    pages = [(1, "DEFINITIONS\nOption means a right.\nTAX\nTax is due on exercise.")]
    assert [c.section for c in chunk(pages)] == ["DEFINITIONS", "TAX"]


def test_no_chunk_spans_two_pages_and_section_carries_over() -> None:
    pages = [
        (1, "4. Vesting\n4.1 Options vest over 48 months."),
        (2, "4.2 The cliff is 12 months.\n5. Exercise\n5.1 Pay the strike price."),
    ]
    got = [(c.page, c.section, c.text) for c in chunk(pages)]
    assert got == [
        (1, "4. Vesting", "4.1 Options vest over 48 months."),
        (2, "4. Vesting", "4.2 The cliff is 12 months."),
        (2, "5. Exercise", "5.1 Pay the strike price."),
    ]


def test_every_chunk_has_page_section_and_embed_text() -> None:
    pages = [(1, "Preamble words.\n1. Name\nBody one."), (2, "2. Pool\nBody two.")]
    for c in chunk(pages, title="ESOP Policy"):
        assert c.page in (1, 2)
        assert c.section
        assert c.embed_text == f"ESOP Policy > {c.section}\n{c.text}"


def test_long_section_is_split_at_clause_boundaries_within_budget() -> None:
    clauses = [f"4.{i} Clause {i}. " + " ".join(["word"] * 15) + "." for i in range(1, 9)]
    pages = [(3, "4. Vesting\n" + "\n".join(clauses))]
    chunks = chunk(pages, max_tokens=40)  # prefix "Doc > 4. Vesting" is 4 of the 40
    assert len(chunks) > 1
    for c in chunks:
        assert word_count(c.embed_text) <= 40
        assert (c.page, c.section) == (3, "4. Vesting")
        assert re.match(r"^4\.\d+ Clause", c.text), "each piece should start at a clause"
    # every clause survives the split
    joined = "\n".join(c.text for c in chunks)
    assert all(clause in joined for clause in clauses)


def test_empty_pages_produce_no_chunks() -> None:
    assert chunk([(1, ""), (2, "")]) == []


# --- footer stripping ---


def test_clean_page_text_strips_footer_and_whitespace() -> None:
    raw = "4. Vesting\n  Options   vest\tmonthly. \n\nESOP Policy  |  Page 3 of 8\n"
    assert clean_page_text(raw) == "4. Vesting\nOptions vest monthly."


@pytest.mark.parametrize("pdf", all_pdfs(), ids=lambda p: p.name)
def test_build_pdfs_footer_never_reaches_a_chunk(pdf: Path) -> None:
    title = pdf_title(pdf)
    pages = load_pdf(pdf)
    chunks = chunk(pages, title=title)
    assert chunks
    footer_prefix = f"{title} | Page"  # build_pdfs.py stamps "<title>  |  Page N of M"
    for _, text in pages:
        assert not FOOTER_PATTERN.search(text)
    for c in chunks:
        assert not FOOTER_PATTERN.search(c.text), c.text
        assert footer_prefix not in " ".join(c.text.split())


# --- real documents ---


def test_policy_sections_match_source_headings() -> None:
    expected = re.findall(r"^## (\d+\. .+)$", POLICY_SRC.read_text(encoding="utf-8"), re.MULTILINE)
    pages = load_pdf(DOCS_DIR / "esop_policy.pdf")
    sections = [s for _, s, _ in split_sections(pages) if s != PREAMBLE]
    assert list(dict.fromkeys(sections)) == expected  # in order, no extra headings
    assert len(expected) == 14


def test_real_chunks_fit_model_window() -> None:
    """With the real tokenizer, every embed_text fits all-MiniLM-L6-v2's input (nothing truncated)."""
    try:
        from app.ingest import embedder
    except Exception as exc:  # missing .env -> pydantic ValidationError when config loads
        pytest.skip(f"config/model unavailable: {exc}")
    limit = embedder.max_input_tokens()
    for pdf in sorted(DOCS_DIR.glob("*.pdf")):
        for c in chunk_document(load_pdf(pdf), pdf_title(pdf)):
            assert embedder.count_tokens(c.embed_text) <= limit, (pdf.name, c.page, c.section)
