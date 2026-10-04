"""PDF -> [(page_number, text)] (spec FR-1).

Text is extracted page by page with pypdf so every piece of text keeps the
page it came from. Whitespace is normalised and the "<title> | Page N of M"
footer stamped by scripts/build_pdfs.py is removed, so it never ends up in a
chunk or an embedding.
"""

import re
from pathlib import Path

from pypdf import PdfReader

# Matches the whole footer line, e.g. "ESOP Policy | Page 2 of 8" (after whitespace is collapsed).
FOOTER_RE = re.compile(r"^.*\|\s*Page \d+ of \d+$")

PageText = tuple[int, str]


def normalise_line(line: str) -> str:
    """Collapse runs of whitespace (spaces, tabs, NBSP) to one space and trim the ends."""
    return " ".join(line.split())


def clean_page_text(raw: str) -> str:
    """Normalise each line, drop blank lines and the page-number footer."""
    lines = (normalise_line(line) for line in raw.splitlines())
    return "\n".join(line for line in lines if line and not FOOTER_RE.match(line))


def load_pdf(path: Path) -> list[PageText]:
    """Return [(page_number, text)] for every page, 1-based (text is "" for a page with none)."""
    reader = PdfReader(path)
    return [(number, clean_page_text(page.extract_text() or "")) for number, page in enumerate(reader.pages, start=1)]


def pdf_title(path: Path) -> str:
    """Return the title from the PDF metadata, or the file stem if there is none."""
    meta = PdfReader(path).metadata
    return (meta.title if meta and meta.title else path.stem).strip()
