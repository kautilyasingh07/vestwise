"""Render the synthetic Markdown documents in data/docs_src/ to PDF (spec §10).

Run from the repo root:  python scripts/build_pdfs.py

data/docs_src/foo.md            -> data/docs/foo.pdf
data/docs_src/compliance/bar.md -> data/docs/compliance/bar.pdf

Supports only the Markdown we use: a `title:` front-matter line, `#`/`##`/`###`
headings, paragraphs, pipe tables, `**bold**`, and `<!-- pagebreak -->`.
Pages break only where the source says so; the build fails if a page
overflows, so every clause stays on the page the source puts it on.
"""

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import escape

from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Flowable, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "data" / "docs_src"
OUT_DIR = ROOT / "data" / "docs"

PAGEBREAK = "<!-- pagebreak -->"
MARGIN = 20 * mm


@dataclass
class Source:
    """A parsed Markdown file: its title (front matter) and body lines."""

    title: str
    lines: list[str] = field(default_factory=list)


def parse_source(path: Path) -> Source:
    """Split a Markdown file into its front-matter title and body lines."""
    text = path.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    if not match:
        raise ValueError(f"{path}: missing front matter with a 'title:' line")
    meta = dict(line.split(":", 1) for line in match.group(1).splitlines() if ":" in line)
    return Source(title=meta["title"].strip(), lines=text[match.end():].splitlines())


def inline(text: str) -> str:
    """Escape XML special characters, then turn **bold** into <b>bold</b>."""
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escape(text))


def make_styles() -> dict[str, ParagraphStyle]:
    """Paragraph styles for body text, headings and table cells."""
    base = getSampleStyleSheet()
    body = ParagraphStyle("Body", parent=base["Normal"], fontName="Helvetica",
                          fontSize=10, leading=13.5, spaceAfter=6)
    return {
        "h1": ParagraphStyle("H1", parent=base["Heading1"], fontName="Helvetica-Bold",
                             fontSize=15, leading=19, alignment=TA_CENTER, spaceAfter=10),
        "h2": ParagraphStyle("H2", parent=base["Heading2"], fontName="Helvetica-Bold",
                             fontSize=12, leading=15, spaceBefore=8, spaceAfter=6),
        "h3": ParagraphStyle("H3", parent=base["Heading3"], fontName="Helvetica-Bold",
                             fontSize=10.5, leading=14, spaceBefore=6, spaceAfter=4),
        "body": body,
        "cell": ParagraphStyle("Cell", parent=body, spaceAfter=0),
    }


def build_table(rows: list[str], styles: dict[str, ParagraphStyle]) -> Table:
    """Turn Markdown pipe-table lines into a reportlab Table (first row is the header)."""
    cells = [[c.strip() for c in row.strip().strip("|").split("|")] for row in rows]
    cells = [r for r in cells if not all(re.fullmatch(r":?-{3,}:?", c) for c in r)]  # drop |---|
    data = [[Paragraph(inline(c), styles["cell"]) for c in r] for r in cells]
    width = A4[0] - 2 * MARGIN
    table = Table(data, colWidths=[width / len(cells[0])] * len(cells[0]), hAlign="LEFT")
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8e8e8")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return table


def to_flowables(lines: list[str], styles: dict[str, ParagraphStyle]) -> list[Flowable]:
    """Convert body lines to flowables. Consecutive text lines form one paragraph."""
    story: list[Flowable] = []
    para: list[str] = []
    table: list[str] = []

    def flush() -> None:
        if para:
            story.append(Paragraph("<br/>".join(inline(p) for p in para), styles["body"]))
            para.clear()
        if table:
            story.extend([build_table(table, styles), Spacer(1, 8)])
            table.clear()

    for raw in lines:
        line = raw.strip()
        if line.startswith("|"):
            if para:
                flush()
            table.append(line)
            continue
        if table:
            flush()
        if not line:
            flush()
        elif line == PAGEBREAK:
            flush()
            story.append(PageBreak())
        elif heading := re.match(r"^(#{1,3})\s+(.*)$", line):
            flush()
            story.append(Paragraph(inline(heading.group(2)), styles[f"h{len(heading.group(1))}"]))
        else:
            para.append(line)
    flush()
    return story


def numbered_canvas(title: str) -> type[Canvas]:
    """Return a Canvas class that stamps '<title> | Page N of M' in each footer.

    The total M is only known after the last page, so pages are buffered and
    the footers drawn in save().
    """

    class NumberedCanvas(Canvas):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self._pages: list[dict] = []

        def showPage(self) -> None:  # noqa: N802 (reportlab API name)
            self._pages.append(dict(self.__dict__))
            self._startPage()

        def save(self) -> None:
            total = len(self._pages)
            for state in self._pages:
                self.__dict__.update(state)
                self.setFont("Helvetica", 8.5)
                self.setFillColor(colors.grey)
                self.drawCentredString(A4[0] / 2, 12 * mm,
                                       f"{title}  |  Page {self._pageNumber} of {total}")
                super().showPage()
            super().save()

    return NumberedCanvas


def render(src: Path, out: Path) -> int:
    """Render one Markdown file to PDF and return its page count.

    Raises RuntimeError if the PDF has more pages than the source declares,
    which means some page overflowed and clauses moved.
    """
    source = parse_source(src)
    expected = 1 + sum(1 for line in source.lines if line.strip() == PAGEBREAK)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(out), pagesize=A4, title=source.title, author="Nimbus Robotics Pvt Ltd",
        leftMargin=MARGIN, rightMargin=MARGIN, topMargin=MARGIN, bottomMargin=MARGIN,
        invariant=1,  # no timestamp/random ID: same source -> byte-identical PDF -> same file hash
    )
    doc.build(to_flowables(source.lines, make_styles()), canvasmaker=numbered_canvas(source.title))
    pages = len(PdfReader(out).pages)
    if pages != expected:
        raise RuntimeError(f"{src.name}: expected {expected} pages, got {pages} (a page overflowed)")
    return pages


def main() -> int:
    """Render every .md under data/docs_src/ into the mirrored path under data/docs/."""
    sources = sorted(SRC_DIR.rglob("*.md"))
    if not sources:
        print(f"No Markdown files found in {SRC_DIR}")
        return 1
    for src in sources:
        out = OUT_DIR / src.relative_to(SRC_DIR).with_suffix(".pdf")
        pages = render(src, out)
        print(f"{out.relative_to(ROOT)}: {pages} page(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
